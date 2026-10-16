// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "fetch.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#if defined(HAVE_LIBCURL)
#include <curl/curl.h>

namespace {

std::once_flag curl_init_flag;
void ensure_curl_init() {
  std::call_once(curl_init_flag, []() {
    curl_global_init(CURL_GLOBAL_DEFAULT);
  });
}

struct RangeTask {
  std::string url;
  int64_t file_offset{0};
  int64_t length{0};
  torch::Tensor tensor;  // Retain tensor reference while fetching
  char* dest_ptr{nullptr};
  int64_t bytes_written{0};
  int retries{0};
  std::chrono::steady_clock::time_point retry_after{};
  CURL* easy{nullptr};
  std::string range_header;
  std::string error_msg;
  bool is_file{false};
  bool done{false};

  ~RangeTask() {
    if (easy) {
      curl_easy_cleanup(easy);
      easy = nullptr;
    }
  }
};

size_t write_callback(char* ptr, size_t size, size_t nmemb, void* userdata) {
  size_t total = size * nmemb;
  RangeTask* task = static_cast<RangeTask*>(userdata);
  if (task->bytes_written + static_cast<int64_t>(total) > task->length) {
    task->error_msg = "server sent more than " + std::to_string(task->length) + " bytes";
    return 0;  // abort transfer with CURLE_WRITE_ERROR
  }
  std::memcpy(task->dest_ptr + task->bytes_written, ptr, total);
  task->bytes_written += total;
  return total;
}

bool is_connection_error(CURLcode res) {
  switch (res) {
    case CURLE_COULDNT_RESOLVE_PROXY:
    case CURLE_COULDNT_RESOLVE_HOST:
    case CURLE_COULDNT_CONNECT:
    case CURLE_WEIRD_SERVER_REPLY:
    case CURLE_PARTIAL_FILE:
    case CURLE_OPERATION_TIMEDOUT:
    case CURLE_SSL_CONNECT_ERROR:
    case CURLE_GOT_NOTHING:
    case CURLE_SEND_ERROR:
    case CURLE_RECV_ERROR:
#if defined(CURLE_HTTP2)
    case CURLE_HTTP2:
#endif
#if defined(CURLE_HTTP2_STREAM)
    case CURLE_HTTP2_STREAM:
#endif
      return true;
    default:
      return false;
  }
}

void init_easy_handle(RangeTask* task, int64_t stall_timeout) {
  if (!task->easy) {
    task->easy = curl_easy_init();
    if (!task->easy) {
      throw std::runtime_error("curl_easy_init failed");
    }
  }
  CURL* easy = task->easy;
  curl_easy_reset(easy);
  curl_easy_setopt(easy, CURLOPT_URL, task->url.c_str());
  task->is_file = (task->url.rfind("file://", 0) == 0);
  if (task->file_offset >= 0 && task->length > 0) {
    task->range_header = std::to_string(task->file_offset) + "-" +
                         std::to_string(task->file_offset + task->length - 1);
    curl_easy_setopt(easy, CURLOPT_RANGE, task->range_header.c_str());
  }
  curl_easy_setopt(easy, CURLOPT_WRITEFUNCTION, &write_callback);
  curl_easy_setopt(easy, CURLOPT_WRITEDATA, task);
  curl_easy_setopt(easy, CURLOPT_PRIVATE, task);
  curl_easy_setopt(easy, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(easy, CURLOPT_MAXREDIRS, 10L);
  curl_easy_setopt(easy, CURLOPT_FAILONERROR, 1L);
  curl_easy_setopt(easy, CURLOPT_NOSIGNAL, 1L);
  if (stall_timeout > 0) {
    curl_easy_setopt(easy, CURLOPT_LOW_SPEED_LIMIT, 1L);
    curl_easy_setopt(easy, CURLOPT_LOW_SPEED_TIME, stall_timeout);
    curl_easy_setopt(easy, CURLOPT_CONNECTTIMEOUT, stall_timeout);
  }
  curl_easy_setopt(easy, CURLOPT_USERAGENT, "vxpu/1.0");
}

}  // namespace

void fetch_ranges(
    const py::list& range_list,
    int64_t max_concurrent = 64,
    int64_t max_retries = 3,
    int64_t stall_timeout = 30) {
  ensure_curl_init();

  if (max_concurrent <= 0) max_concurrent = 64;
  if (max_retries < 0) max_retries = 0;
  if (stall_timeout < 0) stall_timeout = 30;

  std::vector<std::unique_ptr<RangeTask>> tasks;
  tasks.reserve(range_list.size());

  for (size_t i = 0; i < range_list.size(); ++i) {
    py::handle item = range_list[i];
    TORCH_CHECK(py::isinstance<py::sequence>(item), "each request must be a tuple or list of 5 elements");
    py::sequence seq = item.cast<py::sequence>();
    TORCH_CHECK(seq.size() == 5, "each request must be (url, offset, length, dest_tensor, dest_offset)");

    auto task = std::make_unique<RangeTask>();
    task->url = seq[0].cast<std::string>();
    task->file_offset = seq[1].cast<int64_t>();
    task->length = seq[2].cast<int64_t>();
    task->tensor = seq[3].cast<torch::Tensor>();
    int64_t dest_offset = seq[4].cast<int64_t>();

    TORCH_CHECK(task->file_offset >= 0, "offset must be non-negative");
    TORCH_CHECK(task->length >= 0, "length must be non-negative");
    TORCH_CHECK(dest_offset >= 0, "dest_offset must be non-negative");
    TORCH_CHECK(task->tensor.is_contiguous(), "destination tensor must be contiguous");
    TORCH_CHECK(task->tensor.device().is_cpu(), "destination tensor must be a CPU or pinned tensor");

    int64_t tensor_bytes = task->tensor.nbytes();
    TORCH_CHECK(dest_offset + task->length <= tensor_bytes,
                "destination range [", dest_offset, ", ", dest_offset + task->length,
                ") exceeds tensor size (", tensor_bytes, " bytes)");

    char* base = reinterpret_cast<char*>(task->tensor.data_ptr());
    task->dest_ptr = base + dest_offset;

    if (task->length == 0) {
      task->done = true;
    }

    tasks.push_back(std::move(task));
  }

  std::deque<RangeTask*> pending;
  for (const auto& task : tasks) {
    if (!task->done) {
      pending.push_back(task.get());
    }
  }

  if (pending.empty()) {
    return;
  }

  CURLM* multi = curl_multi_init();
  if (!multi) {
    throw std::runtime_error("curl_multi_init failed");
  }

  std::string first_error;
  int active_count = 0;

  {
    py::gil_scoped_release release;

    auto add_task = [&](RangeTask* task) {
      task->bytes_written = 0;
      task->error_msg.clear();
      init_easy_handle(task, stall_timeout);
      curl_multi_add_handle(multi, task->easy);
      active_count++;
    };

    while (active_count > 0 || !pending.empty()) {
      auto now = std::chrono::steady_clock::now();
      int checked = 0;
      int pending_size = static_cast<int>(pending.size());

      while (active_count < max_concurrent && checked < pending_size && first_error.empty()) {
        RangeTask* task = pending.front();
        pending.pop_front();
        checked++;
        if (task->retry_after <= now) {
          add_task(task);
        } else {
          pending.push_back(task);
        }
      }

      if (active_count == 0 && pending.empty()) {
        break;
      }

      if (active_count == 0 && !pending.empty()) {
        if (!first_error.empty()) {
          break;
        }
        // Wait until the earliest retry_after
        auto earliest = pending.front()->retry_after;
        for (const auto* t : pending) {
          if (t->retry_after < earliest) earliest = t->retry_after;
        }
        auto wait_ms = std::chrono::duration_cast<std::chrono::milliseconds>(earliest - std::chrono::steady_clock::now()).count();
        if (wait_ms > 0) {
          int numfds = 0;
          curl_multi_poll(multi, NULL, 0, static_cast<int>(std::min<int64_t>(wait_ms, 1000)), &numfds);
        }
        continue;
      }

      int still_running = 0;
      curl_multi_perform(multi, &still_running);

      int msgs_left = 0;
      CURLMsg* msg = nullptr;
      while ((msg = curl_multi_info_read(multi, &msgs_left))) {
        if (msg->msg == CURLMSG_DONE) {
          CURL* easy = msg->easy_handle;
          CURLcode res = msg->data.result;
          RangeTask* task = nullptr;
          curl_easy_getinfo(easy, CURLINFO_PRIVATE, &task);
          curl_multi_remove_handle(multi, easy);
          active_count--;

          long http_code = 0;
          curl_easy_getinfo(easy, CURLINFO_RESPONSE_CODE, &http_code);

          bool success = false;
          bool retryable = false;
          std::string err;

          if (task->is_file) {
            if (res == CURLE_OK && task->bytes_written == task->length) {
              success = true;
            } else {
              err = (res != CURLE_OK) ? curl_easy_strerror(res)
                                      : ("expected " + std::to_string(task->length) +
                                         " bytes, got " + std::to_string(task->bytes_written));
              retryable = false;
            }
          } else {
            if (res == CURLE_OK) {
              if (task->file_offset > 0 && http_code != 206) {
                err = "expected HTTP 206 Partial Content, got " + std::to_string(http_code);
                if (http_code >= 500 && http_code < 600) retryable = true;
              } else if (http_code != 200 && http_code != 206) {
                err = "HTTP error " + std::to_string(http_code);
                if (http_code >= 500 && http_code < 600) retryable = true;
              } else if (task->bytes_written != task->length) {
                err = "expected " + std::to_string(task->length) + " bytes, got " +
                      std::to_string(task->bytes_written);
                retryable = true;
              } else {
                success = true;
              }
            } else if (res == CURLE_HTTP_RETURNED_ERROR) {
              err = "HTTP error " + std::to_string(http_code);
              if (http_code >= 500 && http_code < 600) retryable = true;
            } else {
              err = !task->error_msg.empty() ? task->error_msg : curl_easy_strerror(res);
              retryable = is_connection_error(res);
            }
          }

          if (success) {
            task->done = true;
          } else {
            if (retryable && task->retries < max_retries && first_error.empty()) {
              task->retries++;
              int backoff_ms = 50 * (1 << std::min(task->retries - 1, 6));
              task->retry_after = std::chrono::steady_clock::now() + std::chrono::milliseconds(backoff_ms);
              pending.push_back(task);
            } else {
              task->done = true;
              if (first_error.empty()) {
                first_error = err + " fetching " + task->url + " range " + task->range_header;
              }
            }
          }
        }
      }

      if (still_running > 0) {
        int numfds = 0;
        curl_multi_poll(multi, NULL, 0, 1000, &numfds);
      }
    }

    curl_multi_cleanup(multi);
  }

  if (!first_error.empty()) {
    throw std::runtime_error(first_error);
  }
}

void register_fetch(py::module_& m) {
  m.def("fetch_ranges", &fetch_ranges,
        py::arg("requests"),
        py::arg("max_concurrent") = 64,
        py::arg("max_retries") = 3,
        py::arg("stall_timeout") = 30,
        "Fetch weight ranges with libcurl multi straight into destination tensors (no GIL, native TLS)");
  m.def("has_fetch", []() { return true; }, "whether native libcurl range fetcher is compiled in");
}

#else  // !defined(HAVE_LIBCURL)

void register_fetch(py::module_& m) {
  m.def("has_fetch", []() { return false; }, "whether native libcurl range fetcher is compiled in");
  m.def("fetch_ranges", [](const py::list&, int64_t, int64_t, int64_t) {
    throw std::runtime_error("vxpu extension was built without libcurl support");
  }, py::arg("requests"), py::arg("max_concurrent") = 64, py::arg("max_retries") = 3, py::arg("stall_timeout") = 30);
}

#endif  // HAVE_LIBCURL
