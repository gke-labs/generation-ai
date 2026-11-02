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

// The native data path of a decode step's host nodes: no Python between
// messages.
//
//   NodeServer  (#394) dataflow nodes, as AOTInductor packages, answering
//               their mailboxes over TCP. Python builds the packages and
//               loads the nodes' residents (nodes.py, nodeserver.py), then
//               starts the server; from there, a message is read, run and
//               answered by the threads here.
//   NodeClient  the engine's side of a tcp:// mailbox.
//   RangeServer (#387) a range of whole layers on this machine, for the
//               engine running the layers before it: a decode step's
//               message replays the range's captured step graph (and its
//               step host answers the range's own host nodes), with no
//               Python; prefill, a session switch and the rest go to Python.
//   StepHost    (#409) the engine's side of the step graph's mailboxes
//               (stepgraph.py): a thread that waits for each host node's
//               flag from the GPU, runs the node (its native package here,
//               or a node server's through a NodeClient) on the pinned
//               inputs, writes the reply into the pinned outbox and raises
//               the flag back, timing each step of the crossing.
//
// The wire format, both ways, little-endian (tcp:// mailboxes):
//
//   request   "VXN1"  u32 node  u32 count  count x tensor
//   reply     "VXR2"  u32 status (0: ok)   then, ok: u32 count, count x tensor,
//                                                    i64 read_ns, i64 run_ns
//                                          else: u32 length, the error text
//   tensor    u8 dtype (c10::ScalarType)  u8 ndim  ndim x i64 size  raw bytes
//
// A node is called with the message's tensors and then its residents, the
// tensors it reads that never travel (the banks of an experts node, the
// table of an n-gram node), and replies with the package's outputs. The
// reply says how long the server took to read the request (from its first
// byte) and to run the node, so the caller can tell the wire from the
// work (#409). Node kEchoNode replies with the request's tensors: the
// round trip with no work, for measuring the wire.

#include <torch/extension.h>
#include <algorithm>
#include <map>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <dlfcn.h>

#include <atomic>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace {

constexpr uint32_t kMaxTensors = 64;
constexpr uint32_t kEchoNode = 0xFFFFFFFF;
constexpr uint8_t kMaxDims = 8;
constexpr int64_t kMaxBytes = int64_t(1) << 32;  // a message's tensor; a decode step's are kilobytes

void read_exact(int fd, void* buf, size_t n) {
  auto* p = static_cast<char*>(buf);
  while (n > 0) {
    ssize_t r = ::recv(fd, p, n, 0);
    if (r <= 0) throw std::runtime_error("connection closed");
    p += r;
    n -= static_cast<size_t>(r);
  }
}

void write_exact(int fd, const void* buf, size_t n) {
  auto* p = static_cast<const char*>(buf);
  while (n > 0) {
    ssize_t w = ::send(fd, p, n, MSG_NOSIGNAL);
    if (w <= 0) throw std::runtime_error("connection closed");
    p += w;
    n -= static_cast<size_t>(w);
  }
}

template <typename T>
T read_value(int fd) {
  T v;
  read_exact(fd, &v, sizeof(v));
  return v;
}

// A message, assembled whole and sent with one send(): written field by
// field, a message left as ~20 small packets (TCP_NODELAY), and the
// receiver paid for each.
class Out {
 public:
  void bytes(const void* p, size_t n) {
    auto* c = static_cast<const char*>(p);
    buf_.insert(buf_.end(), c, c + n);
  }

  template <typename T>
  void value(T v) {
    bytes(&v, sizeof(v));
  }

  void tensor(const at::Tensor& t0) {
    auto t = t0.contiguous().cpu();
    value<uint8_t>(static_cast<uint8_t>(t.scalar_type()));
    value<uint8_t>(static_cast<uint8_t>(t.dim()));
    for (auto s : t.sizes()) value<int64_t>(s);
    bytes(t.data_ptr(), static_cast<size_t>(t.numel() * t.element_size()));
  }

  void send(int fd) {
    write_exact(fd, buf_.data(), buf_.size());
    buf_.clear();
  }

 private:
  std::vector<char> buf_;
};

int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

at::Tensor read_tensor(int fd) {
  uint8_t code = read_value<uint8_t>(fd);
  if (code >= static_cast<uint8_t>(c10::ScalarType::NumOptions)) throw std::runtime_error("not a dtype");
  auto dtype = static_cast<c10::ScalarType>(code);
  uint8_t ndim = read_value<uint8_t>(fd);
  if (ndim > kMaxDims) throw std::runtime_error("too many dimensions");
  std::vector<int64_t> sizes(ndim);
  int64_t numel = 1;
  for (auto& s : sizes) {
    s = read_value<int64_t>(fd);
    if (s < 0) throw std::runtime_error("a negative size");
    if (s != 0 && numel > kMaxBytes / s) throw std::runtime_error("a tensor larger than the limit");
    numel *= s;
  }
  // checked before allocating: the sizes come from the peer
  int64_t bytes = numel * static_cast<int64_t>(c10::elementSize(dtype));
  if (bytes > kMaxBytes) throw std::runtime_error("a tensor larger than the limit");
  auto t = at::empty(sizes, at::TensorOptions().dtype(dtype));
  read_exact(fd, t.data_ptr(), static_cast<size_t>(bytes));
  return t;
}

struct Node {
  std::unique_ptr<torch::inductor::AOTIModelPackageLoader> loader;
  std::vector<at::Tensor> residents;
  std::mutex mu;  // one call of a node at a time: its package's buffers are its own
};

}  // namespace

// NodeServer serves nodes (by index: the order they were given in) on a
// port, until stopped. The accept loop and each connection run on their
// own threads, without the GIL.
// Connections are the connections a server is answering, each on a
// thread of its own. stop() ends them and waits for their threads: a
// detached thread would go on using the server after it is freed.
class Connections {
 public:
  // start runs serve(conn) on a thread; conn is closed when it returns.
  void start(int conn, std::function<void(int)> serve) {
    std::lock_guard<std::mutex> lock(mu_);
    if (stopped_) {
      ::close(conn);
      return;
    }
    open_.insert(conn);
    threads_.emplace_back([this, conn, serve = std::move(serve)] {
      serve(conn);
      std::lock_guard<std::mutex> lock(mu_);  // closed under the lock: stop() never shuts down a reused fd
      open_.erase(conn);
      ::close(conn);
    });
  }

  void stop() {
    std::vector<std::thread> threads;
    {
      std::lock_guard<std::mutex> lock(mu_);
      stopped_ = true;
      for (int fd : open_) ::shutdown(fd, SHUT_RDWR);
      threads.swap(threads_);
    }
    // a connection may be waiting on the GIL (a server whose nodes run in
    // Python): let it have it while we wait
    std::optional<pybind11::gil_scoped_release> release;
    if (PyGILState_Check()) release.emplace();
    for (auto& t : threads) t.join();
  }

 private:
  std::mutex mu_;
  bool stopped_ = false;
  std::set<int> open_;
  std::vector<std::thread> threads_;
};

class NodeServer {
 public:
  NodeServer(int64_t port, const std::vector<std::string>& packages,
             const std::vector<std::vector<at::Tensor>>& residents) {
    TORCH_CHECK(packages.size() == residents.size(), "a residents list per package");
    for (size_t i = 0; i < packages.size(); ++i) {
      auto node = std::make_unique<Node>();
      node->loader = std::make_unique<torch::inductor::AOTIModelPackageLoader>(packages[i]);
      node->residents = residents[i];
      nodes_.push_back(std::move(node));
    }
    fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    TORCH_CHECK(fd_ >= 0, "node server: socket");
    int one = 1;
    ::setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);  // other machines' engines call it: a private network
    addr.sin_port = htons(static_cast<uint16_t>(port));
    TORCH_CHECK(::bind(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0,
                "node server: bind to port ", port, ": ", std::strerror(errno));
    TORCH_CHECK(::listen(fd_, 16) == 0, "node server: listen");
    socklen_t len = sizeof(addr);
    ::getsockname(fd_, reinterpret_cast<sockaddr*>(&addr), &len);
    port_ = ntohs(addr.sin_port);
    accept_ = std::thread([this] { accept_loop(); });
  }

  ~NodeServer() { stop(); }

  int64_t port() const { return port_; }

  int64_t messages() const { return messages_.load(); }

  void stop() {
    if (stopped_.exchange(true)) return;
    ::shutdown(fd_, SHUT_RDWR);
    ::close(fd_);
    if (accept_.joinable()) accept_.join();
    connections_.stop();
  }

 private:
  void accept_loop() {
    while (!stopped_) {
      int conn = ::accept(fd_, nullptr, nullptr);
      if (conn < 0) {
        if (stopped_) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));  // out of fds, say: not a spin
        continue;
      }
      int one = 1;
      ::setsockopt(conn, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));  // a message is small and waited on
      connections_.start(conn, [this](int c) { serve(c); });
    }
  }

  void serve(int conn) {
    try {
      for (;;) {
        char magic[4];
        read_exact(conn, magic, 4);
        const int64_t t0 = now_ns();
        if (std::memcmp(magic, "VXN1", 4) != 0) throw std::runtime_error("not a node message");
        uint32_t index = read_value<uint32_t>(conn);
        uint32_t count = read_value<uint32_t>(conn);
        if (count > kMaxTensors) throw std::runtime_error("too many tensors");
        std::vector<at::Tensor> inputs;
        for (uint32_t i = 0; i < count; ++i) inputs.push_back(read_tensor(conn));
        const int64_t t1 = now_ns();
        std::vector<at::Tensor> outputs;
        std::string error;
        try {
          if (index == kEchoNode) {
            outputs = inputs;
          } else {
            if (index >= nodes_.size()) throw std::runtime_error("no such node");
            Node& node = *nodes_[index];
            inputs.insert(inputs.end(), node.residents.begin(), node.residents.end());
            std::lock_guard<std::mutex> lock(node.mu);
            outputs = node.loader->run(inputs);
          }
        } catch (const std::exception& e) {
          error = e.what();
        }
        const int64_t t2 = now_ns();
        Out reply;
        reply.bytes("VXR2", 4);
        if (!error.empty()) {
          reply.value<uint32_t>(1);
          reply.value<uint32_t>(static_cast<uint32_t>(error.size()));
          reply.bytes(error.data(), error.size());
          reply.send(conn);
          continue;
        }
        reply.value<uint32_t>(0);
        reply.value<uint32_t>(static_cast<uint32_t>(outputs.size()));
        for (const auto& t : outputs) reply.tensor(t);
        reply.value<int64_t>(t1 - t0);
        reply.value<int64_t>(t2 - t1);
        reply.send(conn);
        messages_++;
      }
    } catch (const std::exception&) {
      // the peer went away, or sent something that is not a message: drop it
    }
  }

  std::vector<std::unique_ptr<Node>> nodes_;
  int fd_ = -1;
  int64_t port_ = 0;
  std::thread accept_;
  Connections connections_;
  std::atomic<bool> stopped_{false};
  std::atomic<int64_t> messages_{0};
};

// NodeClient calls a node server's nodes: one connection, kept, one call
// at a time; the engine's side of a tcp:// mailbox.
// A node client is one connection to a node server. A call that fails
// (the server gone, or silent for kCallTimeoutSec) fails the step and
// closes the connection, and the next call connects again: a hung or
// restarted server must not wedge the step host that waits on it.
class NodeClient {
 public:
  NodeClient(const std::string& host, int64_t port) : host_(host), port_(port) { connect(); }

  ~NodeClient() {
    if (fd_ >= 0) ::close(fd_);
  }

  std::vector<at::Tensor> call(int64_t node, const std::vector<at::Tensor>& inputs) {
    std::lock_guard<std::mutex> lock(mu_);
    if (fd_ < 0) connect();
    try {
      return exchange(node, inputs);
    } catch (...) {
      ::close(fd_);  // mid-message, or the server's state unknown: start over
      fd_ = -1;
      throw;
    }
  }

  // The last call's server times: reading the request, running the node (ns).
  std::pair<int64_t, int64_t> server_times() const { return {read_ns_, run_ns_}; }

 private:
  // a long prefill's experts on a CPU take seconds; a server silent for
  // two minutes is gone
  static constexpr int kCallTimeoutSec = 120;

  void connect() {
    const std::string& host = host_;
    const int64_t port = port_;
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* found = nullptr;
    int rc = ::getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &found);
    TORCH_CHECK(rc == 0, "node client: resolving ", host, ": ", ::gai_strerror(rc));
    for (addrinfo* a = found; a != nullptr; a = a->ai_next) {
      fd_ = ::socket(a->ai_family, a->ai_socktype, a->ai_protocol);
      if (fd_ >= 0 && ::connect(fd_, a->ai_addr, a->ai_addrlen) == 0) break;
      if (fd_ >= 0) ::close(fd_);
      fd_ = -1;
    }
    ::freeaddrinfo(found);
    TORCH_CHECK(fd_ >= 0, "node client: cannot connect to ", host, ":", port);
    int one = 1;
    ::setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    timeval timeout{kCallTimeoutSec, 0};
    ::setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    ::setsockopt(fd_, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
  }

  std::vector<at::Tensor> exchange(int64_t node, const std::vector<at::Tensor>& inputs) {
    Out request;
    request.bytes("VXN1", 4);
    request.value<uint32_t>(static_cast<uint32_t>(node));
    request.value<uint32_t>(static_cast<uint32_t>(inputs.size()));
    for (const auto& t : inputs) request.tensor(t);
    request.send(fd_);
    char magic[4];
    read_exact(fd_, magic, 4);
    TORCH_CHECK(std::memcmp(magic, "VXR2", 4) == 0, "node client: not a node server reply (VXR2)");
    if (read_value<uint32_t>(fd_) != 0) {
      uint32_t length = read_value<uint32_t>(fd_);
      std::string error(length, '\0');
      read_exact(fd_, error.data(), length);
      TORCH_CHECK(false, "node ", node, ": ", error);
    }
    uint32_t count = read_value<uint32_t>(fd_);
    TORCH_CHECK(count <= kMaxTensors, "node client: too many tensors");
    std::vector<at::Tensor> outputs;
    for (uint32_t i = 0; i < count; ++i) outputs.push_back(read_tensor(fd_));
    read_ns_ = read_value<int64_t>(fd_);
    run_ns_ = read_value<int64_t>(fd_);
    return outputs;
  }

  std::string host_;
  int64_t port_;
  int fd_ = -1;
  std::mutex mu_;
  int64_t read_ns_ = 0, run_ns_ = 0;
};

namespace {

void spin_pause() {
#if defined(__x86_64__)
  _mm_pause();
#else
  std::this_thread::yield();
#endif
}

}  // namespace

// StepHost answers the step graph's host nodes, in step order, once per
// step. Host node i owns two int32 flags in pinned memory the GPU can
// see: flags[2i] (to the CPU: its inputs are in the pinned inboxes) and
// flags[2i+1] (from the CPU: the reply is in the pinned outbox). Each
// node is answered by its own native package (local: residents passed by
// address, as a node server runs it) or by a node server (remote: a
// NodeClient and the server's node index). Per node and step it records,
// in steady-clock ns: when it saw the GPU's flag, when the reply was in
// the outbox, and when it raised the flag back; for a remote node also
// the server's own times (reading the request, running the node).
class StepHost {
 public:
  explicit StepHost(at::Tensor flags) : flags_(std::move(flags)) {
    TORCH_CHECK(flags_.dtype() == at::kInt && flags_.is_contiguous() && flags_.is_pinned(),
                "step host: the flags must be pinned contiguous int32");
  }

  ~StepHost() { stop(); }

  void add_local(const std::vector<at::Tensor>& inputs, const at::Tensor& outbox, const std::string& package,
                 const std::vector<at::Tensor>& residents) {
    auto box = make_box(inputs, outbox);
    box->loader = std::make_unique<torch::inductor::AOTIModelPackageLoader>(package);
    box->residents = residents;
    boxes_.push_back(std::move(box));
  }

  // prefix: tensors sent before the inputs, set from Python between steps
  // (a range's session, #387)
  void add_remote(const std::vector<at::Tensor>& inputs, const at::Tensor& outbox,
                  std::shared_ptr<NodeClient> client, int64_t node, const std::vector<at::Tensor>& prefix) {
    auto box = make_box(inputs, outbox);
    box->client = std::move(client);
    box->node = node;
    box->prefix = prefix;
    boxes_.push_back(std::move(box));
  }

  void start() {
    TORCH_CHECK(!thread_.joinable(), "step host: already started");
    TORCH_CHECK(flags_.numel() >= 2 * static_cast<int64_t>(boxes_.size()), "step host: two flags per host node");
    times_ = at::zeros({static_cast<int64_t>(boxes_.size()), 5}, at::kLong);
    thread_ = std::thread([this] { loop(); });
  }

  // One step: the GPU is about to raise the first flag.
  void go() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      ++steps_requested_;
    }
    cv_.notify_one();
  }

  // After the GPU finished the step: the error of the step's failed node
  // ("" if none). A failure raises every from-CPU flag, so the GPU never
  // waits forever; the caller then clears the flags before the next step.
  std::string take_error() {
    std::lock_guard<std::mutex> lock(mu_);
    std::string e;
    std::swap(e, error_);
    return e;
  }

  // [nodes, 5] int64: per host node of the last step, when the GPU's flag
  // was seen, when the reply was ready, when the flag was raised back
  // (steady-clock ns), then for a remote node the server's read and run
  // times (ns), for a local node when its package returned (and 0).
  at::Tensor times() const { return times_; }

  bool running() const { return thread_.joinable(); }

  void stop() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stopping_ = true;
    }
    halt_ = true;  // also ends a wait on a GPU flag that will never come (a step the GPU abandoned)
    cv_.notify_one();
    if (thread_.joinable()) thread_.join();
  }

 private:
  struct Box {
    std::vector<at::Tensor> inputs;  // pinned, filled by the GPU
    at::Tensor outbox;               // pinned, read by the GPU
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> loader;
    std::vector<at::Tensor> residents;
    std::shared_ptr<NodeClient> client;
    int64_t node = 0;
    std::vector<at::Tensor> prefix;
  };

  std::unique_ptr<Box> make_box(const std::vector<at::Tensor>& inputs, const at::Tensor& outbox) {
    for (const auto& t : inputs) TORCH_CHECK(t.is_pinned(), "step host: inputs must be pinned");
    TORCH_CHECK(outbox.is_pinned() && outbox.is_contiguous(), "step host: the outbox must be pinned and contiguous");
    auto box = std::make_unique<Box>();
    box->inputs = inputs;
    box->outbox = outbox;
    return box;
  }

  void loop() {
    volatile int32_t* flag = flags_.data_ptr<int32_t>();
    int64_t* t = times_.data_ptr<int64_t>();
    for (;;) {
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [this] { return stopping_ || steps_done_ < steps_requested_; });
        if (stopping_) return;
      }
      size_t i = 0;
      try {
        for (; i < boxes_.size(); ++i) {
          Box& box = *boxes_[i];
          while (flag[2 * i] != 1) {
            if (halt_.load(std::memory_order_relaxed)) return;
            spin_pause();
          }
          flag[2 * i] = 0;
          t[5 * i] = now_ns();
          std::vector<at::Tensor> out;
          if (box.client) {
            if (box.prefix.empty()) {
              out = box.client->call(box.node, box.inputs);
            } else {
              std::vector<at::Tensor> message(box.prefix);
              message.insert(message.end(), box.inputs.begin(), box.inputs.end());
              out = box.client->call(box.node, message);
            }
            auto [read_ns, run_ns] = box.client->server_times();
            t[5 * i + 3] = read_ns;
            t[5 * i + 4] = run_ns;
          } else {
            std::vector<at::Tensor> args(box.inputs);
            args.insert(args.end(), box.residents.begin(), box.residents.end());
            out = box.loader->run(args);
            t[5 * i + 3] = now_ns();  // a local node: when its package returned (the reply's copy follows)
          }
          TORCH_CHECK(!out.empty(), "host node ", i, ": no reply");
          {
            // a memcpy, not copy_: torch runs a copy of more than 32k elements on OpenMP's workers,
            // whose wake from the host thread costs 5 to 10 ms on the virtual machine; a decode
            // step's reply is 6k elements a row (serial under 6 rows, parallel above)
            at::Tensor reply = out[0].view(box.outbox.sizes());
            if (reply.is_contiguous() && box.outbox.is_contiguous() && reply.scalar_type() == box.outbox.scalar_type() &&
                reply.device().is_cpu() && box.outbox.device().is_cpu()) {
              std::memcpy(box.outbox.data_ptr(), reply.data_ptr(), static_cast<size_t>(reply.numel() * reply.element_size()));
            } else {
              box.outbox.copy_(reply);
            }
          }
          t[5 * i + 1] = now_ns();
          t[5 * i + 2] = now_ns();  // as the flag goes up: before it, so the step's reader never sees it unwritten
          std::atomic_thread_fence(std::memory_order_seq_cst);  // the reply (and the times) before the flag
          flag[2 * i + 1] = 1;
        }
      } catch (const std::exception& e) {
        {
          std::lock_guard<std::mutex> lock(mu_);
          error_ = "host node " + std::to_string(i) + ": " + e.what();
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
        for (size_t k = 0; k < boxes_.size(); ++k) flag[2 * k + 1] = 1;  // release the GPU
      }
      std::lock_guard<std::mutex> lock(mu_);
      ++steps_done_;
    }
  }

  at::Tensor flags_;
  at::Tensor times_;
  std::vector<std::unique_ptr<Box>> boxes_;
  std::thread thread_;
  std::mutex mu_;
  std::condition_variable cv_;
  int64_t steps_requested_ = 0, steps_done_ = 0;
  bool stopping_ = false;
  std::atomic<bool> halt_{false};
  std::string error_;
};


namespace {

// The CUDA driver's calls a range server makes, found at run time: the
// kernels are built without CUDA, as the step graph's own driver calls
// are made from Python (stepgraph.py). A runtime handle (a cudaGraphExec_t,
// a cudaStream_t) is the driver's.
struct Driver {
  int (*graph_launch)(void*, void*) = nullptr;
  int (*memcpy_async)(uint64_t, uint64_t, size_t, void*) = nullptr;
  int (*stream_sync)(void*) = nullptr;
  int (*primary_ctx_retain)(void**, int) = nullptr;
  int (*ctx_set_current)(void*) = nullptr;
  int (*event_create)(void**, unsigned) = nullptr;
  int (*event_record)(void*, void*) = nullptr;
  int (*stream_wait_event)(void*, void*, unsigned) = nullptr;
  int (*stream_wait_value)(void*, uint64_t, uint32_t, unsigned) = nullptr;   // cuStreamWaitValue32_v2
  int (*stream_write_value)(void*, uint64_t, uint32_t, unsigned) = nullptr;  // cuStreamWriteValue32_v2
  int (*mem_host_alloc)(void**, size_t, unsigned) = nullptr;                 // cuMemHostAlloc
  int (*mem_host_device_pointer)(uint64_t*, void*, unsigned) = nullptr;      // cuMemHostGetDevicePointer_v2

  static Driver& get() {
    static Driver d = [] {
      Driver d;
      void* lib = ::dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
      TORCH_CHECK(lib != nullptr, "range server: no CUDA driver (libcuda.so.1)");
      auto sym = [lib](const char* name) {
        void* p = ::dlsym(lib, name);
        TORCH_CHECK(p != nullptr, "range server: the CUDA driver lacks ", name);
        return p;
      };
      d.graph_launch = reinterpret_cast<int (*)(void*, void*)>(sym("cuGraphLaunch"));
      d.memcpy_async = reinterpret_cast<int (*)(uint64_t, uint64_t, size_t, void*)>(sym("cuMemcpyAsync"));
      d.stream_sync = reinterpret_cast<int (*)(void*)>(sym("cuStreamSynchronize"));
      d.primary_ctx_retain = reinterpret_cast<int (*)(void**, int)>(sym("cuDevicePrimaryCtxRetain"));
      d.ctx_set_current = reinterpret_cast<int (*)(void*)>(sym("cuCtxSetCurrent"));
      d.event_create = reinterpret_cast<int (*)(void**, unsigned)>(sym("cuEventCreate"));
      d.event_record = reinterpret_cast<int (*)(void*, void*)>(sym("cuEventRecord"));
      d.stream_wait_event = reinterpret_cast<int (*)(void*, void*, unsigned)>(sym("cuStreamWaitEvent"));
      d.stream_wait_value = reinterpret_cast<int (*)(void*, uint64_t, uint32_t, unsigned)>(sym("cuStreamWaitValue32_v2"));
      d.stream_write_value = reinterpret_cast<int (*)(void*, uint64_t, uint32_t, unsigned)>(sym("cuStreamWriteValue32_v2"));
      d.mem_host_alloc = reinterpret_cast<int (*)(void**, size_t, unsigned)>(sym("cuMemHostAlloc"));
      d.mem_host_device_pointer = reinterpret_cast<int (*)(uint64_t*, void*, unsigned)>(sym("cuMemHostGetDevicePointer_v2"));
      return d;
    }();
    return d;
  }

  static void check(int status, const char* what) {
    TORCH_CHECK(status == 0, "range server: ", what, " failed (CUresult ", status, ")");
  }
};

// Turnstile: with several batches in flight, each GPU runs the batches' graphs one at a time,
// in the order their loops launched them (VXPU_PIPELINE_TURNSTILE=1; hero run 13). A step's
// launch takes one ticket per GPU under a host mutex; the GPU's stream waits for the GPU's
// counter to reach the ticket before its graph (cuStreamWaitValue32, no SM busy) and advances
// the counter after it. Without it two batches' graphs share a GPU's SMs at once and both run
// slower than one after the other. The counters live in mapped pinned memory, one 32-bit word
// per GPU; a loop that does not step takes no ticket, so an idle batch blocks nobody.
struct Turnstile {
  std::mutex mu;
  uint32_t* host = nullptr;
  uint64_t device = 0;  // the device pointer of `host` (portable mapped memory: one address for every GPU)
  std::vector<uint32_t> next;
  bool enabled = false;
  static Turnstile& get() {
    static Turnstile t;  // a mutex is not copyable: built in place, the switch read once
    static const bool init = [] {
      const char* env = std::getenv("VXPU_PIPELINE_TURNSTILE");
      t.enabled = env != nullptr && *env != 0 && *env != '0';
      return true;
    }();
    (void)init;
    return t;
  }
  void ensure(size_t gpus) {  // with mu held
    if (host == nullptr) {
      auto& cu = Driver::get();
      void* p = nullptr;
      Driver::check(cu.mem_host_alloc(&p, 64 * sizeof(uint32_t), 0x01 | 0x02), "cuMemHostAlloc");  // portable, device-mapped
      std::memset(p, 0, 64 * sizeof(uint32_t));
      host = static_cast<uint32_t*>(p);
      Driver::check(cu.mem_host_device_pointer(&device, p, 0), "cuMemHostGetDevicePointer");
      next.assign(64, 0);
    }
    TORCH_CHECK(gpus <= 64, "turnstile: more than 64 GPUs");
  }
};

}  // namespace

// RangeServer runs a range of whole layers for the engine before it, over
// one tcp:// mailbox in the node wire format. Every message starts with its
// session (the engine's id, as bytes). Node 0 is a decode step: with the
// range's step graph captured (set_decode) and the session the one bound,
// the inputs are copied into the graph's input buffers, the range's step
// host is told to go, the graphs replay, and the outputs go back, all here.
// Anything else (prefill, a decode step for another session, the steps
// before the capture, a reset, a description) is handed to Python's
// fallback(node, tensors) -> tensors, with the GIL.
class RangeServer {
 public:
  RangeServer(int64_t port, pybind11::function fallback) : fallback_(std::move(fallback)) {
    fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    TORCH_CHECK(fd_ >= 0, "range server: socket");
    int one = 1;
    ::setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(static_cast<uint16_t>(port));
    TORCH_CHECK(::bind(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0,
                "range server: bind to port ", port, ": ", std::strerror(errno));
    TORCH_CHECK(::listen(fd_, 16) == 0, "range server: listen");
    socklen_t len = sizeof(addr);
    ::getsockname(fd_, reinterpret_cast<sockaddr*>(&addr), &len);
    port_ = ntohs(addr.sin_port);
    accept_ = std::thread([this] { accept_loop(); });
  }

  ~RangeServer() { stop(); }

  int64_t port() const { return port_; }
  int64_t messages() const { return messages_.load(); }
  int64_t native_steps() const { return native_.load(); }

  // The captured decode step: its input buffers (on the GPU, in the order
  // a decode message carries them after the session), its outputs (the
  // graph's own buffers), the graphs to replay (cudaGraphExec_t, the
  // engine's GPU first) with each one's stream and device, the step host
  // and its flags. host: the StepHost object, kept alive here. Called
  // from the fallback after a decode step, which already holds the step
  // lock (taking it here deadlocked the first GPU run).
  void set_decode(const std::vector<at::Tensor>& inputs, const std::vector<at::Tensor>& outputs,
                  const std::vector<int64_t>& execs, const std::vector<int64_t>& streams,
                  const std::vector<int64_t>& devices, pybind11::object host, at::Tensor flags) {
    decode_inputs_ = inputs;
    decode_outputs_ = outputs;
    execs_ = execs;
    streams_ = streams;
    devices_ = devices;
    ctxs_.clear();  // each GPU's primary context, retained once (the driver counts retains)
    for (int64_t d : devices_) {
      void* c = nullptr;
      Driver::check(Driver::get().primary_ctx_retain(&c, static_cast<int>(d)), "cuDevicePrimaryCtxRetain");
      ctxs_.push_back(c);
    }
    host_obj_ = std::move(host);
    host_ = host_obj_.cast<StepHost*>();
    flags_ = std::move(flags);
  }

  void stop() {
    if (stopped_.exchange(true)) return;
    ::shutdown(fd_, SHUT_RDWR);
    ::close(fd_);
    if (accept_.joinable()) accept_.join();
    connections_.stop();
  }

 private:
  void accept_loop() {
    while (!stopped_) {
      int conn = ::accept(fd_, nullptr, nullptr);
      if (conn < 0) {
        if (stopped_) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }
      int one = 1;
      ::setsockopt(conn, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
      connections_.start(conn, [this](int c) { serve(c); });
    }
  }

  static std::string bytes_of(const at::Tensor& t) {
    auto c = t.contiguous();
    return std::string(static_cast<const char*>(c.data_ptr()), static_cast<size_t>(c.numel() * c.element_size()));
  }

  std::vector<at::Tensor> native_decode(const std::vector<at::Tensor>& inputs) {
    auto& cu = Driver::get();
    TORCH_CHECK(inputs.size() == decode_inputs_.size() + 1, "range server: a decode step of ", inputs.size() - 1,
                " values, the range takes ", decode_inputs_.size());
    void* ctx = ctxs_[0];
    Driver::check(cu.ctx_set_current(ctx), "cuCtxSetCurrent");
    void* stream = reinterpret_cast<void*>(streams_[0]);
    for (size_t i = 0; i < decode_inputs_.size(); ++i) {
      auto src = inputs[i + 1].contiguous();
      auto& dst = decode_inputs_[i];
      TORCH_CHECK(src.numel() == dst.numel() && src.scalar_type() == dst.scalar_type(),
                  "range server: decode value ", i, " does not fit its buffer");
      Driver::check(cu.memcpy_async(reinterpret_cast<uint64_t>(dst.data_ptr()), reinterpret_cast<uint64_t>(src.data_ptr()),
                                    static_cast<size_t>(src.numel() * src.element_size()), stream), "cuMemcpyAsync");
    }
    host_->go();
    for (size_t g = 0; g < execs_.size(); ++g) {  // the engine's GPU's graph first: the others wait on its flags
      Driver::check(cu.ctx_set_current(ctxs_[g]), "cuCtxSetCurrent");
      Driver::check(cu.graph_launch(reinterpret_cast<void*>(execs_[g]), reinterpret_cast<void*>(streams_[g])), "cuGraphLaunch");
    }
    for (size_t g = 0; g < execs_.size(); ++g) {  // each GPU's stream under its own context: stream 0
      Driver::check(cu.ctx_set_current(ctxs_[g]), "cuCtxSetCurrent");  // (the default) names the current one's
      Driver::check(cu.stream_sync(reinterpret_cast<void*>(streams_[g])), "cuStreamSynchronize");
    }
    Driver::check(cu.ctx_set_current(ctx), "cuCtxSetCurrent");
    std::string error = host_->take_error();
    if (!error.empty()) {
      flags_.zero_();  // released every flag; the next step starts clean
      throw std::runtime_error("range step host: " + error);
    }
    std::vector<at::Tensor> out;
    for (const auto& t : decode_outputs_) {
      auto host = at::empty(t.sizes(), t.options().device(at::kCPU));
      Driver::check(cu.memcpy_async(reinterpret_cast<uint64_t>(host.data_ptr()), reinterpret_cast<uint64_t>(t.data_ptr()),
                                    static_cast<size_t>(t.numel() * t.element_size()), stream), "cuMemcpyAsync");
      out.push_back(host);
    }
    Driver::check(cu.stream_sync(stream), "cuStreamSynchronize");
    native_++;
    return out;
  }

  void serve(int conn) {
    try {
      for (;;) {
        char magic[4];
        read_exact(conn, magic, 4);
        const int64_t t0 = now_ns();
        if (std::memcmp(magic, "VXN1", 4) != 0) throw std::runtime_error("not a node message");
        uint32_t node = read_value<uint32_t>(conn);
        uint32_t count = read_value<uint32_t>(conn);
        if (count > kMaxTensors) throw std::runtime_error("too many tensors");
        std::vector<at::Tensor> inputs;
        for (uint32_t i = 0; i < count; ++i) inputs.push_back(read_tensor(conn));
        const int64_t t1 = now_ns();
        std::vector<at::Tensor> outputs;
        std::string error;
        {
          std::lock_guard<std::mutex> lock(mu_);  // one step at a time: the range's state is one session's
          try {
            std::string session = inputs.empty() ? std::string() : bytes_of(inputs[0]);
            if (node == 0 && host_ != nullptr && session == bound_) {
              outputs = native_decode(inputs);
            } else {
              pybind11::gil_scoped_acquire gil;
              // Python binds the session's state before the step runs: if the
              // step then fails, the resident state is no longer bound_'s, and
              // that session's next decode must not replay natively on it
              if (node <= 1) bound_.clear();
              outputs = fallback_(node, inputs).cast<std::vector<at::Tensor>>();
              if (node <= 1) bound_ = session;  // Python bound it for the step
            }
          } catch (const std::exception& e) {
            error = e.what();
          }
        }
        const int64_t t2 = now_ns();
        Out reply;
        reply.bytes("VXR2", 4);
        if (!error.empty()) {
          reply.value<uint32_t>(1);
          reply.value<uint32_t>(static_cast<uint32_t>(error.size()));
          reply.bytes(error.data(), error.size());
          reply.send(conn);
          continue;
        }
        reply.value<uint32_t>(0);
        reply.value<uint32_t>(static_cast<uint32_t>(outputs.size()));
        for (const auto& t : outputs) reply.tensor(t);
        reply.value<int64_t>(t1 - t0);
        reply.value<int64_t>(t2 - t1);
        reply.send(conn);
        messages_++;
      }
    } catch (const std::exception&) {
      // the peer went away, or sent something that is not a message
    }
  }

  pybind11::function fallback_;
  std::mutex mu_;
  std::string bound_;
  std::vector<at::Tensor> decode_inputs_, decode_outputs_;
  std::vector<int64_t> execs_, streams_, devices_;
  std::vector<void*> ctxs_;
  pybind11::object host_obj_;
  StepHost* host_ = nullptr;
  at::Tensor flags_;
  int fd_ = -1;
  int64_t port_ = 0;
  std::thread accept_;
  Connections connections_;
  std::atomic<bool> stopped_{false};
  std::atomic<int64_t> messages_{0}, native_{0};
};

// DecodeLoop: a batch engine's rows, decoded with no Python between steps.
// The step graph is captured with the greedy choice at its end (each
// position's argmax, into `chosen`). A native thread runs every step: it
// writes each row's tokens and position into the graph's input buffers, has
// the step host go, launches the graphs, and reads the chosen tokens back.
// With steps of S tokens (speculative decoding) and the drafter's captured
// step (set_draft), each step first runs the drafter on the row's next token
// and position, then verifies its S - 1 tokens: the row keeps each draft
// that equals the choice before it, then the choice where they first
// differ, as _Rows does. A row then emits its tokens, stops on an
// end-of-turn id or its limits, or goes on from its new position. A row
// with nothing to decode feeds its last token again (whatever that writes,
// its next prefill replaces). Python admits a request between steps
// (pause, prefill the row, admit, resume), cancels one, and takes the
// emitted tokens as events.
class DecodeLoop {
 public:
  // input_ids [B, S] and positions [B] int64 on the GPU (the graph's input
  // buffers), chosen [B, S] int64 on the GPU (the graph's argmax), then the
  // graphs to launch as RangeServer::set_decode takes them. Call start()
  // (after set_draft, for speculative steps).
  DecodeLoop(at::Tensor input_ids, at::Tensor positions, at::Tensor chosen, const std::vector<int64_t>& execs,
             const std::vector<int64_t>& streams, const std::vector<int64_t>& devices, pybind11::object host,
             at::Tensor flags, int64_t out_graph = 0)
      : verify_{std::move(input_ids), std::move(positions), std::move(chosen), execs, streams, devices,
                std::move(host), std::move(flags), out_graph} {
    TORCH_CHECK(out_graph >= 0 && out_graph < static_cast<int64_t>(verify_.execs.size()),
                "decode loop: out_graph must index the graphs");
    const at::Tensor& ids = verify_.ids;
    TORCH_CHECK(ids.scalar_type() == at::kLong && ids.dim() == 2, "decode loop: input_ids must be int64 [B, S]");
    rows_n_ = ids.size(0);
    S_ = ids.size(1);
    TORCH_CHECK(verify_.pos.scalar_type() == at::kLong && verify_.pos.numel() == rows_n_,
                "decode loop: positions must be int64 [B]");
    TORCH_CHECK(verify_.out.scalar_type() == at::kLong && verify_.out.numel() == rows_n_ * S_,
                "decode loop: chosen must be int64 [B, S]");
    verify_.retain();
    rows_.resize(rows_n_);
    issued_.assign(rows_n_, 0);
    auto pinned = at::TensorOptions().dtype(at::kLong).pinned_memory(true);
    ids_host_ = at::zeros({rows_n_ * S_}, pinned);
    pos_host_ = at::zeros({rows_n_}, pinned);
    chosen_host_ = at::zeros({rows_n_ * S_}, pinned);
    // VXPU_STEP_TRACE=path: the host side of every step, a JSON line each (the launch, the
    // collect, each GPU's stream drain, each host node's start and end), for the steps that
    // read 30 or 60 ms of wall around 14 ms of GPU work
    if (const char* path = std::getenv("VXPU_STEP_TRACE"); path != nullptr && *path != 0) {
      trace_ = std::fopen(path, "a");
    }
  }

  ~DecodeLoop() {
    stop();
    if (trace_ != nullptr) std::fclose(trace_);
  }

  // The drafter's captured step: input_ids [B, 1] and positions [B] int64
  // on the GPU, its tokens out [B, K] int64 (K >= S - 1), its graphs.
  // `compose` is a graph exec on the verify's first stream's GPU that copies the drafter's
  // tokens out[:, :S-1] into the verify's input_ids[:, 1:] on the GPU (captured by
  // StepGraph.native_loop); with it the step never brings the drafts to the host before
  // the verify. 0 keeps the host path.
  // `block` is the tokens' width when out[:, block:] carries each draft's confidence (the
  // drafter's one output buffer holds both, see deepseek_v41.Draft); 0: tokens only.
  void set_draft(at::Tensor input_ids, at::Tensor positions, at::Tensor out, const std::vector<int64_t>& execs,
                 const std::vector<int64_t>& streams, const std::vector<int64_t>& devices, pybind11::object host,
                 at::Tensor flags, int64_t compose = 0, int64_t block = 0) {
    TORCH_CHECK(!thread_.joinable(), "decode loop: set_draft before start");
    TORCH_CHECK(S_ > 1, "decode loop: a drafter needs steps of several tokens");
    TORCH_CHECK(block == 0 || (block >= S_ - 1 && 2 * block <= out.size(1)),
                "decode loop: the drafter's block must cover the step's drafts and leave room for their confidence");
    draft_block_ = block;
    TORCH_CHECK(input_ids.scalar_type() == at::kLong && input_ids.numel() == rows_n_ && positions.numel() == rows_n_,
                "decode loop: the drafter takes int64 input_ids [B, 1] and positions [B]");
    TORCH_CHECK(out.scalar_type() == at::kLong && out.dim() == 2 && out.size(0) == rows_n_ && out.size(1) >= S_ - 1,
                "decode loop: the drafter's tokens must be int64 [B, K] with K >= S - 1");
    draft_ = std::make_unique<Graphs>(Graphs{std::move(input_ids), std::move(positions), std::move(out), execs, streams,
                                             devices, std::move(host), std::move(flags)});
    draft_->retain();
    draft_->out_graph = static_cast<int64_t>(std::find(devices.begin(), devices.end(), static_cast<int64_t>(draft_->out.get_device())) - devices.begin());
    TORCH_CHECK(draft_->out_graph < static_cast<int64_t>(devices.size()), "decode loop: the drafter's tokens live on none of its graphs' GPUs");
    auto pinned = at::TensorOptions().dtype(at::kLong).pinned_memory(true);
    draft_out_host_ = at::zeros({rows_n_ * draft_->out.size(1)}, pinned);
    compose_ = compose;
    if (compose_ != 0) {  // the event the verify's stream waits on: recorded after the drafter's graphs
      auto& cu = Driver::get();
      Driver::check(cu.ctx_set_current(draft_->ctxs[draft_->out_graph]), "cuCtxSetCurrent");
      Driver::check(cu.event_create(&drafted_event_, 0x2 /* CU_EVENT_DISABLE_TIMING */), "cuEventCreate");
    }
  }

  // The speculation policy (vxpu.speculation.Policy, hero run 13): how many of the drafter's
  // tokens a row issues each step. `max_draft` caps the drafts issued (-1: all S - 1); with
  // `min_score` > 0 a row stops issuing at the first draft whose confidence score (the drafter's
  // int64 score after its tokens, see set_draft's `block`) is under it. The verify graph has S
  // token rows whatever is issued (the export fixes S), so the positions past the issued drafts
  // carry the last issued token again (the first token when none is issued): a padded token the
  // verify happens to agree with is a correct greedy token, so acceptance stays exact, and the
  // statistics count only the drafts issued. The policy needs the drafts on the host before the
  // verify, so it takes the host path (no compose). Call after set_draft, before start.
  // A verify step with fewer token rows than the export's S (hero run 15): a graph captured from an
  // artifact exported with `rows` token rows, bound to the same state. A step whose rows issue at
  // most rows - 1 drafts runs through it instead of the S-row graph, so a short draft costs a
  // short step (the padded S-row verify cost 25 ms whatever was issued). Host path only (a policy
  // chooses the drafts on the host); the row count chosen a step is 1 + the most drafts any row
  // issued (vxpu.speculation.verify_rows). Call after set_draft, before start.
  void set_verify(int64_t rows, at::Tensor input_ids, at::Tensor positions, at::Tensor chosen,
                  const std::vector<int64_t>& execs, const std::vector<int64_t>& streams,
                  const std::vector<int64_t>& devices, pybind11::object host, at::Tensor flags, int64_t out_graph = 0) {
    TORCH_CHECK(!thread_.joinable(), "decode loop: set_verify before start");
    TORCH_CHECK(rows >= 1 && rows < S_, "decode loop: a shorter verify needs 1 <= rows < S");
    TORCH_CHECK(input_ids.scalar_type() == at::kLong && input_ids.dim() == 2 && input_ids.size(0) == rows_n_
                    && input_ids.size(1) == rows,
                "decode loop: a shorter verify's input_ids must be int64 [B, rows]");
    TORCH_CHECK(chosen.scalar_type() == at::kLong && chosen.numel() == rows_n_ * rows,
                "decode loop: a shorter verify's chosen must be int64 [B, rows]");
    auto g = std::make_unique<Graphs>(Graphs{std::move(input_ids), std::move(positions), std::move(chosen), execs,
                                             streams, devices, std::move(host), std::move(flags), out_graph});
    g->retain();
    verifies_[rows] = std::move(g);
  }

  void set_policy(int64_t max_draft, int64_t min_score) {
    TORCH_CHECK(!thread_.joinable(), "decode loop: set_policy before start");
    TORCH_CHECK(draft_, "decode loop: set_policy needs a drafter (set_draft first)");
    TORCH_CHECK(max_draft >= -1 && max_draft <= S_ - 1, "decode loop: max_draft must be -1 or 0 to S - 1");
    TORCH_CHECK(min_score <= 0 || (draft_block_ > 0 && draft_block_ < draft_->out.size(1)),
                "decode loop: a confidence threshold needs the drafter's confidence output (set_draft's block)");
    policy_max_draft_ = max_draft;
    policy_min_score_ = min_score;
    compose_ = 0;
  }

  // The graph's sampling settings, one per row on the GPU: temperature
  // (float32; 0 chooses the argmax), top_k (int64; 0 is off), top_p
  // (float32; 1 is off). The loop writes each row's before every step.
  void set_sampling(at::Tensor temperature, at::Tensor top_k, at::Tensor top_p) {
    TORCH_CHECK(!thread_.joinable(), "decode loop: set_sampling before start");
    TORCH_CHECK(temperature.scalar_type() == at::kFloat && temperature.numel() == rows_n_ &&
                    top_k.scalar_type() == at::kLong && top_k.numel() == rows_n_ &&
                    top_p.scalar_type() == at::kFloat && top_p.numel() == rows_n_,
                "decode loop: sampling settings are float32 temperature, int64 top_k and float32 top_p, one per row");
    temperature_ = std::move(temperature);
    top_k_ = std::move(top_k);
    top_p_ = std::move(top_p);
    temperature_host_ = at::zeros({rows_n_}, at::TensorOptions().dtype(at::kFloat).pinned_memory(true));
    top_k_host_ = at::zeros({rows_n_}, at::TensorOptions().dtype(at::kLong).pinned_memory(true));
    top_p_host_ = at::ones({rows_n_}, at::TensorOptions().dtype(at::kFloat).pinned_memory(true));
  }

  void start() {
    TORCH_CHECK(!thread_.joinable(), "decode loop: already started");
    TORCH_CHECK(S_ == 1 || draft_, "decode loop: steps of several tokens need a drafter (set_draft)");
    thread_ = std::thread([this] { loop(); });
  }

  // Hold the loop between steps (returns once no step runs).
  void pause() {
    std::unique_lock<std::mutex> lk(mu_);
    paused_ = true;
    idle_.wait(lk, [&] { return !stepping_; });
  }

  void resume() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      paused_ = false;
    }
    wake_.notify_one();
  }

  // Row `row` decodes request `request` from here: it feeds next_token at
  // `position`, has emitted `emitted` tokens of at most max_new, stops on
  // any of eos, and may not feed past max_position. Call while paused.
  void admit(int64_t row, int64_t request, int64_t next_token, int64_t position, int64_t emitted, int64_t max_new,
             const std::vector<int64_t>& eos, int64_t max_position, double temperature, int64_t top_k, double top_p) {
    std::lock_guard<std::mutex> lk(mu_);
    TORCH_CHECK(row >= 0 && row < rows_n_, "decode loop: no row ", row);
    TORCH_CHECK(!rows_[row].active, "decode loop: row ", row, " is decoding another request");
    Row& r = rows_[row];
    r = Row{};
    r.active = r.used = true;
    r.request = request;
    r.next = next_token;
    r.position = position;
    r.emitted = emitted;
    r.max_new = max_new;
    r.max_position = max_position;
    r.eos = eos;
    r.temperature = static_cast<float>(temperature);
    r.top_k = top_k;
    r.top_p = static_cast<float>(top_p);
  }

  // The request stops at the next step boundary (its row frees).
  void cancel(int64_t request) {
    std::lock_guard<std::mutex> lk(mu_);
    for (Row& r : rows_)
      if (r.active && r.request == request) finish(r, kCancelled, "");
  }

  // The rows that are free.
  std::vector<int64_t> free_rows() {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<int64_t> out;
    for (int64_t i = 0; i < rows_n_; ++i)
      if (!rows_[i].active) out.push_back(i);
    return out;
  }

  // Emitted tokens since the last call, per request: (request, tokens,
  // finish: 0 still decoding, 1 eos, 2 length, 3 cancelled, 4 error, the
  // error). Waits up to timeout_s for something to report.
  std::vector<std::tuple<int64_t, std::vector<int64_t>, int64_t, std::string>> events(double timeout_s) {
    std::unique_lock<std::mutex> lk(mu_);
    events_cv_.wait_for(lk, std::chrono::duration<double>(timeout_s), [&] { return !pending_.empty() || stop_; });
    std::vector<std::tuple<int64_t, std::vector<int64_t>, int64_t, std::string>> out;
    for (auto& [request, e] : pending_) out.emplace_back(request, std::move(e.tokens), e.finish, e.error);
    pending_.clear();
    return out;
  }

  // Steps run; since the last call, each step's time in ms (inputs written
  // to chosen tokens read, the drafter included) and how many rows decoded;
  // drafts made and accepted, in all.
  int64_t steps() const { return steps_.load(); }
  std::vector<std::pair<double, int64_t>> take_steps() {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<std::pair<double, int64_t>> out;
    std::swap(out, step_log_);
    return out;
  }
  std::pair<int64_t, int64_t> drafts() {
    std::lock_guard<std::mutex> lk(mu_);
    return {drafted_, accepted_};
  }
  // Since the last call, one record per decoding row and step: the drafts accepted, then each
  // draft's confidence as the drafter wrote it (an int64 score; -1 when the drafter gives none,
  // -2 at a position the policy did not issue).
  // [N, S] int64; hero run 12's speculation statistics (vxpu.speculation.stats_from_events).
  at::Tensor speculation_events() {
    std::lock_guard<std::mutex> lk(mu_);
    const int64_t width = S_;
    const int64_t n = static_cast<int64_t>(spec_events_.size()) / width;
    at::Tensor out = at::empty({n, width}, at::TensorOptions().dtype(at::kLong));
    std::copy(spec_events_.begin(), spec_events_.begin() + n * width, out.data_ptr<int64_t>());
    spec_events_.clear();
    return out;
  }

  void stop() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (stop_) return;
      stop_ = true;
    }
    wake_.notify_all();
    events_cv_.notify_all();
    if (thread_.joinable()) thread_.join();
    print_phases();
  }

 private:
  enum Finish : int64_t { kDecoding = 0, kEos = 1, kLength = 2, kCancelled = 3, kError = 4 };
  struct Row {
    bool active = false, used = false;
    int64_t request = -1, next = 0, position = 0, emitted = 0, max_new = 0, max_position = 0;
    int64_t fed = 0, fed_position = 0;  // the token it fed last, and where: an idle row feeds it again
    std::vector<int64_t> eos;
    float temperature = 0.f, top_p = 1.f;  // the row's sampling settings (greedy by default)
    int64_t top_k = 0;
  };
  struct Event {
    std::vector<int64_t> tokens;
    int64_t finish = kDecoding;
    std::string error;
  };
  // A captured step: its input buffers, its output, its graphs, its step host.
  struct Graphs {
    at::Tensor ids, pos, out;
    std::vector<int64_t> execs, streams, devices;
    pybind11::object host_obj;
    at::Tensor flags;
    int64_t out_graph = 0;  // the graph (and GPU) that writes `out`: the one holding the head
    std::vector<void*> ctxs;
    StepHost* host = nullptr;
    void retain() {
      host = host_obj.cast<StepHost*>();
      for (int64_t d : devices) {
        void* c = nullptr;
        Driver::check(Driver::get().primary_ctx_retain(&c, static_cast<int>(d)), "cuDevicePrimaryCtxRetain");
        ctxs.push_back(c);
      }
    }
    // The inputs (pinned host arrays) in and the graphs launched, nothing waited for. With
    // `after` (an event on another stream) and `compose` (a graph exec), the first stream
    // waits on the event and runs compose before the step: the speculative step writes the
    // drafter's tokens into this step's input ids on the GPU that way, with no host between.
    void launch(const int64_t* ids_host, size_t ids_bytes, const int64_t* pos_host, size_t pos_bytes,
                void* after = nullptr, int64_t compose = 0) {
      auto& cu = Driver::get();
      void* stream = reinterpret_cast<void*>(streams[0]);
      Driver::check(cu.ctx_set_current(ctxs[0]), "cuCtxSetCurrent");
      Driver::check(cu.memcpy_async(reinterpret_cast<uint64_t>(ids.data_ptr()), reinterpret_cast<uint64_t>(ids_host),
                                    ids_bytes, stream), "cuMemcpyAsync");
      Driver::check(cu.memcpy_async(reinterpret_cast<uint64_t>(pos.data_ptr()), reinterpret_cast<uint64_t>(pos_host),
                                    pos_bytes, stream), "cuMemcpyAsync");
      if (after != nullptr) Driver::check(cu.stream_wait_event(stream, after, 0), "cuStreamWaitEvent");
      if (compose != 0) Driver::check(cu.graph_launch(reinterpret_cast<void*>(compose), stream), "cuGraphLaunch");
      host->go();
      Turnstile& turn = Turnstile::get();
      std::vector<uint32_t> tickets;
      if (turn.enabled) {  // one ticket per GPU for this step, taken together so every GPU sees the same order
        std::lock_guard<std::mutex> lock(turn.mu);
        turn.ensure(static_cast<size_t>(*std::max_element(devices.begin(), devices.end())) + 1);
        for (int64_t d : devices) tickets.push_back(turn.next[static_cast<size_t>(d)]++);
      }
      for (size_t g = 0; g < execs.size(); ++g) {  // the engine's GPU's graph first: the others wait on its flags
        Driver::check(cu.ctx_set_current(ctxs[g]), "cuCtxSetCurrent");
        void* s = reinterpret_cast<void*>(streams[g]);
        const uint64_t counter = turn.enabled ? turn.device + sizeof(uint32_t) * static_cast<uint64_t>(devices[g]) : 0;
        if (turn.enabled)
          Driver::check(cu.stream_wait_value(s, counter, tickets[g], 0x1 /* CU_STREAM_WAIT_VALUE_EQ */), "cuStreamWaitValue32");
        Driver::check(cu.graph_launch(reinterpret_cast<void*>(execs[g]), s), "cuGraphLaunch");
        if (turn.enabled)
          Driver::check(cu.stream_write_value(s, counter, tickets[g] + 1, 0), "cuStreamWriteValue32");
      }
    }
    void* out_stream() const { return reinterpret_cast<void*>(streams[out_graph]); }
    // The output read on its own GPU's stream, after that GPU's graph (the last, with layers sharded).
    void collect(int64_t* out_host) {
      auto& cu = Driver::get();
      Driver::check(cu.ctx_set_current(ctxs[out_graph]), "cuCtxSetCurrent");
      Driver::check(cu.memcpy_async(reinterpret_cast<uint64_t>(out_host), reinterpret_cast<uint64_t>(out.data_ptr()),
                                    static_cast<size_t>(out.numel()) * sizeof(int64_t), out_stream()), "cuMemcpyAsync");
    }
    void sync(int64_t* synced_ns = nullptr) {  // synced_ns: a timestamp per GPU as its stream drains (the step trace)
      auto& cu = Driver::get();
      for (size_t g = 0; g < execs.size(); ++g) {
        Driver::check(cu.ctx_set_current(ctxs[g]), "cuCtxSetCurrent");
        Driver::check(cu.stream_sync(reinterpret_cast<void*>(streams[g])), "cuStreamSynchronize");
        if (synced_ns != nullptr) synced_ns[g] = now_ns();
      }
      Driver::check(cu.ctx_set_current(ctxs[0]), "cuCtxSetCurrent");
      check();
    }
    // The step host's verdict alone: for graphs whose streams another Graphs' sync already drained
    // (the drafter's run on the verify's streams, ahead of it).
    void check() {
      std::string error = host->take_error();
      if (!error.empty()) {
        flags.zero_();  // the host released every flag; the next step starts clean
        throw std::runtime_error(error);
      }
    }
    // One replay: the inputs in, the output back into `out_host`, waited for.
    void run(const int64_t* ids_host, size_t ids_bytes, const int64_t* pos_host, size_t pos_bytes, int64_t* out_host) {
      launch(ids_host, ids_bytes, pos_host, pos_bytes);
      collect(out_host);
      sync();
    }
  };

  void finish(Row& r, int64_t why, const std::string& error) {  // with mu_ held
    Event& e = pending_[r.request];
    e.finish = why;
    e.error = error;
    r.active = false;
    events_cv_.notify_all();
  }

  bool any_active() const {
    for (const Row& r : rows_)
      if (r.active) return true;
    return false;
  }

  // Emit `token` for row r; true once the row is finished by it.
  bool emit(Row& r, int64_t token) {  // with mu_ held
    pending_[r.request].tokens.push_back(token);
    events_cv_.notify_all();
    r.emitted++;
    if (std::find(r.eos.begin(), r.eos.end(), token) != r.eos.end()) {
      finish(r, kEos, "");
      return true;
    }
    if (r.emitted >= r.max_new) {
      finish(r, kLength, "");
      return true;
    }
    return false;
  }

  void loop() {
    int64_t* ids = ids_host_.data_ptr<int64_t>();
    int64_t* pos = pos_host_.data_ptr<int64_t>();
    int64_t* chosen = chosen_host_.data_ptr<int64_t>();
    const int64_t K = draft_ ? draft_->out.size(1) : 0;
    int64_t* drafted = draft_ ? draft_out_host_.data_ptr<int64_t>() : nullptr;
    std::vector<int64_t> first(rows_n_);  // each row's first token of the step
    for (;;) {
      int64_t decoding = 0;
      {
        std::unique_lock<std::mutex> lk(mu_);
        wake_.wait(lk, [&] { return stop_ || (!paused_ && any_active()); });
        if (stop_) return;
        stepping_ = true;
        for (int64_t i = 0; i < rows_n_; ++i) {
          Row& r = rows_[i];
          if (r.active) {
            r.fed = r.next;
            r.fed_position = r.position;
            decoding++;
          }
          first[i] = r.used ? r.fed : 0;  // an idle row feeds its last token again; a row never used, 0 at 0
          pos[i] = r.used ? r.fed_position : 0;
          if (temperature_.defined()) {
            temperature_host_.data_ptr<float>()[i] = r.active ? r.temperature : 0.f;
            top_k_host_.data_ptr<int64_t>()[i] = r.active ? r.top_k : 0;
            top_p_host_.data_ptr<float>()[i] = r.active ? r.top_p : 1.f;
          }
        }
      }
      const int64_t t0 = now_ns();
      int64_t t_drafted = 0, t_launched = 0, t_synced = 0;
      std::string error;
      Graphs* g = &verify_;  // the verify's graphs: the export's S-row step unless a shorter one fits (set_verify)
      int64_t S = S_;
      try {
        void* after = nullptr;
        if (draft_ && compose_ != 0) {  // the drafter's graphs, then its tokens copied out and an event for the verify
          if (trace_ != nullptr) tr_draft0_ = now_ns();
          draft_->launch(first.data(), rows_n_ * sizeof(int64_t), pos, rows_n_ * sizeof(int64_t));
          if (trace_ != nullptr) tr_draft1_ = now_ns();
          draft_->collect(drafted);  // read by the host after the verify, for the acceptance
          Driver::check(Driver::get().event_record(drafted_event_, draft_->out_stream()), "cuEventRecord");
          after = drafted_event_;
          for (int64_t i = 0; i < rows_n_; ++i) {  // the rest come from the GPU (compose): every draft issued
            ids[i * S_] = first[i];
            issued_[i] = S_ - 1;
          }
          t_drafted = now_ns();
        } else {
          if (draft_) {  // the drafter on each row's next token: its K tokens after it, through the host
            if (trace_ != nullptr) tr_draft0_ = now_ns();
            draft_->run(first.data(), rows_n_ * sizeof(int64_t), pos, rows_n_ * sizeof(int64_t), drafted);
            if (trace_ != nullptr) tr_draft1_ = now_ns();
          }
          int64_t most = 0;
          for (int64_t i = 0; i < rows_n_; ++i) {
            // the drafts this row issues under the policy (all S - 1 without one); the positions
            // past them carry the last issued token again
            int64_t issued = draft_ ? S_ - 1 : 0;
            if (draft_ && policy_max_draft_ >= 0) issued = std::min<int64_t>(issued, policy_max_draft_);
            if (draft_ && policy_min_score_ > 0)
              for (int64_t j = 0; j < issued; ++j)
                if (drafted[i * K + draft_block_ + j] < policy_min_score_) { issued = j; break; }
            issued_[i] = issued;
            if (rows_[i].active) most = std::max(most, issued);
          }
          // the verify sized to the drafts: the shortest registered graph with room for every
          // row's drafts (1 + the most issued), else the export's S-row graph
          for (auto it = verifies_.begin(); it != verifies_.end(); ++it)
            if (it->first >= most + 1) { g = it->second.get(); S = it->first; break; }
          for (int64_t i = 0; i < rows_n_; ++i) {
            const int64_t issued = issued_[i];
            ids[i * S] = first[i];
            for (int64_t j = 1; j < S; ++j)
              ids[i * S + j] = j <= issued ? drafted[i * K + j - 1] : (issued > 0 ? drafted[i * K + issued - 1] : first[i]);
          }
        }
        if (temperature_.defined()) {  // the rows' sampling settings, ahead of the step on its stream
          auto& cu = Driver::get();
          void* stream = reinterpret_cast<void*>(g->streams[0]);
          Driver::check(cu.ctx_set_current(g->ctxs[0]), "cuCtxSetCurrent");
          for (auto* t : {&temperature_, &top_k_, &top_p_}) {
            const at::Tensor& host = t == &temperature_ ? temperature_host_ : t == &top_k_ ? top_k_host_ : top_p_host_;
            Driver::check(cu.memcpy_async(reinterpret_cast<uint64_t>(t->data_ptr()), reinterpret_cast<uint64_t>(host.data_ptr()),
                                          static_cast<size_t>(host.numel() * host.element_size()), stream), "cuMemcpyAsync");
          }
        }
        if (trace_ != nullptr) tr_launch0_ = now_ns();
        g->launch(ids, rows_n_ * S * sizeof(int64_t), pos, rows_n_ * sizeof(int64_t), after, compose_);
        if (trace_ != nullptr) tr_launch1_ = now_ns();
        g->collect(chosen);
        t_launched = now_ns();
        if (trace_ != nullptr) {  // the drafter's graphs ran on these same streams, ahead of the verify's
          tr_synced_.assign(g->execs.size(), 0);
          g->sync(tr_synced_.data());
        } else {
          g->sync();
        }
        if (after != nullptr) draft_->check();
        if (trace_ != nullptr) tr_draft_synced_ = now_ns();
        t_synced = now_ns();
        if (after != nullptr) {  // the host's view of the step's ids: the drafts the GPU verified
          for (int64_t i = 0; i < rows_n_; ++i)
            for (int64_t j = 1; j < S_; ++j) ids[i * S_ + j] = drafted[i * K + j - 1];
        }
      } catch (const std::exception& e) {
        error = e.what();
      }
      const int64_t t_end = now_ns();
      const double ms = (t_end - t0) / 1e6;
      if (timing_ && t_launched && t_synced)
        phases_.push_back({(t_drafted ? t_drafted - t0 : 0) / 1e6, (t_launched - (t_drafted ? t_drafted : t0)) / 1e6,
                           (t_synced - t_launched) / 1e6, (t_end - t_synced) / 1e6});
      if (trace_ != nullptr && error.empty()) trace_step(t0, t_end, decoding);
      std::lock_guard<std::mutex> lk(mu_);
      stepping_ = false;
      steps_++;
      step_log_.emplace_back(ms, decoding);
      if (!error.empty()) {
        for (Row& r : rows_)
          if (r.active) finish(r, kError, "decode loop: " + error);
      } else {
        for (int64_t i = 0; i < rows_n_; ++i) {
          Row& r = rows_[i];
          if (!r.active) continue;
          // keep each draft that equals the choice before it, then the choice where they first differ
          int64_t n = 0;
          while (n < S - 1 && ids[i * S + n + 1] == chosen[i * S + n]) n++;
          // the statistics count the drafts issued (a padded position the verify agreed with is
          // a correct token the row keeps, not a draft accepted)
          const int64_t issued = S_ > 1 ? issued_[i] : 0;
          drafted_ += issued;
          accepted_ += std::min(n, issued);
          if (S_ > 1 && spec_events_.size() < kSpecEventsMax) {  // the record: n, then each draft's confidence (-2: not issued)
            spec_events_.push_back(std::min(n, issued));
            const bool conf = draft_block_ > 0 && draft_block_ < K;
            for (int64_t j = 0; j < S_ - 1; ++j)
              spec_events_.push_back(j >= issued ? -2 : conf ? drafted[i * K + draft_block_ + j] : -1);
          }
          bool done = false;
          for (int64_t j = 1; j <= n && !done; ++j) done = emit(r, ids[i * S + j]);
          if (!done) done = emit(r, chosen[i * S + n]);
          if (done) continue;
          r.position += n + 1;
          r.next = chosen[i * S + n];
          if (r.position + S_ > r.max_position) finish(r, kLength, "");
        }
      }
      idle_.notify_all();
    }
  }

  Graphs verify_;
  std::map<int64_t, std::unique_ptr<Graphs>> verifies_;  // shorter verify steps by token rows (set_verify)
  std::unique_ptr<Graphs> draft_;
  // VXPU_LOOP_TIMING=1: each step's host phases in ms (the drafter launched, the verify launched, the GPUs
  // waited for, the acceptance), their medians printed when the loop stops
  const bool timing_ = std::getenv("VXPU_LOOP_TIMING") != nullptr;
  std::vector<std::array<double, 4>> phases_;
  void print_phases() {
    if (!timing_ || phases_.empty()) return;
    const char* names[4] = {"drafter launched", "verify launched", "GPUs waited for", "acceptance"};
    fprintf(stderr, "[vxpu] decode loop phases over %zu steps (median ms):", phases_.size());
    for (int k = 0; k < 4; ++k) {
      std::vector<double> v;
      for (const auto& p : phases_) v.push_back(p[k]);
      std::sort(v.begin(), v.end());
      fprintf(stderr, " %s %.3f;", names[k], v[v.size() / 2]);
    }
    fprintf(stderr, "\n");
  }
  int64_t compose_ = 0;  // the graph that writes the drafts into the verify's ids on the GPU (0: via the host)
  int64_t policy_max_draft_ = -1;  // set_policy: drafts issued a step at most (-1: all)
  int64_t policy_min_score_ = 0;  // set_policy: stop issuing at the first draft scored under this (0: none)
  std::vector<int64_t> issued_;  // per row, the drafts issued this step
  void* drafted_event_ = nullptr;
  int64_t rows_n_ = 0, S_ = 1;
  at::Tensor ids_host_, pos_host_, chosen_host_, draft_out_host_;
  at::Tensor temperature_, top_k_, top_p_, temperature_host_, top_k_host_, top_p_host_;
  std::vector<Row> rows_;
  std::map<int64_t, Event> pending_;
  std::vector<std::pair<double, int64_t>> step_log_;
  std::FILE* trace_ = nullptr;
  int64_t tr_launch0_ = 0, tr_launch1_ = 0, tr_draft0_ = 0, tr_draft1_ = 0, tr_draft_synced_ = 0;
  std::vector<int64_t> tr_synced_;

  // One JSON line per step: milliseconds from the step's start to each phase's end, the GPUs'
  // stream drains in launch order, and each host node's start and end (StepHost::times: the
  // host thread's clock, the same steady clock).
  void trace_step(int64_t t0, int64_t t_end, int64_t decoding) {
    std::string line = "{\"step\":" + std::to_string(steps_.load()) + ",\"decoding\":" + std::to_string(decoding) +
                       ",\"t0_ns\":" + std::to_string(t0) +  // absolute, so batches' steps can be laid side by side
                       ",\"ms\":" + std::to_string((t_end - t0) / 1e6) +
                       ",\"launch0_ms\":" + std::to_string((tr_launch0_ - t0) / 1e6) +
                       ",\"launched_ms\":" + std::to_string((tr_launch1_ - t0) / 1e6) + ",\"synced_ms\":[";
    for (size_t g = 0; g < tr_synced_.size(); ++g)
      line += (g ? "," : "") + std::to_string((tr_synced_[g] - t0) / 1e6);
    line += "]";
    if (draft_) {
      line += ",\"draft_launch0_ms\":" + std::to_string((tr_draft0_ - t0) / 1e6) +
              ",\"draft_launched_ms\":" + std::to_string((tr_draft1_ - t0) / 1e6) +
              ",\"draft_checked_ms\":" + std::to_string((tr_draft_synced_ - t0) / 1e6) + ",\"draft_host\":[";
      if (draft_->host != nullptr) {
        at::Tensor dtimes = draft_->host->times();
        const int64_t* dt = dtimes.data_ptr<int64_t>();
        for (int64_t i = 0; i < dtimes.size(0); ++i) {
          if (dt[5 * i] == 0) continue;
          line += (i ? "," : "") + std::string("{\"start_ms\":") + std::to_string((dt[5 * i] - t0) / 1e6) +
                  ",\"end_ms\":" + std::to_string((dt[5 * i + 1] - t0) / 1e6) + "}";
        }
      }
      line += "]";
    }
    line += ",\"host\":[";
    if (verify_.host != nullptr) {
      at::Tensor times = verify_.host->times();
      const int64_t* t = times.data_ptr<int64_t>();
      for (int64_t i = 0; i < times.size(0); ++i) {
        if (t[5 * i] == 0) continue;
        line += (i ? "," : "") + std::string("{\"start_ms\":") + std::to_string((t[5 * i] - t0) / 1e6) +
                ",\"end_ms\":" + std::to_string((t[5 * i + 1] - t0) / 1e6) + "}";
      }
    }
    line += "]}\n";
    std::fputs(line.c_str(), trace_);
    std::fflush(trace_);
  }
  std::atomic<int64_t> steps_{0};
  int64_t drafted_ = 0, accepted_ = 0;
  int64_t draft_block_ = 0;  // the drafter's tokens' width when its output also carries confidences
  static constexpr size_t kSpecEventsMax = 1 << 24;  // records kept between speculation_events() calls
  std::vector<int64_t> spec_events_;
  std::mutex mu_;
  std::condition_variable wake_, idle_, events_cv_;
  bool paused_ = false, stepping_ = false, stop_ = false;
  std::thread thread_;
};

void register_node_server(pybind11::module& m) {
  pybind11::class_<NodeClient, std::shared_ptr<NodeClient>>(m, "NodeClient")
      .def(pybind11::init<const std::string&, int64_t>())
      .def("call", &NodeClient::call, pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("server_times", &NodeClient::server_times);
  pybind11::class_<NodeServer>(m, "NodeServer", pybind11::dynamic_attr())
      .def(pybind11::init<int64_t, const std::vector<std::string>&, const std::vector<std::vector<at::Tensor>>&>(),
           pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("port", &NodeServer::port)
      .def("messages", &NodeServer::messages)
      .def("stop", &NodeServer::stop, pybind11::call_guard<pybind11::gil_scoped_release>());
  pybind11::class_<StepHost>(m, "StepHost")
      .def(pybind11::init<at::Tensor>())
      .def("add_local", &StepHost::add_local, pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("add_remote", &StepHost::add_remote, pybind11::arg("inputs"), pybind11::arg("outbox"), pybind11::arg("client"),
           pybind11::arg("node"), pybind11::arg("prefix") = std::vector<at::Tensor>{})
      .def("start", &StepHost::start)
      .def("go", &StepHost::go)
      .def("take_error", &StepHost::take_error)
      .def("times", &StepHost::times)
      .def("running", &StepHost::running)
      .def("stop", &StepHost::stop, pybind11::call_guard<pybind11::gil_scoped_release>());
  pybind11::class_<DecodeLoop>(m, "DecodeLoop")
      .def(pybind11::init<at::Tensor, at::Tensor, at::Tensor, const std::vector<int64_t>&, const std::vector<int64_t>&,
                          const std::vector<int64_t>&, pybind11::object, at::Tensor>())
      .def(pybind11::init<at::Tensor, at::Tensor, at::Tensor, const std::vector<int64_t>&, const std::vector<int64_t>&,
                          const std::vector<int64_t>&, pybind11::object, at::Tensor, int64_t>())  // + out_graph
      .def("set_draft", &DecodeLoop::set_draft, pybind11::arg("input_ids"), pybind11::arg("positions"),
           pybind11::arg("out"), pybind11::arg("execs"), pybind11::arg("streams"), pybind11::arg("devices"),
           pybind11::arg("host"), pybind11::arg("flags"), pybind11::arg("compose") = 0, pybind11::arg("block") = 0)
      .def("set_policy", &DecodeLoop::set_policy, pybind11::arg("max_draft"), pybind11::arg("min_score"))
      .def("set_verify", &DecodeLoop::set_verify, pybind11::arg("rows"), pybind11::arg("input_ids"),
           pybind11::arg("positions"), pybind11::arg("chosen"), pybind11::arg("execs"), pybind11::arg("streams"),
           pybind11::arg("devices"), pybind11::arg("host"), pybind11::arg("flags"), pybind11::arg("out_graph") = 0)
      .def("set_sampling", &DecodeLoop::set_sampling)
      .def("start", &DecodeLoop::start)
      .def("pause", &DecodeLoop::pause, pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("resume", &DecodeLoop::resume)
      .def("admit", &DecodeLoop::admit)
      .def("cancel", &DecodeLoop::cancel)
      .def("free_rows", &DecodeLoop::free_rows)
      .def("events", &DecodeLoop::events, pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("steps", &DecodeLoop::steps)
      .def("take_steps", &DecodeLoop::take_steps)
      .def("drafts", &DecodeLoop::drafts)
      .def("speculation_events", &DecodeLoop::speculation_events)
      .def("stop", &DecodeLoop::stop, pybind11::call_guard<pybind11::gil_scoped_release>());
  pybind11::class_<RangeServer>(m, "RangeServer")
      .def(pybind11::init<int64_t, pybind11::function>())
      .def("port", &RangeServer::port)
      .def("messages", &RangeServer::messages)
      .def("native_steps", &RangeServer::native_steps)
      .def("set_decode", &RangeServer::set_decode)
      .def("stop", &RangeServer::stop, pybind11::call_guard<pybind11::gil_scoped_release>());
}
