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

// Fused kernels over packed fp4 (e2m1) expert banks with e8m0 block
// scales, for the CPU.
//
// The model maker's module dequantizes an expert to bf16 (exact: fp4
// values times a power-of-two scale fit bf16) and multiplies; PyTorch's
// plain-aten path does that by table lookups with int64 indices and
// materializes the bf16 copy, which costs more than the multiply. Here
// the nibbles are decoded in registers. Numerically x * (v * s) equals
// (x * v) * s exactly (s is a power of two), so the only difference
// from the reference is fp32 accumulation order, as with any kernel.
//
// Layouts (the fork's): a bank w is [E, N, K/2] uint8, element 2j in the
// low nibble of byte j and 2j+1 in the high nibble; scales are
// [E, N, K/32] e8m0 (one byte per row per 32-block along K); x is bf16.
//
// Entry points, mirroring the fork's two routings:
//   fp4_gather_mm(x [M, K], w, scale, sel [M]) -> bf16 [M, N]
//       row i uses expert sel[i]: the decode routing, a token's top-k
//       experts, each a matrix-vector product read once from the packed
//       bank and never materialized (memory-bound by design).
//   fp4_grouped_mm(x [M, K], w, scale, offs [E]) -> bf16 [M, N]
//       rows offs[e-1]:offs[e] use expert e: the prefill routing. Each
//       expert is decoded once to bf16 by the vectorized dequant and the
//       FLOPs go to torch's bf16 GEMM, which is the right tool for them.
//   fp4_dequant(w, scale, e) -> bf16 [N, K]      one expert, for tests.
//
// SIMD: AVX-512 (x86, runtime-dispatched) and NEON (arm64); a scalar
// path everywhere else.

#include <torch/extension.h>
#include <c10/util/Float8_e4m3fn.h>
#include <ATen/Parallel.h>

#include <cmath>
#include <cstdint>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <mutex>
#include <unordered_map>
#include <string>

#if defined(VXPU_HAVE_IO_URING)
#include <liburing.h>
#endif

#if defined(__aarch64__)
#include <arm_neon.h>
#endif
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include <algorithm>
#include <functional>
#include <thread>
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

// parallel_range(n, grain, fn): fn(begin, end) over [0, n) in chunks of
// at least `grain`, on every core. OpenMP when the extension was built
// with it (Linux, gcc); std::thread otherwise (Apple clang ships no
// OpenMP runtime). at::parallel_for would be the natural choice, but it
// runs serially in an extension compiled without OpenMP.
int64_t kernel_threads() {
  // VXPU_KERNEL_THREADS: the kernels are memory-bound and want every
  // core; the graph's thousands of small ops do better on fewer (torch's
  // own count), so the two can differ.
  static const int64_t n = [] {
    if (const char* v = std::getenv("VXPU_KERNEL_THREADS")) {
      const long k = std::atol(v);
      if (k > 0) return int64_t(k);
    }
    return int64_t(0);
  }();
  return n > 0 ? n : std::max<int64_t>(1, at::get_num_threads());
}

void parallel_range(int64_t n, int64_t grain, const std::function<void(int64_t, int64_t)>& fn) {
  if (n <= 0) return;
  int64_t threads = kernel_threads();
  // 16 chunks per thread, taken dynamically: a fused expert phase is a
  // few hundred microseconds over ~800 work items, and with 4 chunks per
  // thread the last chunk (tens of microseconds) left the other threads
  // waiting at the phase's end; finer chunks cut the decode step's expert
  // time by about 4% on a 48-core node
  int64_t chunks = std::min(n / std::max<int64_t>(grain, 1), threads * 16);
  if (chunks <= 1) { fn(0, n); return; }
  const int64_t step = (n + chunks - 1) / chunks;
#if defined(_OPENMP)
  #pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
  for (int64_t c = 0; c < chunks; ++c) {
    const int64_t b = c * step, e = std::min(n, b + step);
    if (b < e) fn(b, e);
  }
#else
  std::vector<std::thread> pool;
  std::atomic<int64_t> next{0};
  for (int64_t t = 0; t < std::min(threads, chunks); ++t) {
    pool.emplace_back([&]() {
      for (int64_t c = next.fetch_add(1); c < chunks; c = next.fetch_add(1)) {
        const int64_t b = c * step, e = std::min(n, b + step);
        if (b < e) fn(b, e);
      }
    });
  }
  for (auto& th : pool) th.join();
#endif
}

const float kFp4[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                        -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
// the same values doubled, as small integers (for integer lookups)
const int8_t kFp4x2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

// 2^(e-127) by building the float's exponent field; e = 0 (2^-127) is
// below fp32's normal range and goes through ldexp.
inline float e8m0(uint8_t e) {
  if (e == 0) return std::ldexp(1.0f, -127);
  if (e == 0xFF) return std::numeric_limits<float>::quiet_NaN();  // the format's NaN, as torch decodes it
  const uint32_t bits = uint32_t(e) << 23;
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

// fp32 -> bf16, round to nearest even (what at::BFloat16 does)
inline uint16_t bf16_bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return uint16_t(u >> 16);
}

// ---------------------------------------------------------------- scalar

inline void decode_row_scalar(const uint8_t* w, const uint8_t* s, int64_t K, uint16_t* out) {
  for (int64_t b = 0; b < K / 32; ++b) {
    const float sc = e8m0(s[b]);
    const uint8_t* wb = w + b * 16;
    uint16_t* ob = out + b * 32;
    for (int j = 0; j < 16; ++j) {
      ob[2 * j] = bf16_bits(kFp4[wb[j] & 15] * sc);
      ob[2 * j + 1] = bf16_bits(kFp4[wb[j] >> 4] * sc);
    }
  }
}

inline float dot_packed_scalar(const float* x, const uint8_t* w, const uint8_t* s, int64_t K) {
  float acc = 0.f;
  for (int64_t b = 0; b < K / 32; ++b) {
    const uint8_t* wb = w + b * 16;
    const float* xb = x + b * 32;
    float part = 0.f;
    for (int j = 0; j < 16; ++j) {
      part += xb[2 * j] * kFp4[wb[j] & 15] + xb[2 * j + 1] * kFp4[wb[j] >> 4];
    }
    acc += part * e8m0(s[b]);
  }
  return acc;
}

// ---------------------------------------------------------------- AVX-512

#if defined(__x86_64__) || defined(_M_X64)
bool have_avx512() {
  static const bool ok = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw");
  return ok;
}

__attribute__((target("avx512f,avx512bw,avx512vl")))
inline __m256i to_bf16_avx512(__m512 f) {
  __m512i u = _mm512_castps_si512(f);
  __m512i lsb = _mm512_and_si512(_mm512_srli_epi32(u, 16), _mm512_set1_epi32(1));
  u = _mm512_add_epi32(u, _mm512_add_epi32(_mm512_set1_epi32(0x7FFF), lsb));
  return _mm512_cvtepi32_epi16(_mm512_srli_epi32(u, 16));
}

// x de-interleaved into even/odd elements: dot with a packed row,
// scale applied per 32-block, one horizontal reduction per row.
__attribute__((target("avx512f,avx512bw,avx512vl")))
inline float dot_packed_avx512(const float* xe, const float* xo, const uint8_t* w, const uint8_t* s, int64_t K) {
  const __m512 table = _mm512_loadu_ps(kFp4);
  const __m512i lo_mask = _mm512_set1_epi32(0x0F);
  __m512 acc = _mm512_setzero_ps();
  for (int64_t b = 0; b < K / 32; ++b) {
    __m512i bytes = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + b * 16)));
    __m512 vlo = _mm512_permutexvar_ps(_mm512_and_si512(bytes, lo_mask), table);
    __m512 vhi = _mm512_permutexvar_ps(_mm512_srli_epi32(bytes, 4), table);
    __m512 part = _mm512_mul_ps(vlo, _mm512_loadu_ps(xe + b * 16));
    part = _mm512_fmadd_ps(vhi, _mm512_loadu_ps(xo + b * 16), part);
    acc = _mm512_fmadd_ps(part, _mm512_set1_ps(e8m0(s[b])), acc);
  }
  return _mm512_reduce_add_ps(acc);
}

__attribute__((target("avx512f,avx512bw,avx512vl")))
inline void decode_row_avx512(const uint8_t* w, const uint8_t* s, int64_t K, uint16_t* out) {
  const __m512 table = _mm512_loadu_ps(kFp4);
  const __m512i lo_mask = _mm512_set1_epi32(0x0F);
  // interleave lo/hi lanes back into element order: a = lo0 hi0 .. lo7 hi7, b = lo8 hi8 .. lo15 hi15
  const __m512i ia = _mm512_set_epi32(23, 7, 22, 6, 21, 5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0);
  const __m512i ib = _mm512_set_epi32(31, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26, 10, 25, 9, 24, 8);
  for (int64_t b = 0; b < K / 32; ++b) {
    __m512i bytes = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + b * 16)));
    __m512 vlo = _mm512_permutexvar_ps(_mm512_and_si512(bytes, lo_mask), table);
    __m512 vhi = _mm512_permutexvar_ps(_mm512_srli_epi32(bytes, 4), table);
    __m512 sc = _mm512_set1_ps(e8m0(s[b]));
    __m512 a = _mm512_mul_ps(_mm512_permutex2var_ps(vlo, ia, vhi), sc);
    __m512 c = _mm512_mul_ps(_mm512_permutex2var_ps(vlo, ib, vhi), sc);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + b * 32), to_bf16_avx512(a));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + b * 32 + 16), to_bf16_avx512(c));
  }
}
#else
bool have_avx512() { return false; }
#endif

// ---------------------------------------------------------------- NEON

#if defined(__aarch64__)
inline uint16x8_t to_bf16_neon(float32x4_t f0, float32x4_t f1) {
  uint32x4_t u0 = vreinterpretq_u32_f32(f0), u1 = vreinterpretq_u32_f32(f1);
  uint32x4_t one = vdupq_n_u32(1), bias = vdupq_n_u32(0x7FFF);
  u0 = vaddq_u32(u0, vaddq_u32(bias, vandq_u32(vshrq_n_u32(u0, 16), one)));
  u1 = vaddq_u32(u1, vaddq_u32(bias, vandq_u32(vshrq_n_u32(u1, 16), one)));
  return vcombine_u16(vshrn_n_u32(u0, 16), vshrn_n_u32(u1, 16));
}

// 16 int8 (values doubled) -> four float32x4 (undoubled by the caller's scale)
inline void widen_f32(int8x16_t v, float32x4_t out[4]) {
  int16x8_t l = vmovl_s8(vget_low_s8(v)), h = vmovl_s8(vget_high_s8(v));
  out[0] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(l)));
  out[1] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(l)));
  out[2] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(h)));
  out[3] = vcvtq_f32_s32(vmovl_s16(vget_high_s16(h)));
}

inline float dot_packed_neon(const float* xe, const float* xo, const uint8_t* w, const uint8_t* s, int64_t K) {
  const int8x16_t table = vld1q_s8(kFp4x2);
  const uint8x16_t lo_mask = vdupq_n_u8(0x0F);
  float32x4_t acc0 = vdupq_n_f32(0.f), acc1 = vdupq_n_f32(0.f), acc2 = vdupq_n_f32(0.f), acc3 = vdupq_n_f32(0.f);
  for (int64_t b = 0; b < K / 32; ++b) {
    uint8x16_t bytes = vld1q_u8(w + b * 16);
    int8x16_t lo = vqtbl1q_s8(table, vandq_u8(bytes, lo_mask));
    int8x16_t hi = vqtbl1q_s8(table, vshrq_n_u8(bytes, 4));
    float32x4_t vl[4], vh[4];
    widen_f32(lo, vl);
    widen_f32(hi, vh);
    const float* xeb = xe + b * 16;
    const float* xob = xo + b * 16;
    float32x4_t p0 = vmulq_f32(vl[0], vld1q_f32(xeb));
    float32x4_t p1 = vmulq_f32(vl[1], vld1q_f32(xeb + 4));
    float32x4_t p2 = vmulq_f32(vl[2], vld1q_f32(xeb + 8));
    float32x4_t p3 = vmulq_f32(vl[3], vld1q_f32(xeb + 12));
    p0 = vfmaq_f32(p0, vh[0], vld1q_f32(xob));
    p1 = vfmaq_f32(p1, vh[1], vld1q_f32(xob + 4));
    p2 = vfmaq_f32(p2, vh[2], vld1q_f32(xob + 8));
    p3 = vfmaq_f32(p3, vh[3], vld1q_f32(xob + 12));
    const float32x4_t sc = vdupq_n_f32(0.5f * e8m0(s[b]));  // undo the doubling
    acc0 = vfmaq_f32(acc0, p0, sc);
    acc1 = vfmaq_f32(acc1, p1, sc);
    acc2 = vfmaq_f32(acc2, p2, sc);
    acc3 = vfmaq_f32(acc3, p3, sc);
  }
  return vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
}

inline void decode_row_neon(const uint8_t* w, const uint8_t* s, int64_t K, uint16_t* out) {
  const int8x16_t table = vld1q_s8(kFp4x2);
  const uint8x16_t lo_mask = vdupq_n_u8(0x0F);
  for (int64_t b = 0; b < K / 32; ++b) {
    uint8x16_t bytes = vld1q_u8(w + b * 16);
    int8x16_t lo = vqtbl1q_s8(table, vandq_u8(bytes, lo_mask));
    int8x16_t hi = vqtbl1q_s8(table, vshrq_n_u8(bytes, 4));
    int8x16_t a = vzip1q_s8(lo, hi), c = vzip2q_s8(lo, hi);  // element order
    float32x4_t fa[4], fc[4];
    widen_f32(a, fa);
    widen_f32(c, fc);
    const float32x4_t sc = vdupq_n_f32(0.5f * e8m0(s[b]));
    uint16_t* ob = out + b * 32;
    vst1q_u16(ob, to_bf16_neon(vmulq_f32(fa[0], sc), vmulq_f32(fa[1], sc)));
    vst1q_u16(ob + 8, to_bf16_neon(vmulq_f32(fa[2], sc), vmulq_f32(fa[3], sc)));
    vst1q_u16(ob + 16, to_bf16_neon(vmulq_f32(fc[0], sc), vmulq_f32(fc[1], sc)));
    vst1q_u16(ob + 24, to_bf16_neon(vmulq_f32(fc[2], sc), vmulq_f32(fc[3], sc)));
  }
}
#endif

// ---------------------------------------------------------------- dispatch

inline bool simd_dot() {
#if defined(__aarch64__)
  return true;
#else
  return have_avx512();
#endif
}

inline float dot_packed(const float* x, const float* xe, const float* xo, const uint8_t* w, const uint8_t* s, int64_t K) {
#if defined(__aarch64__)
  return dot_packed_neon(xe, xo, w, s, K);
#elif defined(__x86_64__) || defined(_M_X64)
  if (have_avx512()) return dot_packed_avx512(xe, xo, w, s, K);
  return dot_packed_scalar(x, w, s, K);
#else
  return dot_packed_scalar(x, w, s, K);
#endif
}

inline void decode_row(const uint8_t* w, const uint8_t* s, int64_t K, uint16_t* out) {
#if defined(__aarch64__)
  decode_row_neon(w, s, K, out);
#elif defined(__x86_64__) || defined(_M_X64)
  if (have_avx512()) decode_row_avx512(w, s, K, out); else decode_row_scalar(w, s, K, out);
#else
  decode_row_scalar(w, s, K, out);
#endif
}

void check_bank(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& scale) {
  TORCH_CHECK(x.dtype() == torch::kBFloat16 && x.dim() == 2 && x.is_contiguous(), "x must be contiguous bf16 [M, K]");
  TORCH_CHECK(w.dtype() == torch::kUInt8 && w.dim() == 3 && w.is_contiguous(), "w must be contiguous uint8 [E, N, K/2]");
  TORCH_CHECK(scale.dtype() == torch::kFloat8_e8m0fnu && scale.dim() == 3 && scale.is_contiguous(),
              "scale must be contiguous e8m0 [E, N, K/32]");
  TORCH_CHECK(x.size(1) == 2 * w.size(2), "x and w disagree on K");
  TORCH_CHECK(x.size(1) % 32 == 0 && scale.size(2) * 32 == x.size(1), "K must be a multiple of 32 with one scale per block");
  TORCH_CHECK(scale.size(0) == w.size(0) && scale.size(1) == w.size(1), "scale and w disagree on E or N");
  TORCH_CHECK(x.device().is_cpu() && w.device().is_cpu() && scale.device().is_cpu(), "CPU tensors only");
}

// x as fp32, and de-interleaved even/odd copies for the SIMD packed dot.
struct XRows {
  std::vector<float> full, even, odd;
  int64_t M, K;
  explicit XRows(const torch::Tensor& x) : M(x.size(0)), K(x.size(1)) {
    full.resize(M * K);
    const at::BFloat16* src = x.data_ptr<at::BFloat16>();
    for (int64_t i = 0; i < M * K; ++i) full[i] = static_cast<float>(src[i]);
    if (simd_dot()) {
      even.resize(M * K / 2);
      odd.resize(M * K / 2);
      for (int64_t i = 0; i < M; ++i)
        for (int64_t k = 0; k < K / 2; ++k) {
          even[i * (K / 2) + k] = full[i * K + 2 * k];
          odd[i * (K / 2) + k] = full[i * K + 2 * k + 1];
        }
    }
  }
};

}  // namespace

torch::Tensor fp4_gather_mm(torch::Tensor x, torch::Tensor w, torch::Tensor scale, torch::Tensor sel, int64_t intermediate_size) {
  const bool per_expert = (w.dim() == 1);
  int64_t E = 0, N = 0;
  if (per_expert) {
    TORCH_CHECK(x.dtype() == torch::kBFloat16 && x.dim() == 2 && x.is_contiguous(), "x must be contiguous bf16 [M, K]");
    TORCH_CHECK(w.dtype() == torch::kInt64 && w.is_contiguous(), "w must be contiguous int64");
    TORCH_CHECK(scale.dtype() == torch::kInt64 && scale.is_contiguous() && scale.size(0) == w.size(0), "scale mismatch");
    TORCH_CHECK(intermediate_size > 0, "intermediate_size must be specified for per-expert pointers");
    E = w.size(0);
    N = intermediate_size;
  } else {
    check_bank(x, w, scale);
    E = w.size(0);
    N = w.size(1);
  }
  TORCH_CHECK(sel.dtype() == torch::kInt64 && sel.dim() == 1 && sel.size(0) == x.size(0), "sel must be int64 [M]");
  const int64_t M = x.size(0), K = x.size(1);
  auto out = torch::empty({M, N}, x.options());
  XRows xr(x);
  auto selc = sel.contiguous();
  const int64_t* selp = selc.data_ptr<int64_t>();
  for (int64_t i = 0; i < M; ++i) TORCH_CHECK(selp[i] >= 0 && selp[i] < E, "expert index out of range");

  const int64_t* wptrs = per_expert ? w.data_ptr<int64_t>() : nullptr;
  const int64_t* sptrs = per_expert ? scale.data_ptr<int64_t>() : nullptr;
  const uint8_t* wp = per_expert ? nullptr : w.data_ptr<uint8_t>();
  const uint8_t* sp = per_expert ? nullptr : reinterpret_cast<const uint8_t*>(scale.data_ptr());

  at::BFloat16* op = out.data_ptr<at::BFloat16>();
  const int64_t nblock = 16;  // output rows per task
  const int64_t nb = (N + nblock - 1) / nblock;
  parallel_range(M * nb, 1, [&](int64_t t0, int64_t t1) {
    for (int64_t t = t0; t < t1; ++t) {
      const int64_t i = t / nb, n0 = (t % nb) * nblock, e = selp[i];
      const uint8_t* we = per_expert ? reinterpret_cast<const uint8_t*>(wptrs[e]) : (wp + e * N * (K / 2));
      const uint8_t* se = per_expert ? reinterpret_cast<const uint8_t*>(sptrs[e]) : (sp + e * N * (K / 32));
      const float* xf = xr.full.data() + i * K;
      const float* xe = xr.even.empty() ? nullptr : xr.even.data() + i * (K / 2);
      const float* xo = xr.odd.empty() ? nullptr : xr.odd.data() + i * (K / 2);
      for (int64_t n = n0; n < std::min(N, n0 + nblock); ++n) {
        op[i * N + n] = static_cast<at::BFloat16>(dot_packed(xf, xe, xo, we + n * (K / 2), se + n * (K / 32), K));
      }
    }
  });
  return out;
}

// Decode expert e into `out` ([N, K] bf16, contiguous).
void fp4_dequant_into(const torch::Tensor& w, const torch::Tensor& scale, int64_t e, torch::Tensor& out, int64_t intermediate_size = 0) {
  const bool per_expert = (w.dim() == 1);
  const int64_t N = per_expert ? intermediate_size : w.size(1);
  const int64_t K = out.size(1);
  const uint8_t* wp = per_expert ? reinterpret_cast<const uint8_t*>(w.data_ptr<int64_t>()[e])
                                 : (w.data_ptr<uint8_t>() + e * N * (K / 2));
  const uint8_t* sp = per_expert ? reinterpret_cast<const uint8_t*>(scale.data_ptr<int64_t>()[e])
                                 : (reinterpret_cast<const uint8_t*>(scale.data_ptr()) + e * N * (K / 32));
  uint16_t* op = reinterpret_cast<uint16_t*>(out.data_ptr<at::BFloat16>());
  parallel_range(N, 8, [&](int64_t n0, int64_t n1) {
    for (int64_t n = n0; n < n1; ++n) decode_row(wp + n * (K / 2), sp + n * (K / 32), K, op + n * K);
  });
}

torch::Tensor fp4_dequant(torch::Tensor w, torch::Tensor scale, int64_t e, int64_t intermediate_size, int64_t dim) {
  const bool per_expert = (w.dim() == 1);
  int64_t N = 0, K = 0;
  if (per_expert) {
    TORCH_CHECK(w.dtype() == torch::kInt64 && w.is_contiguous(), "w must be contiguous int64 pointers");
    TORCH_CHECK(scale.dtype() == torch::kInt64 && scale.is_contiguous() && scale.size(0) == w.size(0), "scale mismatch");
    TORCH_CHECK(intermediate_size > 0 && dim > 0, "intermediate_size and dim must be specified for per-expert pointers");
    N = intermediate_size;
    K = dim;
  } else {
    TORCH_CHECK(w.dtype() == torch::kUInt8 && w.dim() == 3 && w.is_contiguous(), "w must be contiguous uint8 [E, N, K/2]");
    TORCH_CHECK(scale.dtype() == torch::kFloat8_e8m0fnu && scale.dim() == 3 && scale.is_contiguous(),
                "scale must be contiguous e8m0 [E, N, K/32]");
    TORCH_CHECK(scale.size(0) == w.size(0) && scale.size(1) == w.size(1) && scale.size(2) * 16 == w.size(2),
                "scale and w disagree on E, N or K");
    TORCH_CHECK(w.device().is_cpu() && scale.device().is_cpu(), "CPU tensors only");
    N = w.size(1);
    K = w.size(2) * 2;
  }
  TORCH_CHECK(e >= 0 && e < w.size(0), "expert index out of range");
  auto out = torch::empty({N, K}, w.options().dtype(torch::kBFloat16));
  fp4_dequant_into(w, scale, e, out, N);
  return out;
}

torch::Tensor fp4_grouped_mm(torch::Tensor x, torch::Tensor w, torch::Tensor scale, torch::Tensor offs, int64_t intermediate_size) {
  const bool per_expert = (w.dim() == 1);
  int64_t E = 0, N = 0;
  if (per_expert) {
    TORCH_CHECK(x.dtype() == torch::kBFloat16 && x.dim() == 2 && x.is_contiguous(), "x must be contiguous bf16 [M, K]");
    TORCH_CHECK(w.dtype() == torch::kInt64 && w.is_contiguous(), "w must be contiguous int64");
    TORCH_CHECK(scale.dtype() == torch::kInt64 && scale.is_contiguous() && scale.size(0) == w.size(0), "scale mismatch");
    TORCH_CHECK(intermediate_size > 0, "intermediate_size must be specified for per-expert pointers");
    E = w.size(0);
    N = intermediate_size;
  } else {
    check_bank(x, w, scale);
    E = w.size(0);
    N = w.size(1);
  }
  TORCH_CHECK(offs.dtype() == torch::kInt32 && offs.dim() == 1 && offs.size(0) == E, "offs must be int32 [E]");
  const int64_t M = x.size(0);
  auto out = torch::empty({M, N}, x.options());
  auto offc = offs.contiguous();
  const int32_t* offp = offc.data_ptr<int32_t>();
  TORCH_CHECK(E == 0 || offp[E - 1] == M, "offs must end at M");
  for (int64_t e = 0; e < E; ++e)
    TORCH_CHECK(offp[e] >= (e ? offp[e - 1] : 0), "offs must be non-decreasing and start at or above 0");
  // one decode buffer, reused for every expert (a fresh allocation per
  // expert would pay its page faults every time)
  auto we = torch::empty({N, x.size(1)}, x.options());
  int64_t r0 = 0;
  for (int64_t e = 0; e < E; ++e) {
    const int64_t r1 = offp[e];
    if (r1 > r0) {
      // decoded once per expert; the GEMM over its tokens is torch's bf16 mm
      fp4_dequant_into(w, scale, e, we, N);
      auto rows = out.slice(0, r0, r1);
      at::mm_out(rows, x.slice(0, r0, r1), we.t());
    }
    r0 = r1;
  }
  return out;
}

// ---------------------------------------------------------------- fp8 dense

// e4m3fn byte -> fp32 (no infinities; 0x7F/0xFF are NaN, decoded as NaN as torch does)
inline float e4m3(uint8_t b) {
  if ((b & 0x7F) == 0x7F) return std::numeric_limits<float>::quiet_NaN();
  const uint32_t sign = uint32_t(b & 0x80) << 24;
  const uint32_t exp = (b >> 3) & 0xF, man = b & 7;
  uint32_t bits;
  if (exp == 0) {  // subnormal: man * 2^-9
    float f = float(man) * (1.0f / 512.0f);
    std::memcpy(&bits, &f, 4);
  } else {
    bits = ((exp + 120u) << 23) | (man << 20);
  }
  bits |= sign;
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

#if defined(__x86_64__) || defined(_M_X64)
__attribute__((target("avx512f,avx512bw,avx512vl")))
inline __m512 e4m3_avx512(__m128i bytes16) {
  __m512i b = _mm512_cvtepu8_epi32(bytes16);
  __m512i sign = _mm512_slli_epi32(_mm512_and_si512(b, _mm512_set1_epi32(0x80)), 24);
  __m512i exp = _mm512_and_si512(_mm512_srli_epi32(b, 3), _mm512_set1_epi32(0xF));
  __m512i man = _mm512_and_si512(b, _mm512_set1_epi32(7));
  __m512i normal = _mm512_or_si512(_mm512_slli_epi32(_mm512_add_epi32(exp, _mm512_set1_epi32(120)), 23), _mm512_slli_epi32(man, 20));
  __m512 sub = _mm512_mul_ps(_mm512_cvtepi32_ps(man), _mm512_set1_ps(1.0f / 512.0f));
  __mmask16 is_sub = _mm512_cmpeq_epi32_mask(exp, _mm512_setzero_si512());
  __m512 val = _mm512_mask_blend_ps(is_sub, _mm512_castsi512_ps(normal), sub);
  val = _mm512_castsi512_ps(_mm512_or_si512(_mm512_castps_si512(val), sign));
  // exponent 15 with mantissa 7 is the format's NaN (0x7F / 0xFF)
  const __mmask16 is_nan = _mm512_cmpeq_epi32_mask(_mm512_and_si512(b, _mm512_set1_epi32(0x7F)), _mm512_set1_epi32(0x7F));
  return _mm512_mask_blend_ps(is_nan, val, _mm512_set1_ps(std::numeric_limits<float>::quiet_NaN()));
}

// x[K] fp32 dot one fp8 row with its K/32 block scales (row n's scales)
__attribute__((target("avx512f,avx512bw,avx512vl")))
inline float dot_fp8_avx512(const float* x, const uint8_t* w, const uint8_t* s, int64_t K) {
  __m512 acc = _mm512_setzero_ps();
  for (int64_t b = 0; b < K / 32; ++b) {
    const uint8_t* wb = w + b * 32;
    __m512 part = _mm512_mul_ps(e4m3_avx512(_mm_loadu_si128(reinterpret_cast<const __m128i*>(wb))), _mm512_loadu_ps(x + b * 32));
    part = _mm512_fmadd_ps(e4m3_avx512(_mm_loadu_si128(reinterpret_cast<const __m128i*>(wb + 16))), _mm512_loadu_ps(x + b * 32 + 16), part);
    acc = _mm512_fmadd_ps(part, _mm512_set1_ps(e8m0(s[b])), acc);
  }
  return _mm512_reduce_add_ps(acc);
}

__attribute__((target("avx512f,avx512bw,avx512vl")))
inline void decode_fp8_row_avx512(const uint8_t* w, const uint8_t* s, int64_t K, uint16_t* out) {
  for (int64_t b = 0; b < K / 32; ++b) {
    const __m512 sc = _mm512_set1_ps(e8m0(s[b]));
    const uint8_t* wb = w + b * 32;
    __m512 a = _mm512_mul_ps(e4m3_avx512(_mm_loadu_si128(reinterpret_cast<const __m128i*>(wb))), sc);
    __m512 c = _mm512_mul_ps(e4m3_avx512(_mm_loadu_si128(reinterpret_cast<const __m128i*>(wb + 16))), sc);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + b * 32), to_bf16_avx512(a));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + b * 32 + 16), to_bf16_avx512(c));
  }
}
#endif

inline float dot_fp8_scalar(const float* x, const uint8_t* w, const uint8_t* s, int64_t K) {
  float acc = 0.f;
  for (int64_t b = 0; b < K / 32; ++b) {
    float part = 0.f;
    for (int j = 0; j < 32; ++j) part += x[b * 32 + j] * e4m3(w[b * 32 + j]);
    acc += part * e8m0(s[b]);
  }
  return acc;
}

inline void decode_fp8_row_scalar(const uint8_t* w, const uint8_t* s, int64_t K, uint16_t* out) {
  for (int64_t b = 0; b < K / 32; ++b) {
    const float sc = e8m0(s[b]);
    for (int j = 0; j < 32; ++j) out[b * 32 + j] = bf16_bits(e4m3(w[b * 32 + j]) * sc);
  }
}

inline float dot_fp8(const float* x, const uint8_t* w, const uint8_t* s, int64_t K) {
#if defined(__x86_64__) || defined(_M_X64)
  if (have_avx512()) return dot_fp8_avx512(x, w, s, K);
#endif
  return dot_fp8_scalar(x, w, s, K);
}

inline void decode_fp8_row(const uint8_t* w, const uint8_t* s, int64_t K, uint16_t* out) {
#if defined(__x86_64__) || defined(_M_X64)
  if (have_avx512()) { decode_fp8_row_avx512(w, s, K, out); return; }
#endif
  decode_fp8_row_scalar(w, s, K, out);
}

// The fork's act_quant for one row: per 32-block, amax (floored at
// 1e-4), the smallest power of two >= amax/448 as the scale (a power
// of two maps to itself), x/scale clamped to +-448 and rounded to the
// e4m3 grid (nearest even, as a cast does), then times the scale again:
// the exact value the reference multiplies with.
inline float pow2_ceil(float v) {
  int e;
  const float m = std::frexp(v, &e);  // v = m * 2^e, m in [0.5, 1)
  return std::ldexp(1.0f, m == 0.5f ? e - 1 : e);
}

inline void act_quant_row(const at::BFloat16* x, int64_t K, float* out) {
  for (int64_t b = 0; b < K / 32; ++b) {
    float amax = 0.f;
    for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(static_cast<float>(x[b * 32 + j])));
    amax = std::max(amax, 1e-4f);
    const float s = pow2_ceil(amax / 448.0f);
    for (int j = 0; j < 32; ++j) {
      float q = static_cast<float>(x[b * 32 + j]) / s;
      q = std::min(448.0f, std::max(-448.0f, q));
      out[b * 32 + j] = static_cast<float>(c10::Float8_e4m3fn(q)) * s;
    }
  }
}

void check_fp8(const torch::Tensor& x, const torch::Tensor& w, const torch::Tensor& scale) {
  TORCH_CHECK(x.dtype() == torch::kBFloat16 && x.dim() == 2 && x.is_contiguous(), "x must be contiguous bf16 [M, K]");
  TORCH_CHECK(w.dtype() == torch::kFloat8_e4m3fn && w.dim() == 2 && w.is_contiguous(), "w must be contiguous fp8 [N, K]");
  TORCH_CHECK(scale.dtype() == torch::kFloat8_e8m0fnu && scale.dim() == 2 && scale.is_contiguous(), "scale must be contiguous e8m0 [N/32, K/32]");
  TORCH_CHECK(x.size(1) == w.size(1) && w.size(1) % 32 == 0 && w.size(0) % 32 == 0, "N and K must be multiples of 32");
  TORCH_CHECK(scale.size(0) * 32 == w.size(0) && scale.size(1) * 32 == w.size(1), "one scale per 32x32 block");
  TORCH_CHECK(x.device().is_cpu() && w.device().is_cpu() && scale.device().is_cpu(), "CPU tensors only");
}

// The dense fp8 linear: x [M, K] bf16 (already the fork's act_quant'd
// values, exact in bf16) times w [N, K] fp8 with [N/32, K/32] e8m0
// block scales -> bf16 [M, N]. One token: a fused GEMV per output row;
// more: each row decoded once to bf16 and torch's GEMM over the tokens.
torch::Tensor fp8_linear(torch::Tensor x, torch::Tensor w, torch::Tensor scale) {
  check_fp8(x, w, scale);
  const int64_t M = x.size(0), K = x.size(1), N = w.size(0), KB = K / 32;
  const uint8_t* wp = reinterpret_cast<const uint8_t*>(w.data_ptr());
  const uint8_t* sp = reinterpret_cast<const uint8_t*>(scale.data_ptr());
  const at::BFloat16* xs = x.data_ptr<at::BFloat16>();
  if (M == 1) {
    auto out = torch::empty({1, N}, x.options());
    std::vector<float> xf(K);
    act_quant_row(xs, K, xf.data());
    at::BFloat16* op = out.data_ptr<at::BFloat16>();
    parallel_range(N, 8, [&](int64_t n0, int64_t n1) {
      for (int64_t n = n0; n < n1; ++n) {
        op[n] = static_cast<at::BFloat16>(dot_fp8(xf.data(), wp + n * K, sp + (n / 32) * KB, K));
      }
    });
    return out;
  }
  // many tokens: quantized activations (exact in bf16) and the weight
  // decoded once, then torch's GEMM
  auto xq = torch::empty({M, K}, x.options());
  at::BFloat16* xqp = xq.data_ptr<at::BFloat16>();
  parallel_range(M, 1, [&](int64_t i0, int64_t i1) {
    std::vector<float> row(K);
    for (int64_t i = i0; i < i1; ++i) {
      act_quant_row(xs + i * K, K, row.data());
      for (int64_t k = 0; k < K; ++k) xqp[i * K + k] = static_cast<at::BFloat16>(row[k]);
    }
  });
  auto wd = torch::empty({N, K}, x.options());
  uint16_t* dp = reinterpret_cast<uint16_t*>(wd.data_ptr<at::BFloat16>());
  parallel_range(N, 8, [&](int64_t n0, int64_t n1) {
    for (int64_t n = n0; n < n1; ++n) decode_fp8_row(wp + n * K, sp + (n / 32) * KB, K, dp + n * K);
  });
  return at::mm(xq, wd.t());
}

// ---------------------------------------------------------------- fused expert step

// One decode layer's routed experts in one call: the fork's
// MoE._forward_batched with its _ffn between the projections, for n
// tokens each routed to topk experts. Rounding points are the
// reference's: gate and up rounded to bf16 (the bf16 matmul's output),
// SwiGLU with the clamps and the routing weight in fp32, the hidden
// vector rounded to bf16 then onto the fp8 grid per 32-block (the
// fork's act_quant; exact in bf16), the down projection rounded to bf16,
// and the topk rows summed in fp32 in k order (index_add's order).
//   x_in     [n, K] bf16, already input-quantized by the module
//   weights  [n, topk] fp32 routing weights; indices [n, topk] int64
//   w1, w3   [E, I, K/2] packed fp4, scales [E, I, K/32]; w2 [E, K, I/2], scale [E, K, I/32]
//            OR 1D int64 tensors of pointers to each expert's weights/scales
//   -> y [n, K] fp32
torch::Tensor fp4_moe_decode(torch::Tensor x_in, torch::Tensor weights, torch::Tensor indices,
                             torch::Tensor w1, torch::Tensor s1, torch::Tensor w3, torch::Tensor s3,
                             torch::Tensor w2, torch::Tensor s2, double swiglu_limit,
                             int64_t intermediate_size) {
  const bool per_expert = (w1.dim() == 1);
  const int64_t n = x_in.size(0), K = x_in.size(1);
  int64_t E = 0, I = 0;
  if (per_expert) {
    TORCH_CHECK(x_in.dtype() == torch::kBFloat16 && x_in.dim() == 2 && x_in.is_contiguous(), "x_in must be contiguous bf16 [n, K]");
    TORCH_CHECK(w1.dtype() == torch::kInt64 && w1.is_contiguous(), "w1 must be contiguous int64");
    TORCH_CHECK(s1.dtype() == torch::kInt64 && s1.is_contiguous() && s1.size(0) == w1.size(0), "s1 pointers mismatch");
    TORCH_CHECK(w3.dtype() == torch::kInt64 && w3.is_contiguous() && w3.size(0) == w1.size(0), "w3 pointers mismatch");
    TORCH_CHECK(s3.dtype() == torch::kInt64 && s3.is_contiguous() && s3.size(0) == w1.size(0), "s3 pointers mismatch");
    TORCH_CHECK(w2.dtype() == torch::kInt64 && w2.is_contiguous() && w2.size(0) == w1.size(0), "w2 pointers mismatch");
    TORCH_CHECK(s2.dtype() == torch::kInt64 && s2.is_contiguous() && s2.size(0) == w1.size(0), "s2 pointers mismatch");
    E = w1.size(0);
    I = intermediate_size;
    TORCH_CHECK(I > 0 && I % 32 == 0, "intermediate_size must be a positive multiple of 32");
  } else {
    // every bank checked in full: the kernel walks each with contiguous
    // strides and reads the scales as raw bytes
    check_bank(x_in, w1, s1);
    check_bank(x_in, w3, s3);
    TORCH_CHECK(w3.sizes() == w1.sizes(), "w1 and w3 disagree");
    E = w1.size(0);
    I = w1.size(1);
    TORCH_CHECK(I % 32 == 0, "the expert's inner size must be a multiple of 32");
    TORCH_CHECK(w2.dtype() == torch::kUInt8 && w2.dim() == 3 && w2.is_contiguous(), "w2 must be contiguous uint8 [E, K, I/2]");
    TORCH_CHECK(s2.dtype() == torch::kFloat8_e8m0fnu && s2.dim() == 3 && s2.is_contiguous(), "s2 must be contiguous e8m0 [E, K, I/32]");
    TORCH_CHECK(w2.size(0) == E && w2.size(1) == K && w2.size(2) * 2 == I, "w2 must be [E, K, I/2]");
    TORCH_CHECK(s2.size(0) == E && s2.size(1) == K && s2.size(2) * 32 == I, "s2 must be [E, K, I/32]");
    TORCH_CHECK(w2.device().is_cpu() && s2.device().is_cpu(), "CPU tensors only");
  }
  auto wc = weights.to(torch::kFloat32).contiguous();
  auto ic = indices.to(torch::kInt64).contiguous();
  TORCH_CHECK(wc.dim() == 2 && wc.size(0) == n && ic.sizes() == wc.sizes(), "weights and indices must be [n, topk]");
  const int64_t topk = wc.size(1), rows = n * topk;
  const float* wp = wc.data_ptr<float>();
  const int64_t* sel = ic.data_ptr<int64_t>();
  for (int64_t r = 0; r < rows; ++r) TORCH_CHECK(sel[r] >= 0 && sel[r] < E, "expert index out of range");

  XRows xr(x_in);  // fp32 and de-interleaved copies of the n input rows
  const int64_t* w1_ptrs = per_expert ? w1.data_ptr<int64_t>() : nullptr;
  const int64_t* s1_ptrs = per_expert ? s1.data_ptr<int64_t>() : nullptr;
  const int64_t* w3_ptrs = per_expert ? w3.data_ptr<int64_t>() : nullptr;
  const int64_t* s3_ptrs = per_expert ? s3.data_ptr<int64_t>() : nullptr;
  const int64_t* w2_ptrs = per_expert ? w2.data_ptr<int64_t>() : nullptr;
  const int64_t* s2_ptrs = per_expert ? s2.data_ptr<int64_t>() : nullptr;

  const uint8_t* w1p = per_expert ? nullptr : w1.data_ptr<uint8_t>();
  const uint8_t* w3p = per_expert ? nullptr : w3.data_ptr<uint8_t>();
  const uint8_t* w2p = per_expert ? nullptr : w2.data_ptr<uint8_t>();
  const uint8_t* s1p = per_expert ? nullptr : reinterpret_cast<const uint8_t*>(s1.data_ptr());
  const uint8_t* s3p = per_expert ? nullptr : reinterpret_cast<const uint8_t*>(s3.data_ptr());
  const uint8_t* s2p = per_expert ? nullptr : reinterpret_cast<const uint8_t*>(s2.data_ptr());

  // phase 1: gate and up for every (token, k) pair, each expert row read once per projection
  std::vector<float> gate(rows * I), up(rows * I);
  const int64_t blk = 16, nb = (I + blk - 1) / blk;
  parallel_range(rows * nb, 1, [&](int64_t t0, int64_t t1) {
    for (int64_t t = t0; t < t1; ++t) {
      const int64_t r = t / nb, i0 = (t % nb) * blk, tok = r / topk, e = sel[r];
      const float* xf = xr.full.data() + tok * K;
      const float* xe = xr.even.empty() ? nullptr : xr.even.data() + tok * (K / 2);
      const float* xo = xr.odd.empty() ? nullptr : xr.odd.data() + tok * (K / 2);
      const uint8_t* ew1 = per_expert ? reinterpret_cast<const uint8_t*>(w1_ptrs[e]) : (w1p + e * I * (K / 2));
      const uint8_t* es1 = per_expert ? reinterpret_cast<const uint8_t*>(s1_ptrs[e]) : (s1p + e * I * (K / 32));
      const uint8_t* ew3 = per_expert ? reinterpret_cast<const uint8_t*>(w3_ptrs[e]) : (w3p + e * I * (K / 2));
      const uint8_t* es3 = per_expert ? reinterpret_cast<const uint8_t*>(s3_ptrs[e]) : (s3p + e * I * (K / 32));
      for (int64_t i = i0; i < std::min(I, i0 + blk); ++i) {
        // rounded to bf16 as the reference's bf16 matmul output is
        gate[r * I + i] = static_cast<float>(static_cast<at::BFloat16>(dot_packed(xf, xe, xo, ew1 + i * (K / 2), es1 + i * (K / 32), K)));
        up[r * I + i] = static_cast<float>(static_cast<at::BFloat16>(dot_packed(xf, xe, xo, ew3 + i * (K / 2), es3 + i * (K / 32), K)));
      }
    }
  });

  // phase 2: SwiGLU with the clamps, times the routing weight, bf16, then
  // the fp8 grid (act_quant), one 32-element block per work item: the
  // quantization is per block, and a decode step has only topk rows, so
  // one item per row left all but six threads idle through the exps
  const float lim = static_cast<float>(swiglu_limit);
  std::vector<float> hf(rows * I), he(rows * I / 2), ho(rows * I / 2);
  const int64_t nblk = I / 32;
  parallel_range(rows * nblk, 1, [&](int64_t q0, int64_t q1) {
    at::BFloat16 hb[32];
    for (int64_t q = q0; q < q1; ++q) {
      const int64_t r = q / nblk, i0 = (q % nblk) * 32;
      const float wgt = wp[r];
      for (int64_t j = 0; j < 32; ++j) {
        float g = gate[r * I + i0 + j], u = up[r * I + i0 + j];
        if (lim > 0.f) {
          u = std::min(lim, std::max(-lim, u));
          g = std::min(lim, g);
        }
        const float silu = g / (1.0f + std::exp(-g));
        hb[j] = static_cast<at::BFloat16>(wgt * (silu * u));
      }
      float* out = hf.data() + r * I + i0;
      act_quant_row(hb, 32, out);
      for (int64_t j = 0; j < 16; ++j) {
        he[r * (I / 2) + i0 / 2 + j] = out[2 * j];
        ho[r * (I / 2) + i0 / 2 + j] = out[2 * j + 1];
      }
    }
  });

  // phase 3: the down projection for every output row and every (token, k), summed in k order
  auto y = torch::zeros({n, K}, x_in.options().dtype(torch::kFloat32));
  float* yp = y.data_ptr<float>();
  const int64_t kb = (K + blk - 1) / blk;
  parallel_range(n * kb, 1, [&](int64_t t0, int64_t t1) {
    for (int64_t t = t0; t < t1; ++t) {
      const int64_t tok = t / kb, o0 = (t % kb) * blk;
      for (int64_t o = o0; o < std::min(K, o0 + blk); ++o) {
        float acc = 0.f;
        for (int64_t k = 0; k < topk; ++k) {
          const int64_t r = tok * topk + k, e = sel[r];
          const uint8_t* ew2 = per_expert ? reinterpret_cast<const uint8_t*>(w2_ptrs[e]) : (w2p + e * K * (I / 2));
          const uint8_t* es2 = per_expert ? reinterpret_cast<const uint8_t*>(s2_ptrs[e]) : (s2p + e * K * (I / 32));
          const float d = dot_packed(hf.data() + r * I, he.data() + r * (I / 2), ho.data() + r * (I / 2),
                                     ew2 + o * (I / 2), es2 + o * (I / 32), I);
          acc += static_cast<float>(static_cast<at::BFloat16>(d));  // the bf16 down output, summed in fp32
        }
        yp[tok * K + o] = acc;
      }
    }
  });
  return y;
}

bool has_avx512() { return have_avx512(); }
bool has_neon() {
#if defined(__aarch64__)
  return true;
#else
  return false;
#endif
}

// Rows of a 2-D table, gathered by index into a new tensor, one row per
// task across the kernel threads. For a table memory-mapped from local SSD
// (the n-gram tables, #364) a row not in the page cache is a page fault
// that blocks only the thread copying it, so the step's reads proceed
// concurrently instead of one after another: the SSDs serve ~1.3M random
// 4 KiB reads a second with many in flight, ~13k one at a time.
torch::Tensor gather_rows(const torch::Tensor& table, const torch::Tensor& indices) {
  TORCH_CHECK(table.dim() == 2 && table.is_contiguous() && table.device().is_cpu(),
              "gather_rows: a contiguous 2-D CPU table");
  // the indices are read on the host below: a device tensor's address would be dereferenced here
  TORCH_CHECK(indices.device().is_cpu(), "gather_rows: indices on the CPU (got ", indices.device(), ")");
  const auto idx = indices.to(torch::kLong).contiguous().view(-1);
  const int64_t n = idx.numel(), rows = table.size(0);
  const int64_t row_bytes = table.size(1) * table.element_size();
  const int64_t* ix = idx.data_ptr<int64_t>();
  for (int64_t i = 0; i < n; ++i) {
    TORCH_CHECK(ix[i] >= 0 && ix[i] < rows, "gather_rows: index ", ix[i], " outside [0, ", rows, ")");
  }
  auto out = torch::empty({n, table.size(1)}, table.options());
  const char* src = static_cast<const char*>(table.data_ptr());
  char* dst = static_cast<char*>(out.data_ptr());
  parallel_range(n, 1, [&](int64_t i0, int64_t i1) {
    for (int64_t i = i0; i < i1; ++i) {
      std::memcpy(dst + i * row_bytes, src + ix[i] * row_bytes, row_bytes);
    }
  });
  auto shape = indices.sizes().vec();
  shape.push_back(table.size(1));
  return out.view(shape);
}

static bool has_io_uring() {
#if defined(VXPU_HAVE_IO_URING)
  return true;
#else
  return false;
#endif
}

#if defined(VXPU_HAVE_IO_URING)
struct RingState {
  io_uring ring;
  bool inited = false;
  bool failed = false;
  std::mutex mu;

  bool init() {
    std::lock_guard<std::mutex> lock(mu);
    if (inited) return true;
    if (failed) return false;
    if (io_uring_queue_init(512, &ring, 0) < 0) {
      failed = true;
      return false;
    }
    inited = true;
    return true;
  }

  void destroy() {
    if (inited) {
      io_uring_queue_exit(&ring);
      inited = false;
    }
    failed = true;
  }

  ~RingState() {
    if (inited) {
      io_uring_queue_exit(&ring);
    }
  }
};
static RingState g_ring;
#endif

bool io_uring_ready() {
#if defined(VXPU_HAVE_IO_URING)
  return g_ring.init();
#else
  return false;
#endif
}

// Rows of an n-gram table's weights and scales read from a checkpoint file
// at their byte offsets in a single io_uring submit (and pread across kernel
// threads as a fallback).
std::tuple<torch::Tensor, torch::Tensor> gather_table_rows(
    int fd,
    int64_t weight_offset,
    int64_t weight_row_bytes,
    int64_t scale_offset,
    int64_t scale_row_bytes,
    int64_t rows,
    const torch::Tensor& indices,
    bool use_uring = true) {
  TORCH_CHECK(fd >= 0, "gather_table_rows: invalid file descriptor ", fd);
  TORCH_CHECK(weight_row_bytes > 0, "gather_table_rows: weight_row_bytes must be > 0 (got ", weight_row_bytes, ")");
  TORCH_CHECK(scale_row_bytes > 0, "gather_table_rows: scale_row_bytes must be > 0 (got ", scale_row_bytes, ")");
  TORCH_CHECK(weight_offset >= 0, "gather_table_rows: weight_offset must be >= 0 (got ", weight_offset, ")");
  TORCH_CHECK(scale_offset >= 0, "gather_table_rows: scale_offset must be >= 0 (got ", scale_offset, ")");
  TORCH_CHECK(rows > 0, "gather_table_rows: rows must be > 0 (got ", rows, ")");
  TORCH_CHECK(indices.device().is_cpu(), "gather_table_rows: indices on the CPU (got ", indices.device(), ")");

  const auto idx = indices.to(torch::kLong).contiguous().view(-1);
  const int64_t n = idx.numel();
  const int64_t* ix = idx.data_ptr<int64_t>();
  for (int64_t i = 0; i < n; ++i) {
    TORCH_CHECK(ix[i] >= 0 && ix[i] < rows, "gather_table_rows: index ", ix[i], " outside [0, ", rows, ")");
  }

  auto out_weights = torch::empty({n, weight_row_bytes}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
  auto out_scales = torch::empty({n, scale_row_bytes}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
  char* dst_w = static_cast<char*>(out_weights.data_ptr());
  char* dst_s = static_cast<char*>(out_scales.data_ptr());

  auto shape = indices.sizes().vec();
  auto shape_w = shape; shape_w.push_back(weight_row_bytes);
  auto shape_s = shape; shape_s.push_back(scale_row_bytes);

  if (n == 0) {
    return {out_weights.view(shape_w), out_scales.view(shape_s)};
  }

  bool ran_uring = false;
#if defined(VXPU_HAVE_IO_URING)
  if (use_uring && g_ring.init()) {
    std::lock_guard<std::mutex> lock(g_ring.mu);
    int64_t submitted_pairs = 0;
    while (submitted_pairs < n) {
      int64_t batch_pairs = std::min<int64_t>(n - submitted_pairs, 256);
      int64_t batch_sqes = batch_pairs * 2;
      unsigned space = io_uring_sq_space_left(&g_ring.ring);
      TORCH_CHECK(space >= batch_sqes, "gather_table_rows: io_uring submission queue full (space: ", space, ", needed: ", batch_sqes, ")");

      for (int64_t i = 0; i < batch_pairs; ++i) {
        int64_t item = submitted_pairs + i;
        io_uring_sqe* sqe_w = io_uring_get_sqe(&g_ring.ring);
        TORCH_CHECK(sqe_w != nullptr, "gather_table_rows: io_uring_get_sqe returned null");
        io_uring_prep_read(sqe_w, fd, dst_w + item * weight_row_bytes, weight_row_bytes, weight_offset + ix[item] * weight_row_bytes);
        sqe_w->user_data = static_cast<uint64_t>(weight_row_bytes);

        io_uring_sqe* sqe_s = io_uring_get_sqe(&g_ring.ring);
        TORCH_CHECK(sqe_s != nullptr, "gather_table_rows: io_uring_get_sqe returned null");
        io_uring_prep_read(sqe_s, fd, dst_s + item * scale_row_bytes, scale_row_bytes, scale_offset + ix[item] * scale_row_bytes);
        sqe_s->user_data = static_cast<uint64_t>(scale_row_bytes);
      }

      int to_submit = batch_sqes;
      while (to_submit > 0) {
        int ret = io_uring_submit(&g_ring.ring);
        if (ret == -EINTR) {
          continue;
        }
        if (ret < 0) {
          g_ring.destroy();
          TORCH_CHECK(false, "gather_table_rows: io_uring_submit failed: ", std::strerror(-ret));
        }
        to_submit -= ret;
      }

      int64_t seen = 0;
      int err = 0;
      int64_t short_expected = 0;
      int64_t short_got = 0;
      while (seen < batch_sqes) {
        io_uring_cqe* cqe = nullptr;
        int wait_ret = io_uring_wait_cqe(&g_ring.ring, &cqe);
        if (wait_ret == -EINTR) {
          continue;
        }
        if (wait_ret < 0) {
          g_ring.destroy();
          TORCH_CHECK(false, "gather_table_rows: io_uring_wait_cqe failed: ", std::strerror(-wait_ret));
        }
        if (cqe->res < 0 && err == 0) {
          err = -cqe->res;
        } else if (cqe->user_data != 0) {
          int64_t expected_bytes = static_cast<int64_t>(cqe->user_data);
          if (cqe->res != expected_bytes && err == 0) {
            err = -1;
            short_expected = expected_bytes;
            short_got = cqe->res;
          }
        }
        seen++;
        io_uring_cqe_seen(&g_ring.ring, cqe);
      }
      if (err > 0) {
        TORCH_CHECK(false, "gather_table_rows: read error: ", std::strerror(err));
      } else if (err < 0) {
        TORCH_CHECK(false, "gather_table_rows: short read: expected ", short_expected, " bytes, got ", short_got);
      }
      submitted_pairs += batch_pairs;
    }
    ran_uring = true;
  }
#endif

  if (!ran_uring) {
    std::atomic<bool> failed{false};
    std::atomic<int> fail_errno{0};
    std::atomic<int64_t> short_expected{0};
    std::atomic<int64_t> short_got{0};
    auto record_fail = [&](int e, int64_t exp, int64_t got) {
      bool expected_failed = false;
      if (failed.compare_exchange_strong(expected_failed, true)) {
        fail_errno.store(e);
        short_expected.store(exp);
        short_got.store(got);
      }
    };

    parallel_range(n, 1, [&](int64_t i0, int64_t i1) {
      for (int64_t i = i0; i < i1; ++i) {
        if (failed.load(std::memory_order_relaxed)) return;
        int64_t w_off = weight_offset + ix[i] * weight_row_bytes;
        ssize_t got_w = pread(fd, dst_w + i * weight_row_bytes, weight_row_bytes, w_off);
        if (got_w < 0) {
          record_fail(errno, 0, 0);
          return;
        }
        if (got_w != weight_row_bytes) {
          record_fail(0, weight_row_bytes, got_w);
          return;
        }
        int64_t s_off = scale_offset + ix[i] * scale_row_bytes;
        ssize_t got_s = pread(fd, dst_s + i * scale_row_bytes, scale_row_bytes, s_off);
        if (got_s < 0) {
          record_fail(errno, 0, 0);
          return;
        }
        if (got_s != scale_row_bytes) {
          record_fail(0, scale_row_bytes, got_s);
          return;
        }
      }
    });

    if (failed.load()) {
      if (fail_errno.load() != 0) {
        TORCH_CHECK(false, "gather_table_rows: pread failed on fd ", fd, ": ", std::strerror(fail_errno.load()));
      } else {
        TORCH_CHECK(false, "gather_table_rows: short read: expected ", short_expected.load(), " bytes, got ", short_got.load());
      }
    }
  }

  return {out_weights.view(shape_w), out_scales.view(shape_s)};
}

// The kernels a node's native package calls (#394): torch ops, so the
// C++ AOTInductor generates for a node calls them through the dispatcher,
// with no Python between messages. Loading this library registers them.
// The scales arrive as their raw bytes (uint8): AOTInductor's CPU code
// cannot name an e8m0 tensor's type, and the kernel reads them as bytes.
static torch::Tensor fp4_moe_decode_bytes(torch::Tensor x_in, torch::Tensor weights, torch::Tensor indices,
                                          torch::Tensor w1, torch::Tensor s1, torch::Tensor w3, torch::Tensor s3,
                                          torch::Tensor w2, torch::Tensor s2, double swiglu_limit,
                                          int64_t intermediate_size) {
  auto e8m0 = [](const torch::Tensor& s) { return s.dim() == 1 ? s : s.view(torch::kFloat8_e8m0fnu); };
  return fp4_moe_decode(x_in, weights, indices, w1, e8m0(s1), w3, e8m0(s3), w2, e8m0(s2), swiglu_limit, intermediate_size);
}

// The pairs no GPU pool holds, one reply per pair (the step graph's
// experts node on a layer with pools, #409): [n, topk, K] fp32, zero where
// held[t, k]. The cold pairs go through fp4_moe_decode as single-pair
// rows, gathered in row-major order, exactly as mailboxes.CpuOwner sends
// them, so the replies are the same bits. The data-dependent count of cold
// pairs stays inside the op: the package around it sees fixed shapes.
static torch::Tensor fp4_moe_decode_cold_bytes(torch::Tensor x_in, torch::Tensor weights, torch::Tensor indices,
                                               torch::Tensor held, torch::Tensor w1, torch::Tensor s1,
                                               torch::Tensor w3, torch::Tensor s3, torch::Tensor w2,
                                               torch::Tensor s2, double swiglu_limit,
                                               int64_t intermediate_size) {
  const int64_t n = indices.size(0), topk = indices.size(1);
  auto out = torch::zeros({n, topk, x_in.size(1)}, x_in.options().dtype(torch::kFloat32));
  auto cold = held.logical_not();
  if (!cold.any().item<bool>()) return out;
  auto tokens = torch::arange(n, indices.options().dtype(torch::kInt64)).view({n, 1}).expand({n, topk}).index({cold});
  auto replies = fp4_moe_decode_bytes(x_in.index_select(0, tokens).contiguous(),
                                      weights.index({cold}).to(torch::kFloat32).view({-1, 1}).contiguous(),
                                      indices.index({cold}).to(torch::kInt64).view({-1, 1}).contiguous(),
                                      w1, s1, w3, s3, w2, s2, swiglu_limit, intermediate_size);
  out.index_put_({cold}, replies);
  return out;
}

TORCH_LIBRARY(vxpu_native, m) {
  m.def("fp4_moe_decode(Tensor x_in, Tensor weights, Tensor indices, Tensor w1, Tensor s1, Tensor w3, Tensor s3, "
        "Tensor w2, Tensor s2, float swiglu_limit, int intermediate_size=0) -> Tensor");
  m.def("fp4_moe_decode_cold(Tensor x_in, Tensor weights, Tensor indices, Tensor held, Tensor w1, Tensor s1, "
        "Tensor w3, Tensor s3, Tensor w2, Tensor s2, float swiglu_limit, int intermediate_size=0) -> Tensor");
  m.def("gather_rows(Tensor table, Tensor indices) -> Tensor");
}

TORCH_LIBRARY_IMPL(vxpu_native, CPU, m) {
  m.impl("fp4_moe_decode", &fp4_moe_decode_bytes);
  m.impl("fp4_moe_decode_cold", &fp4_moe_decode_cold_bytes);
  m.impl("gather_rows", &gather_rows);
}

void register_node_server(pybind11::module& m);  // node_server.cpp: node server, node client, step host

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fp4_gather_mm", &fp4_gather_mm, "x[M,K] bf16 x packed fp4 experts sel[M] -> bf16 [M,N]",
        py::arg("x"), py::arg("w"), py::arg("scale"), py::arg("sel"), py::arg("intermediate_size") = 0);
  m.def("fp4_grouped_mm", &fp4_grouped_mm, "x[M,K] bf16 (sorted by expert, offs[E]) x packed fp4 experts -> bf16 [M,N]",
        py::arg("x"), py::arg("w"), py::arg("scale"), py::arg("offs"), py::arg("intermediate_size") = 0);
  m.def("fp4_dequant", &fp4_dequant, "one expert of a packed bank -> bf16 [N,K]",
        py::arg("w"), py::arg("scale"), py::arg("e"), py::arg("intermediate_size") = 0, py::arg("dim") = 0);
  m.def("fp4_moe_decode", &fp4_moe_decode, "one decode layer's routed experts: gate/up, SwiGLU, act_quant, down, sum -> fp32 [n, K]",
        py::arg("x_in"), py::arg("weights"), py::arg("indices"), py::arg("w1"), py::arg("s1"),
        py::arg("w3"), py::arg("s3"), py::arg("w2"), py::arg("s2"), py::arg("swiglu_limit"),
        py::arg("intermediate_size") = 0);
  m.def("fp8_linear", &fp8_linear, "x[M,K] bf16 (quantized per 32-block inside, as the fork's act_quant) x fp8 [N,K] with e8m0 32x32 block scales -> bf16 [M,N]");
  m.def("gather_rows", &gather_rows, "rows of a 2-D table by index, copied concurrently (page faults overlap) -> [*indices.shape, cols]");
  m.def("gather_table_rows", &gather_table_rows,
        "weights and scales of an n-gram table read directly with io_uring in one submit -> ([*indices.shape, weight_row_bytes], [*indices.shape, scale_row_bytes])",
        py::arg("fd"), py::arg("weight_offset"), py::arg("weight_row_bytes"),
        py::arg("scale_offset"), py::arg("scale_row_bytes"), py::arg("rows"), py::arg("indices"),
        py::arg("use_uring") = true);
  m.def("io_uring_ready", &io_uring_ready, "whether io_uring is available and its ring is initialized");
  m.def("has_io_uring", &has_io_uring, "whether io_uring support was compiled in");
  m.def("has_avx512", &has_avx512, "whether the AVX-512 paths are in use");
  m.def("has_neon", &has_neon, "whether the NEON paths are in use");
  register_node_server(m);
}
