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
#include "fetch.h"
#include <c10/util/Float8_e4m3fn.h>
#include <ATen/Parallel.h>

#include <cmath>
#include <cstdint>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <vector>

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
  int64_t chunks = std::min(n / std::max<int64_t>(grain, 1), threads * 4);
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

torch::Tensor fp4_gather_mm(torch::Tensor x, torch::Tensor w, torch::Tensor scale, torch::Tensor sel) {
  check_bank(x, w, scale);
  TORCH_CHECK(sel.dtype() == torch::kInt64 && sel.dim() == 1 && sel.size(0) == x.size(0), "sel must be int64 [M]");
  const int64_t M = x.size(0), K = x.size(1), E = w.size(0), N = w.size(1);
  auto out = torch::empty({M, N}, x.options());
  XRows xr(x);
  auto selc = sel.contiguous();
  const int64_t* selp = selc.data_ptr<int64_t>();
  for (int64_t i = 0; i < M; ++i) TORCH_CHECK(selp[i] >= 0 && selp[i] < E, "expert index out of range");
  const uint8_t* wp = w.data_ptr<uint8_t>();
  const uint8_t* sp = reinterpret_cast<const uint8_t*>(scale.data_ptr());
  at::BFloat16* op = out.data_ptr<at::BFloat16>();
  const int64_t nblock = 16;  // output rows per task
  const int64_t nb = (N + nblock - 1) / nblock;
  parallel_range(M * nb, 1, [&](int64_t t0, int64_t t1) {
    for (int64_t t = t0; t < t1; ++t) {
      const int64_t i = t / nb, n0 = (t % nb) * nblock, e = selp[i];
      const uint8_t* we = wp + e * N * (K / 2);
      const uint8_t* se = sp + e * N * (K / 32);
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
void fp4_dequant_into(const torch::Tensor& w, const torch::Tensor& scale, int64_t e, torch::Tensor& out) {
  const int64_t N = w.size(1), K = w.size(2) * 2;
  const uint8_t* wp = w.data_ptr<uint8_t>() + e * N * (K / 2);
  const uint8_t* sp = reinterpret_cast<const uint8_t*>(scale.data_ptr()) + e * N * (K / 32);
  uint16_t* op = reinterpret_cast<uint16_t*>(out.data_ptr<at::BFloat16>());
  parallel_range(N, 8, [&](int64_t n0, int64_t n1) {
    for (int64_t n = n0; n < n1; ++n) decode_row(wp + n * (K / 2), sp + n * (K / 32), K, op + n * K);
  });
}

torch::Tensor fp4_dequant(torch::Tensor w, torch::Tensor scale, int64_t e) {
  TORCH_CHECK(w.dtype() == torch::kUInt8 && w.dim() == 3 && w.is_contiguous(), "w must be contiguous uint8 [E, N, K/2]");
  TORCH_CHECK(e >= 0 && e < w.size(0), "expert index out of range");
  const int64_t N = w.size(1), K = w.size(2) * 2;
  auto out = torch::empty({N, K}, w.options().dtype(torch::kBFloat16));
  fp4_dequant_into(w, scale, e, out);
  return out;
}

torch::Tensor fp4_grouped_mm(torch::Tensor x, torch::Tensor w, torch::Tensor scale, torch::Tensor offs) {
  check_bank(x, w, scale);
  TORCH_CHECK(offs.dtype() == torch::kInt32 && offs.dim() == 1 && offs.size(0) == w.size(0), "offs must be int32 [E]");
  const int64_t M = x.size(0), E = w.size(0), N = w.size(1);
  auto out = torch::empty({M, N}, x.options());
  auto offc = offs.contiguous();
  const int32_t* offp = offc.data_ptr<int32_t>();
  TORCH_CHECK(E == 0 || offp[E - 1] == M, "offs must end at M");
  // one decode buffer, reused for every expert (a fresh allocation per
  // expert would pay its page faults every time)
  auto we = torch::empty({N, x.size(1)}, x.options());
  int64_t r0 = 0;
  for (int64_t e = 0; e < E; ++e) {
    const int64_t r1 = offp[e];
    if (r1 > r0) {
      // decoded once per expert; the GEMM over its tokens is torch's bf16 mm
      fp4_dequant_into(w, scale, e, we);
      auto rows = out.slice(0, r0, r1);
      at::mm_out(rows, x.slice(0, r0, r1), we.t());
    }
    r0 = r1;
  }
  return out;
}

// ---------------------------------------------------------------- fp8 dense

// e4m3fn byte -> fp32 (no infinities; 0x7F/0xFF are NaN and never occur in weights)
inline float e4m3(uint8_t b) {
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
  return _mm512_castsi512_ps(_mm512_or_si512(_mm512_castps_si512(val), sign));
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

bool has_avx512() { return have_avx512(); }
bool has_neon() {
#if defined(__aarch64__)
  return true;
#else
  return false;
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  register_fetch(m);
  m.def("fp4_gather_mm", &fp4_gather_mm, "x[M,K] bf16 x packed fp4 experts sel[M] -> bf16 [M,N]");
  m.def("fp4_grouped_mm", &fp4_grouped_mm, "x[M,K] bf16 (sorted by expert, offs[E]) x packed fp4 experts -> bf16 [M,N]");
  m.def("fp4_dequant", &fp4_dequant, "one expert of a packed bank -> bf16 [N,K]");
  m.def("fp8_linear", &fp8_linear, "x[M,K] bf16 (quantized per 32-block inside, as the fork's act_quant) x fp8 [N,K] with e8m0 32x32 block scales -> bf16 [M,N]");
  m.def("has_avx512", &has_avx512, "whether the AVX-512 paths are in use");
  m.def("has_neon", &has_neon, "whether the NEON paths are in use");
}
