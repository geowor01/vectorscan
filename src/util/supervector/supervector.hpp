/*
 * Copyright (c) 2015-2017, Intel Corporation
 * Copyright (c) 2020-2021, VectorCamp PC
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of Intel Corporation nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef SUPERVECTOR_HPP
#define SUPERVECTOR_HPP

#include <cstdint>
#include <cstdio>

#if defined(ARCH_IA32) || defined(ARCH_X86_64)
#include "util/supervector/arch/x86/types.hpp"
#elif defined(ARCH_ARM32) || defined(ARCH_AARCH64)
#include "util/supervector/arch/arm/types.hpp"
#endif

#if defined(HAVE_SIMD_512_BITS)
using Z_TYPE = u64a;
#define Z_BITS 64
#define Z_SHIFT 63
#define DOUBLE_LOAD_MASK(l)        ((~0ULL) >> (Z_BITS -(l)))
#define SINGLE_LOAD_MASK(l)        (((1ULL) << (l)) - 1ULL)
#elif defined(HAVE_SIMD_256_BITS)
using Z_TYPE = u32;
#define Z_BITS 32
#define Z_SHIFT 31
#define DOUBLE_LOAD_MASK(l)        (((1ULL) << (l)) - 1ULL)
#define SINGLE_LOAD_MASK(l)        (((1ULL) << (l)) - 1ULL)
#elif defined(HAVE_SIMD_128_BITS)
using Z_TYPE = u32;
#define Z_BITS 32
#define Z_SHIFT 15
#define DOUBLE_LOAD_MASK(l)        (((1ULL) << (l)) - 1ULL)
#define SINGLE_LOAD_MASK(l)        (((1ULL) << (l)) - 1ULL)
#endif

// Define a common assume_aligned using an appropriate compiler built-in, if
// it's available. Note that we need to handle C or C++ compilation.
#ifdef __cplusplus
#  ifdef HAVE_CXX_BUILTIN_ASSUME_ALIGNED
#    define assume_aligned(x, y) __builtin_assume_aligned((x), (y))
#  endif
#else
#  ifdef HAVE_CC_BUILTIN_ASSUME_ALIGNED
#    define assume_aligned(x, y) __builtin_assume_aligned((x), (y))
#  endif
#endif

// Fallback to identity case.
#ifndef assume_aligned
#define assume_aligned(x, y) (x)
#endif

template <uint16_t SIZE>
class SuperVector;

using m128_t  = SuperVector<16>;
using m256_t  = SuperVector<32>;
using m512_t  = SuperVector<64>;
using m1024_t = SuperVector<128>;

// struct for inferring what underlying types to use
template <int T>
struct BaseVector
{
  static const bool is_valid = false;  // for template matches specialisation
  using type                 = void;
  using movemask_type        = uint32_t;
  using previous_type        = void;
};

template <>
struct BaseVector<128>
{
  static constexpr bool     is_valid  = true;
  static constexpr uint16_t size      = 128;
  using type                          = void;
  using movemask_type                 = u64a;
};

template <>
struct BaseVector<64>
{
  static constexpr bool     is_valid  = true;
  static constexpr uint16_t size      = 64;
  using type                          = m512;
  using movemask_type                 = u64a;
};

// 128 bit implementation
template <>
struct BaseVector<32>
{
  static constexpr bool     is_valid  = true;
  static constexpr uint16_t size      = 32;
  using type                          = m256;
  using movemask_type                 = u32;
};

// 128 bit implementation
template <>
struct BaseVector<16>
{
  static constexpr bool     is_valid  = true;
  static constexpr uint16_t size      = 16;
  using type                          = m128;
  using movemask_type                 = u32;
};

template <uint16_t SIZE>
class SuperVector : public BaseVector<SIZE>
{
  static_assert(BaseVector<SIZE>::is_valid, "invalid SuperVector size");

public:

  using base_type      = BaseVector<SIZE>;

  union {
    typename BaseVector<16>::type ALIGN_ATTR(BaseVector<16>::size) v128[SIZE / BaseVector<16>::size];
    typename BaseVector<32>::type ALIGN_ATTR(BaseVector<32>::size) v256[SIZE / BaseVector<32>::size];
    typename BaseVector<64>::type ALIGN_ATTR(BaseVector<64>::size) v512[SIZE / BaseVector<64>::size];
    uint64_t u64[SIZE / sizeof(uint64_t)];
    int64_t  s64[SIZE / sizeof(int64_t)];
    uint32_t u32[SIZE / sizeof(uint32_t)];
    int32_t  s32[SIZE / sizeof(int32_t)];
    uint16_t u16[SIZE / sizeof(uint16_t)];
    int16_t  s16[SIZE / sizeof(int16_t)];
    uint8_t  u8[SIZE / sizeof(uint8_t)];
    int8_t   s8[SIZE / sizeof(int8_t)];
    float    f32[SIZE / sizeof(float)];
    double   f64[SIZE / sizeof(double)];
  } u;

  SuperVector() {};
  SuperVector(SuperVector const &other);
  SuperVector(typename base_type::type const v);

  template<typename T>
  SuperVector(T const other);

  static SuperVector dup_u8 (uint8_t  other) { return {other}; };
  static SuperVector dup_s8 (int8_t   other) { return {other}; };
  static SuperVector dup_u16(uint16_t other) { return {other}; };
  static SuperVector dup_s16(int16_t  other) { return {other}; };
  static SuperVector dup_u32(uint32_t other) { return {other}; };
  static SuperVector dup_s32(int32_t  other) { return {other}; };
  static SuperVector dup_u64(uint64_t other) { return {other}; };
  static SuperVector dup_s64(int64_t  other) { return {other}; };

  void operator=(SuperVector const &other);

  SuperVector operator&(SuperVector const &b) const;
  SuperVector operator|(SuperVector const &b) const;
  SuperVector operator^(SuperVector const &b) const;

  SuperVector opand(SuperVector const &b) const { return *this & b; }
  SuperVector opor (SuperVector const &b) const { return *this | b; }
  SuperVector opxor(SuperVector const &b) const { return *this ^ b; }
  SuperVector opandnot(SuperVector const &b) const;

  SuperVector eq(SuperVector const &b) const;
  SuperVector operator<<(uint8_t const N) const;
  SuperVector operator>>(uint8_t const N) const;
  typename base_type::movemask_type movemask(void) const;
  typename base_type::movemask_type eqmask(SuperVector const b) const;

  static SuperVector loadu(void const *ptr);
  static SuperVector load(void const *ptr);
  static SuperVector loadu_maskz(void const *ptr, uint8_t const len);
  SuperVector alignr(SuperVector &other, int8_t offset);

  SuperVector pshufb(SuperVector b);
  SuperVector lshift64(uint8_t const N);
  SuperVector rshift64(uint8_t const N);
  SuperVector lshift128(uint8_t const N);
  SuperVector rshift128(uint8_t const N);
  SuperVector lshift128_var(uint8_t const N) const;
  SuperVector rshift128_var(uint8_t const N) const;

  // Constants
  static SuperVector Ones();
  static SuperVector Zeroes();

  #if defined(DEBUG)
  void print8(const char *label) {
      printf("%12s: ", label);
      for(s16 i=SIZE-1; i >= 0; i--)
          printf("%02x ", u.u8[i]);
      printf("\n");
  }

  void print16(const char *label) {
      printf("%12s: ", label);
      for(s16 i=SIZE/sizeof(u16)-1; i >= 0; i--)
          printf("%04x ", u.u16[i]);
      printf("\n");
  }

  void print32(const char *label) {
      printf("%12s: ", label);
      for(s16 i=SIZE/sizeof(u32)-1; i >= 0; i--)
          printf("%08x ", u.u32[i]);
      printf("\n");
  }

  void print64(const char *label) {
      printf("%12s: ", label);
      for(s16 i=SIZE/sizeof(u64a)-1; i >= 0; i--)
          printf("%016lx ", u.u64[i]);
      printf("\n");
  }
#else
  void print8(const char *label UNUSED) {};
  void print16(const char *label UNUSED) {};
  void print32(const char *label UNUSED) {};
  void print64(const char *label UNUSED) {};
#endif
};

#if defined(HS_OPTIMIZE)
#if defined(ARCH_IA32) || defined(ARCH_X86_64)
#include "util/supervector/arch/x86/impl.cpp"
#elif defined(ARCH_ARM32) || defined(ARCH_AARCH64)
#include "util/supervector/arch/arm/impl.cpp"
#endif
#endif

#endif /* SUPERVECTOR_H */

