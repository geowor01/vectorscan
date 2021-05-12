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

#ifndef SIMD_TYPES_HPP
#define SIMD_TYPES_HPP

#include <cstdint>

#if defined(ARCH_IA32) || defined(ARCH_X86_64)
#include "util/simd/arch/x86/types.hpp"
#elif defined(ARCH_ARM32) || defined(ARCH_AARCH64)
#include "util/simd/arch/arm/types.hpp"
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

  SuperVector(SuperVector const &o);
  SuperVector(typename base_type::type const v);

  template<typename T>
  SuperVector(T const o);

  void operator=(SuperVector const &o);
  SuperVector operator&(SuperVector const b) const;
  SuperVector eq(SuperVector const b) const;
  SuperVector operator<<(uint8_t const N) const;
  typename base_type::movemask_type movemask(void) const;
  typename base_type::movemask_type eqmask(SuperVector const b) const;
  static SuperVector loadu(void const *ptr);
  static SuperVector load(void const *ptr);
  SuperVector alignr(SuperVector l, int8_t offset);

  // Constants
  static SuperVector Ones();
  static SuperVector Zeroes();
};

//class SuperVector<16>;
// class SuperVector<32>;
// class SuperVector<64>;
// class SuperVector<128>;

#if defined(ARCH_IA32) || defined(ARCH_X86_64)
#include "util/simd/arch/x86/impl.hpp"
#elif defined(ARCH_ARM32) || defined(ARCH_AARCH64)
#include "util/simd/arch/arm/impl.hpp"
#endif


#endif /* SIMD_TYPES_H */

