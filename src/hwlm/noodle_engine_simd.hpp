/*
 * Copyright (c) 2017, Intel Corporation
 * Copyright (c) 2020, 2021, VectorCamp PC
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

/* SIMD engine agnostic noodle scan parts */

#include "util/simd/types.hpp"

// using Z_TYPE = typename SuperVector<VECTORSIZE>::movemask_type;

#if defined(HAVE_SIMD_512_BITS)
using Z_TYPE = u64a;
#define Z_BITS 64
#define Z_SHIFT 63
#define DOUBLE_LOAD_MASK(l)        ((~0ULL) >> (Z_BITS -l)) 
#define SINGLE_LOAD_MASK(l)        (((1ULL) << l) - 1ULL)
#elif defined(HAVE_SIMD_256_BITS)
using Z_TYPE = u32;
#define Z_BITS 32
#define Z_SHIFT 31
#define DOUBLE_LOAD_MASK(l)        (((1ULL) << l) - 1ULL)
#define SINGLE_LOAD_MASK(l)        (((1ULL) << l) - 1ULL)
#elif defined(HAVE_SIMD_128_BITS)
using Z_TYPE = u32;
#define Z_BITS 32
#define Z_SHIFT 0
#define DOUBLE_LOAD_MASK(l)        (((1ULL) << l) - 1ULL)
#define SINGLE_LOAD_MASK(l)        (((1ULL) << l) - 1ULL)
#endif

static u8 CASEMASK[] = { 0xff, 0xdf };

static really_inline
u8 caseClear8(u8 x, bool noCase)
{
    return static_cast<u8>(x & CASEMASK[(u8)noCase]);
}

template<uint16_t S>
static really_inline SuperVector<S> getMask(u8 c, bool noCase) {
    u8 k = caseClear8(c, noCase);
    return SuperVector<S>(k);
}

template<uint16_t S>
static really_inline SuperVector<S> getCaseMask(void) {
    return SuperVector<S>(CASEMASK[1]);
}

// The short scan routine. It is used both to scan data up to an
// alignment boundary if needed and to finish off data that the aligned scan
// function can't handle (due to small/unaligned chunk at end)
template<uint16_t S>
static really_inline
hwlm_error_t scanSingleUnaligned(const struct noodTable *n, const u8 *buf,
                                 SuperVector<S> caseMask, SuperVector<S> mask1,
                                 const struct cb_info *cbi, size_t len, size_t start,
                                 size_t end) {
    const u8 *d = buf + start;
    DEBUG_PRINTF("start %zu end %zu\n", start, end);
    const size_t l = end - start;
    DEBUG_PRINTF("l = %ld\n", l);
    //assert(l <= 64);
    if (!l) {
        return HWLM_SUCCESS;
    }

    typename SuperVector<S>::movemask_type mask = SINGLE_LOAD_MASK(l);
    SuperVector<S> v = SuperVector<S>::loadu(d) & caseMask;
    typename SuperVector<S>::movemask_type z = mask & mask1.eqmask(v);

    return single_zscan(n, d, buf, z, len, cbi);
}

template<uint16_t S>
static really_inline
hwlm_error_t scanDoubleUnaligned(const struct noodTable *n, const u8 *buf,
                                 SuperVector<S> caseMask, SuperVector<S> mask1, SuperVector<S> mask2, typename SuperVector<S>::movemask_type *lastz1,
                                 const struct cb_info *cbi, size_t len, size_t start, size_t end) {
    const u8 *d = buf + start;
    DEBUG_PRINTF("start %zu end %zu", start, end);
    const size_t l = end - start;
    assert(l <= S);
    if (!l) {
        return HWLM_SUCCESS;
    }
    SuperVector<S> v = SuperVector<S>::loadu(d) & caseMask;

    typename SuperVector<S>::movemask_type mask = DOUBLE_LOAD_MASK(l);
    typename SuperVector<S>::movemask_type z1 = mask1.eqmask(v);
    typename SuperVector<S>::movemask_type z2 = mask2.eqmask(v);
    typename SuperVector<S>::movemask_type z = mask & (*lastz1 | z1 << 1) & z2;
    *lastz1 = z1 >> (l -1);

    return double_zscan(n, d, buf, z, len, cbi);
}
