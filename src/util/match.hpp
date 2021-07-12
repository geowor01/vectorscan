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

#include "ue2common.h"
#include "util/arch.h"
#include "util/bitutils.h"
#include "util/unaligned.h"

#include "util/supervector/supervector.hpp"

template <u16 S>
const u8 *firstMatch(const u8 *buf, typename SuperVector<S>::movemask_type z);

template <u16 S>
const u8 *lastMatch(const u8 *buf, typename SuperVector<S>::movemask_type z);

template <>
really_really_inline
const u8 *firstMatch<16>(const u8 *buf, typename SuperVector<16>::movemask_type z) {
    DEBUG_PRINTF("buf %p z %08x \n", buf, z);
    DEBUG_PRINTF("z %08x\n", z);
    if (unlikely(z != 0xffff)) {
        u32 pos = ctz32(~z & 0xffff);
        DEBUG_PRINTF("~z %08x\n", ~z);
        DEBUG_PRINTF("match @ pos %u\n", pos);
        assert(pos < 16);
        return buf + pos;
    } else {
        return NULL; // no match
    }
}

template <>
really_really_inline
const u8 *firstMatch<32>(const u8 *buf, typename SuperVector<32>::movemask_type z) {
    DEBUG_PRINTF("z 0x%08x\n", z);
    if (unlikely(z != 0xffffffff)) {
        u32 pos = ctz32(~z);
        assert(pos < 32);
        DEBUG_PRINTF("match @ pos %u\n", pos);
        return buf + pos;
    } else {
        return NULL; // no match
    }
}
template <>
really_really_inline
const u8 *firstMatch<64>(const u8 *buf, typename SuperVector<64>::movemask_type z) {
    DEBUG_PRINTF("z 0x%016llx\n", z);
    if (unlikely(z != ~0ULL)) {
        u32 pos = ctz64(~z);
        DEBUG_PRINTF("match @ pos %u\n", pos);
        assert(pos < 64);
        return buf + pos;
    } else {
        return NULL; // no match
    }
}

template <>
really_really_inline
const u8 *lastMatch<16>(const u8 *buf, typename SuperVector<16>::movemask_type z) {
    DEBUG_PRINTF("buf %p z %08x \n", buf, z);
    DEBUG_PRINTF("z %08x\n", z);
    if (unlikely(z != 0xffff)) {
        u32 pos = clz32(~z & 0xffff);
        DEBUG_PRINTF("~z %08x\n", ~z);
        DEBUG_PRINTF("match @ pos %u\n", pos);
        assert(pos >= 16 && pos < 32);
        return buf + (31 - pos);
    } else {
        return NULL; // no match
    }
}

template<>
really_really_inline
const u8 *lastMatch<32>(const u8 *buf, typename SuperVector<32>::movemask_type z) {
    if (unlikely(z != 0xffffffff)) {
        u32 pos = clz32(~z);
        DEBUG_PRINTF("buf=%p, pos=%u\n", buf, pos);
        assert(pos < 32);
        return buf + (31 - pos);
    } else {
        return NULL; // no match
    }
}

template <>
really_really_inline
const u8 *lastMatch<64>(const u8 *buf, typename SuperVector<64>::movemask_type z) {
    DEBUG_PRINTF("z 0x%016llx\n", z);
    if (unlikely(z != ~0ULL)) {
        u32 pos = clz64(~z);
        DEBUG_PRINTF("match @ pos %u\n", pos);
        assert(pos < 64);
        return buf + pos;
    } else {
        return NULL; // no match
    }
}

