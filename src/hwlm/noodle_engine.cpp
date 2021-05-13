/*
 * Copyright (c) 2015-2017, Intel Corporation
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

/** \file
 * \brief Noodle literal matcher: runtime.
 */
#include "hwlm.h"
#include "noodle_engine.h"
#include "noodle_internal.h"
#include "scratch.h"
#include "ue2common.h"
#include "util/arch.h"
#include "util/bitutils.h"
#include "util/compare.h"
#include "util/intrinsics.h"
#include "util/join.h"
#include "util/partial_store.h"
#include "util/simd_utils.h"

#if defined(HAVE_AVX2)
#include "util/arch/x86/masked_move.h"
#endif

#include <ctype.h>
#include <stdbool.h>
#include <string.h>

/** \brief Noodle runtime context. */
struct cb_info {
    HWLMCallback cb; //!< callback function called on match
    u32 id; //!< ID to pass to callback on match
    struct hs_scratch *scratch; //!< scratch to pass to callback
    size_t offsetAdj; //!< used in streaming mode
};


#include "noodle_engine_simd.hpp"

#define RETURN_IF_TERMINATED(x)                                                \
    {                                                                          \
        if ((x) == HWLM_TERMINATED) {                                          \
            return HWLM_TERMINATED;                                            \
        }                                                                      \
    }

// Make sure the rest of the string is there. The single character scanner
// is used only for single chars with case insensitivity used correctly,
// so it can go straight to the callback if we get this far.
static really_inline
hwlm_error_t final(const struct noodTable *n, const u8 *buf, UNUSED size_t len,
                   char single, const struct cb_info *cbi, size_t pos) {
    u64a v{0};
    if (single) {
        if (n->msk_len == 1) {
            goto match;
        }
    }
    assert(len >= n->msk_len);
    v = partial_load_u64a(buf + pos + n->key_offset - n->msk_len, n->msk_len);
    DEBUG_PRINTF("v %016llx msk %016llx cmp %016llx\n", v, n->msk, n->cmp);
    if ((v & n->msk) != n->cmp) {
        /* mask didn't match */
        return HWLM_SUCCESS;
    }

match:
    pos -= cbi->offsetAdj;
    DEBUG_PRINTF("match @ %zu\n", pos + n->key_offset);
    hwlmcb_rv_t rv = cbi->cb(pos + n->key_offset - 1, cbi->id, cbi->scratch);
    if (rv == HWLM_TERMINATE_MATCHING) {
        return HWLM_TERMINATED;
    }
    return HWLM_SUCCESS;
}

static really_really_inline
hwlm_error_t single_zscan(const struct noodTable *n,const u8 *d, const u8 *buf,
		Z_TYPE z, size_t len, const struct cb_info *cbi) {
    while (unlikely(z)) {
        Z_TYPE pos = JOIN(findAndClearLSB_, Z_BITS)(&z);
        size_t matchPos = d - buf + pos;
        DEBUG_PRINTF("match pos %zu\n", matchPos);
        hwlmcb_rv_t rv = final(n, buf, len, 1, cbi, matchPos);
        RETURN_IF_TERMINATED(rv);
    }
    return HWLM_SUCCESS;
}

static really_really_inline
hwlm_error_t double_zscan(const struct noodTable *n,const u8 *d, const u8 *buf,
		Z_TYPE z, size_t len, const struct cb_info *cbi) {
    while (unlikely(z)) {
        Z_TYPE pos = JOIN(findAndClearLSB_, Z_BITS)(&z);
        size_t matchPos = d - buf + pos - 1;                               \
        DEBUG_PRINTF("match pos %zu\n", matchPos);
        hwlmcb_rv_t rv = final(n, buf, len, 0, cbi, matchPos);
        RETURN_IF_TERMINATED(rv);
    }
    return HWLM_SUCCESS;
}

template <uint16_t S>
static really_inline
hwlm_error_t scanSingleMain(const struct noodTable *n, const u8 *buf,
                            size_t len, size_t offset,
                            SuperVector<S> caseMask, SuperVector<S> mask1,
                            const struct cb_info *cbi) {
    size_t start = offset + n->msk_len - 1;
    size_t end = len;

    const u8 *d = buf + start;
    const u8 *e = buf + end;
    DEBUG_PRINTF("start %p end %p \n", d, e);
    assert(d < e);
    if (d + S <= e) {
        // peel off first part to cacheline boundary
        const u8 *d1 = ROUNDUP_PTR(d, S);
        DEBUG_PRINTF("until aligned %p \n", d1);
        if (scanSingleUnaligned(n, buf, caseMask, mask1, cbi, len, start, d1 - buf) == HWLM_TERMINATED) {
            return HWLM_TERMINATED;
        }
        d = d1;

        size_t loops = (end - (d - buf)) / S;
        DEBUG_PRINTF("loops %ld \n", loops);

        for (size_t i = 0; i < loops; i++, d+= S) {
            DEBUG_PRINTF("d %p \n", d);
            const u8 *base = ROUNDUP_PTR(d, 64);
            // On large packet buffers, this prefetch appears to get us about 2%.
            __builtin_prefetch(base + 256);

            SuperVector<S> v = SuperVector<S>::load(d) & caseMask;
            typename SuperVector<S>::movemask_type z = mask1.eqmask(v);

            hwlm_error_t rv = single_zscan(n, d, buf, z, len, cbi);
            RETURN_IF_TERMINATED(rv);
        }
    }

    DEBUG_PRINTF("d %p e %p \n", d, e);
    // finish off tail

    return scanSingleUnaligned(n, buf, caseMask, mask1, cbi, len, d - buf, end);
}

template <uint16_t S>
static really_inline
hwlm_error_t scanDoubleMain(const struct noodTable *n, const u8 *buf,
                            size_t len, size_t offset, 
                            SuperVector<S> caseMask, SuperVector<S> mask1, SuperVector<S> mask2,
                            const struct cb_info *cbi) {
    // we stop scanning for the key-fragment when the rest of the key can't
    // possibly fit in the remaining buffer
    size_t end = len - n->key_offset + 2;

    size_t start = offset + n->msk_len - n->key_offset;

    typename SuperVector<S>::movemask_type lastz1{0};

    const u8 *d = buf + start;
    const u8 *e = buf + end;
    DEBUG_PRINTF("start %p end %p \n", d, e);
    assert(d < e);
    if (d + S <= e) {
        // peel off first part to cacheline boundary
        const u8 *d1 = ROUNDUP_PTR(d, S);
        DEBUG_PRINTF("until aligned %p \n", d1);
        if (scanDoubleUnaligned(n, buf, caseMask, mask1, mask2, &lastz1, cbi, len, start, d1 - buf) == HWLM_TERMINATED) {
            return HWLM_TERMINATED;
        }
        d = d1;

        size_t loops = (end - (d - buf)) / S;
        DEBUG_PRINTF("loops %ld \n", loops);

        for (size_t i = 0; i < loops; i++, d+= S) {
            DEBUG_PRINTF("d %p \n", d);
            const u8 *base = ROUNDUP_PTR(d, 64);
            // On large packet buffers, this prefetch appears to get us about 2%.
            __builtin_prefetch(base + 256);

            SuperVector<S> v = SuperVector<S>::load(d) & caseMask;
            typename SuperVector<S>::movemask_type z1 = mask1.eqmask(v);
            typename SuperVector<S>::movemask_type z2 = mask2.eqmask(v);
            typename SuperVector<S>::movemask_type z = (z1 << 1 | lastz1) & z2;
            lastz1 = z1 >> Z_SHIFT;

            hwlm_error_t rv = double_zscan(n, d, buf, z, len, cbi);
            RETURN_IF_TERMINATED(rv);
        }
    }

    DEBUG_PRINTF("d %p e %p \n", d, e);
    // finish off tail

    return scanDoubleUnaligned(n, buf, caseMask, mask1, mask2, &lastz1, cbi, len, d - buf, end);
}

// Single-character specialisation, used when keyLen = 1
static really_inline
hwlm_error_t scanSingle(const struct noodTable *n, const u8 *buf, size_t len,
                        size_t start, bool noCase, const struct cb_info *cbi) {
    if (!ourisalpha(n->key0)) {
        noCase = 0; // force noCase off if we don't have an alphabetic char
    }

    const SuperVector<VECTORSIZE> caseMask{noCase ? getCaseMask<VECTORSIZE>() : SuperVector<VECTORSIZE>::Ones()};
    const SuperVector<VECTORSIZE> mask1{getMask<VECTORSIZE>(n->key0, noCase)};

    return scanSingleMain(n, buf, len, start, caseMask, mask1, cbi);
}


static really_inline
hwlm_error_t scanDouble(const struct noodTable *n, const u8 *buf, size_t len,
                        size_t start, bool noCase, const struct cb_info *cbi) {

    const SuperVector<VECTORSIZE> caseMask{noCase ? getCaseMask<VECTORSIZE>() : SuperVector<VECTORSIZE>::Ones()};
    const SuperVector<VECTORSIZE> mask1{getMask<VECTORSIZE>(n->key0, noCase)};
    const SuperVector<VECTORSIZE> mask2{getMask<VECTORSIZE>(n->key1, noCase)};

    return scanDoubleMain(n, buf, len, start, caseMask, mask1, mask2, cbi);
}

// main entry point for the scan code
static really_inline
hwlm_error_t scan(const struct noodTable *n, const u8 *buf, size_t len,
                  size_t start, char single, bool noCase,
                  const struct cb_info *cbi) {
    if (len - start < n->msk_len) {
        // can't find string of length keyLen in a shorter buffer
        return HWLM_SUCCESS;
    }

    if (single) {
        return scanSingle(n, buf, len, start, noCase, cbi);
    } else {
        return scanDouble(n, buf, len, start, noCase, cbi);
    }
}

/** \brief Block-mode scanner. */
hwlm_error_t noodExec(const struct noodTable *n, const u8 *buf, size_t len,
                      size_t start, HWLMCallback cb,
                      struct hs_scratch *scratch) {
    assert(n && buf);

    struct cb_info cbi = {cb, n->id, scratch, 0};
    DEBUG_PRINTF("nood scan of %zu bytes for %*s @ %p\n", len, n->msk_len,
                 (const char *)&n->cmp, buf);

    return scan(n, buf, len, start, n->single, n->nocase, &cbi);
}

/** \brief Streaming-mode scanner. */
hwlm_error_t noodExecStreaming(const struct noodTable *n, const u8 *hbuf,
                               size_t hlen, const u8 *buf, size_t len,
                               HWLMCallback cb, struct hs_scratch *scratch) {
    assert(n);

    if (len + hlen < n->msk_len) {
        DEBUG_PRINTF("not enough bytes for a match\n");
        return HWLM_SUCCESS;
    }

    struct cb_info cbi = {cb, n->id, scratch, 0};
    DEBUG_PRINTF("nood scan of %zu bytes (%zu hlen) for %*s @ %p\n", len, hlen,
                 n->msk_len, (const char *)&n->cmp, buf);

    if (hlen && n->msk_len > 1) {
        /*
         * we have history, so build up a buffer from enough of the history
         * buffer plus what we've been given to scan. Since this is relatively
         * short, just check against msk+cmp per byte offset for matches.
         */
        assert(hbuf);
        u8 ALIGN_DIRECTIVE temp_buf[HWLM_LITERAL_MAX_LEN * 2];
        memset(temp_buf, 0, sizeof(temp_buf));

        assert(n->msk_len);
        size_t tl1 = MIN((size_t)n->msk_len - 1, hlen);
        size_t tl2 = MIN((size_t)n->msk_len - 1, len);

        assert(tl1 + tl2 <= sizeof(temp_buf));
        assert(tl1 + tl2 >= n->msk_len);
        assert(tl1 <= sizeof(u64a));
        assert(tl2 <= sizeof(u64a));
        DEBUG_PRINTF("using %zu bytes of hist and %zu bytes of buf\n", tl1, tl2);

        unaligned_store_u64a(temp_buf,
                             partial_load_u64a(hbuf + hlen - tl1, tl1));
        unaligned_store_u64a(temp_buf + tl1, partial_load_u64a(buf, tl2));

        for (size_t i = 0; i <= tl1 + tl2 - n->msk_len; i++) {
            u64a v = unaligned_load_u64a(temp_buf + i);
            if ((v & n->msk) == n->cmp) {
                size_t m_end = -tl1 + i + n->msk_len - 1;
                DEBUG_PRINTF("match @ %zu (i %zu)\n", m_end, i);
                hwlmcb_rv_t rv = cb(m_end, n->id, scratch);
                if (rv == HWLM_TERMINATE_MATCHING) {
                    return HWLM_TERMINATED;
                }
            }
        }
    }

    assert(buf);

    cbi.offsetAdj = 0;
    return scan(n, buf, len, 0, n->single, n->nocase, &cbi);
}
