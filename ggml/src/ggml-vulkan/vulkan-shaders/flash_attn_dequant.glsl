// Asymmetric K/V flash attention: aliased SSBO views of bindings 1 (K) and 2 (V)
// covering every supported FA element type, plus an uber dequantize4() that
// switches on FaTypeK / FaTypeV. After spec-constant specialization the driver
// folds away every path except the one matching the K/V type for this pipeline.
//
// Included by flash_attn.comp and flash_attn_cm1.comp. Not included by
// flash_attn_cm2.comp, which has its own buffer_reference-based decode path.
//
// We use macros (rather than per-quant decode functions taking a struct) on
// purpose: the FA shaders don't enable GL_EXT_shader_explicit_arithmetic_types_float16
// when FLOAT16 isn't defined, which makes float16-containing struct values
// illegal to return from / pass to functions. Macros expand inline where the
// float16 stays in storage and is converted to FLOAT_TYPE at use.

// F32 is fed as a vec4 "block" (4 floats), matching what dequant_funcs_cm2.glsl
// does for F32 in the cm2 shader. FaBlockBytesK/V == 16 for F32.
layout (binding = 1) readonly buffer K_PACKED_F32  { vec4 data[]; }                k_packed_f32;
layout (binding = 2) readonly buffer V_PACKED_F32  { vec4 data[]; }                v_packed_f32;

layout (binding = 1) readonly buffer K_PACKED_Q4_0 { block_q4_0_packed16 data[]; } k_packed_q4_0;
layout (binding = 2) readonly buffer V_PACKED_Q4_0 { block_q4_0_packed16 data[]; } v_packed_q4_0;
layout (binding = 1) readonly buffer K_PACKED_Q4_1 { block_q4_1_packed16 data[]; } k_packed_q4_1;
layout (binding = 2) readonly buffer V_PACKED_Q4_1 { block_q4_1_packed16 data[]; } v_packed_q4_1;
layout (binding = 1) readonly buffer K_PACKED_Q5_0 { block_q5_0_packed16 data[]; } k_packed_q5_0;
layout (binding = 2) readonly buffer V_PACKED_Q5_0 { block_q5_0_packed16 data[]; } v_packed_q5_0;
layout (binding = 1) readonly buffer K_PACKED_Q5_1 { block_q5_1_packed16 data[]; } k_packed_q5_1;
layout (binding = 2) readonly buffer V_PACKED_Q5_1 { block_q5_1_packed16 data[]; } v_packed_q5_1;
layout (binding = 1) readonly buffer K_PACKED_Q8_0 { block_q8_0_packed16 data[]; } k_packed_q8_0;
layout (binding = 2) readonly buffer V_PACKED_Q8_0 { block_q8_0_packed16 data[]; } v_packed_q8_0;
layout (binding = 1) readonly buffer K_PACKED_IQ4_NL { block_iq4_nl_packed16 data[]; } k_packed_iq4_nl;
layout (binding = 2) readonly buffer V_PACKED_IQ4_NL { block_iq4_nl_packed16 data[]; } v_packed_iq4_nl;

layout (binding = 1) readonly buffer K_PACKED_BF16 { u16vec4 data[]; } k_packed_bf16;
layout (binding = 2) readonly buffer V_PACKED_BF16 { u16vec4 data[]; } v_packed_bf16;

// Q4_1 and Q5_1 packed32 views: aliased to the same memory as the packed16
// views, used by the MMQ K-side hot path for fast 4-uint loads.
layout (binding = 1) readonly buffer K_PACKED_Q4_1_P32 { block_q4_1_packed32 data[]; } k_packed_q4_1_p32;
layout (binding = 1) readonly buffer K_PACKED_Q5_1_P32 { block_q5_1_packed32 data[]; } k_packed_q5_1_p32;

// FA_BLK(BUF, i): block i of the K or V view BUF. Chunked K/V (KV_CHUNKED, dual-buffer fork): bindings 1 and 2 are the
// chunk tables of K and V and block i is read through its device address (kv_chunk.glsl).
#if defined(KV_CHUNKED)
#include "kv_chunk.glsl"
KVC_TABLE(1, kvc_k)
KVC_TABLE(2, kvc_v)
KVC_REF(kvc_vec4,    vec4,                  16)
KVC_REF(kvc_u16vec4, u16vec4,               8)
KVC_REF(kvc_q4_0,    block_q4_0_packed16,   2)
KVC_REF(kvc_q4_1,    block_q4_1_packed16,   2)
KVC_REF(kvc_q4_1_32, block_q4_1_packed32,   4)
KVC_REF(kvc_q5_0,    block_q5_0_packed16,   2)
KVC_REF(kvc_q5_1,    block_q5_1_packed16,   4)
KVC_REF(kvc_q5_1_32, block_q5_1_packed32,   4)
KVC_REF(kvc_q8_0,    block_q8_0_packed16,   2)
KVC_REF(kvc_iq4_nl,  block_iq4_nl_packed16, 2)
#define k_packed_f32_at(i)      KVC_AT(kvc_vec4,    kvc_k, i, 16)
#define v_packed_f32_at(i)      KVC_AT(kvc_vec4,    kvc_v, i, 16)
#define k_packed_bf16_at(i)     KVC_AT(kvc_u16vec4, kvc_k, i, 8)
#define v_packed_bf16_at(i)     KVC_AT(kvc_u16vec4, kvc_v, i, 8)
#define k_packed_q4_0_at(i)     KVC_AT(kvc_q4_0,    kvc_k, i, 18)
#define v_packed_q4_0_at(i)     KVC_AT(kvc_q4_0,    kvc_v, i, 18)
#define k_packed_q4_1_at(i)     KVC_AT(kvc_q4_1,    kvc_k, i, 20)
#define v_packed_q4_1_at(i)     KVC_AT(kvc_q4_1,    kvc_v, i, 20)
#define k_packed_q4_1_p32_at(i) KVC_AT(kvc_q4_1_32, kvc_k, i, 20)
#define k_packed_q5_0_at(i)     KVC_AT(kvc_q5_0,    kvc_k, i, 22)
#define v_packed_q5_0_at(i)     KVC_AT(kvc_q5_0,    kvc_v, i, 22)
#define k_packed_q5_1_at(i)     KVC_AT(kvc_q5_1,    kvc_k, i, 24)
#define v_packed_q5_1_at(i)     KVC_AT(kvc_q5_1,    kvc_v, i, 24)
#define k_packed_q5_1_p32_at(i) KVC_AT(kvc_q5_1_32, kvc_k, i, 24)
#define k_packed_q8_0_at(i)     KVC_AT(kvc_q8_0,    kvc_k, i, 34)
#define v_packed_q8_0_at(i)     KVC_AT(kvc_q8_0,    kvc_v, i, 34)
#define k_packed_iq4_nl_at(i)   KVC_AT(kvc_iq4_nl,  kvc_k, i, 18)
#define v_packed_iq4_nl_at(i)   KVC_AT(kvc_iq4_nl,  kvc_v, i, 18)
#define FA_BLK(BUF, I) BUF##_at(I)
#else
#define FA_BLK(BUF, I) BUF.data[I]
#endif

// Per-quant decode bodies are expanded once for the K view set and once for
// the V view set. The macros take the buffer name as a parameter.
#define FA_DEQUANT4_F32(BUF) \
    return FLOAT_TYPEV4(FA_BLK(BUF, a_offset + ib));

#define FA_DEQUANT4_Q4_0(BUF) {                                                                   \
    uint vui_lo = uint(FA_BLK(BUF, a_offset + ib).qs[(iqs & 0xF) / 2 + 0]);                          \
    uint vui_hi = uint(FA_BLK(BUF, a_offset + ib).qs[(iqs & 0xF) / 2 + 1]);                          \
    uint shift = (iqs & 0x10) >> 2;                                                               \
    vui_lo >>= shift;                                                                             \
    vui_hi >>= shift;                                                                             \
    FLOAT_TYPEV4 nibbles = FLOAT_TYPEV4(vui_lo & 0xF, (vui_lo >> 8) & 0xF,                        \
                                        vui_hi & 0xF, (vui_hi >> 8) & 0xF);                       \
    return FLOAT_TYPE(FA_BLK(BUF, a_offset + ib).d) * (nibbles - FLOAT_TYPE(8.0f));                  \
}

#define FA_DEQUANT4_Q4_1(BUF) {                                                                   \
    uint vui_lo = uint(FA_BLK(BUF, a_offset + ib).qs[(iqs & 0xF) / 2 + 0]);                          \
    uint vui_hi = uint(FA_BLK(BUF, a_offset + ib).qs[(iqs & 0xF) / 2 + 1]);                          \
    uint shift = (iqs & 0x10) >> 2;                                                               \
    vui_lo >>= shift;                                                                             \
    vui_hi >>= shift;                                                                             \
    FLOAT_TYPEV4 nibbles = FLOAT_TYPEV4(vui_lo & 0xF, (vui_lo >> 8) & 0xF,                        \
                                        vui_hi & 0xF, (vui_hi >> 8) & 0xF);                       \
    return FLOAT_TYPE(FA_BLK(BUF, a_offset + ib).d) * nibbles                                        \
         + FLOAT_TYPE(FA_BLK(BUF, a_offset + ib).m);                                                 \
}

#define FA_DEQUANT4_Q5_0(BUF) {                                                                   \
    uint vui_lo = uint(FA_BLK(BUF, a_offset + ib).qs[(iqs & 0xF) / 2 + 0]);                          \
    uint vui_hi = uint(FA_BLK(BUF, a_offset + ib).qs[(iqs & 0xF) / 2 + 1]);                          \
    uint shift = (iqs & 0x10) >> 2;                                                               \
    vui_lo >>= shift;                                                                             \
    vui_hi >>= shift;                                                                             \
    uint qh = uint(FA_BLK(BUF, a_offset + ib).qh[0])                                                 \
            | (uint(FA_BLK(BUF, a_offset + ib).qh[1]) << 16);                                        \
    FLOAT_TYPEV4 hb = FLOAT_TYPEV4((qh >> iqs)       & 1, (qh >> (iqs + 1)) & 1,                  \
                                   (qh >> (iqs + 2)) & 1, (qh >> (iqs + 3)) & 1)                  \
                      * FLOAT_TYPE(16.0f);                                                        \
    FLOAT_TYPEV4 nibbles = FLOAT_TYPEV4(vui_lo & 0xF, (vui_lo >> 8) & 0xF,                        \
                                        vui_hi & 0xF, (vui_hi >> 8) & 0xF);                       \
    return FLOAT_TYPE(FA_BLK(BUF, a_offset + ib).d) * (nibbles + hb - FLOAT_TYPE(16.0f));            \
}

#define FA_DEQUANT4_Q5_1(BUF) {                                                                   \
    uint vui_lo = uint(FA_BLK(BUF, a_offset + ib).qs[(iqs & 0xF) / 2 + 0]);                          \
    uint vui_hi = uint(FA_BLK(BUF, a_offset + ib).qs[(iqs & 0xF) / 2 + 1]);                          \
    uint shift = (iqs & 0x10) >> 2;                                                               \
    vui_lo >>= shift;                                                                             \
    vui_hi >>= shift;                                                                             \
    uint qh = FA_BLK(BUF, a_offset + ib).qh;                                                         \
    FLOAT_TYPEV4 hb = FLOAT_TYPEV4((qh >> iqs)       & 1, (qh >> (iqs + 1)) & 1,                  \
                                   (qh >> (iqs + 2)) & 1, (qh >> (iqs + 3)) & 1)                  \
                      * FLOAT_TYPE(16.0f);                                                        \
    FLOAT_TYPEV4 nibbles = FLOAT_TYPEV4(vui_lo & 0xF, (vui_lo >> 8) & 0xF,                        \
                                        vui_hi & 0xF, (vui_hi >> 8) & 0xF);                       \
    return FLOAT_TYPE(FA_BLK(BUF, a_offset + ib).d) * (nibbles + hb)                                 \
         + FLOAT_TYPE(FA_BLK(BUF, a_offset + ib).m);                                                 \
}

#define FA_DEQUANT4_Q8_0(BUF) {                                                                   \
    const i8vec2 v0 = unpack8(int32_t(FA_BLK(BUF, a_offset + ib).qs[iqs / 2    ])).xy;               \
    const i8vec2 v1 = unpack8(int32_t(FA_BLK(BUF, a_offset + ib).qs[iqs / 2 + 1])).xy;               \
    return FLOAT_TYPE(FA_BLK(BUF, a_offset + ib).d) * FLOAT_TYPEV4(v0.x, v0.y, v1.x, v1.y);          \
}

#define FA_DEQUANT4_IQ4_NL(BUF) {                                                                 \
    const uint shift = (iqs & 0x10) >> 2;                                                         \
    const uint qs_i = (iqs & 0xC) >> 1;                                                           \
    const uint qsw = uint(FA_BLK(BUF, a_offset + ib).qs[qs_i])                                       \
                   | (uint(FA_BLK(BUF, a_offset + ib).qs[qs_i + 1u]) << 16);                         \
    const FLOAT_TYPE d = FLOAT_TYPE(FA_BLK(BUF, a_offset + ib).d);                                   \
    const u8vec4 q = unpack8((qsw >> shift) & 0x0F0F0F0Fu);                                       \
    return d * FLOAT_TYPEV4(kvalues_iq4nl[q.x], kvalues_iq4nl[q.y],                               \
                            kvalues_iq4nl[q.z], kvalues_iq4nl[q.w]);                              \
}

#define FA_DEQUANT4_BF16(BUF) \
    return FLOAT_TYPEV4(bf16_to_fp32(uvec4(FA_BLK(BUF, (a_offset + ib) / 4))));

FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    if (binding_idx == BINDING_IDX_K) {
        switch (FaTypeK) {
            case GGML_TYPE_F32:  FA_DEQUANT4_F32 (k_packed_f32)
            case GGML_TYPE_Q4_0: FA_DEQUANT4_Q4_0(k_packed_q4_0)
            case GGML_TYPE_Q4_1: FA_DEQUANT4_Q4_1(k_packed_q4_1)
            case GGML_TYPE_Q5_0: FA_DEQUANT4_Q5_0(k_packed_q5_0)
            case GGML_TYPE_Q5_1: FA_DEQUANT4_Q5_1(k_packed_q5_1)
            case GGML_TYPE_Q8_0: FA_DEQUANT4_Q8_0(k_packed_q8_0)
            case GGML_TYPE_IQ4_NL: FA_DEQUANT4_IQ4_NL(k_packed_iq4_nl)
            case GGML_TYPE_BF16: FA_DEQUANT4_BF16(k_packed_bf16)
        }
    } else {
        switch (FaTypeV) {
            case GGML_TYPE_F32:  FA_DEQUANT4_F32 (v_packed_f32)
            case GGML_TYPE_Q4_0: FA_DEQUANT4_Q4_0(v_packed_q4_0)
            case GGML_TYPE_Q4_1: FA_DEQUANT4_Q4_1(v_packed_q4_1)
            case GGML_TYPE_Q5_0: FA_DEQUANT4_Q5_0(v_packed_q5_0)
            case GGML_TYPE_Q5_1: FA_DEQUANT4_Q5_1(v_packed_q5_1)
            case GGML_TYPE_Q8_0: FA_DEQUANT4_Q8_0(v_packed_q8_0)
            case GGML_TYPE_IQ4_NL: FA_DEQUANT4_IQ4_NL(v_packed_iq4_nl)
            case GGML_TYPE_BF16: FA_DEQUANT4_BF16(v_packed_bf16)
        }
    }
    return FLOAT_TYPEV4(0);
}
