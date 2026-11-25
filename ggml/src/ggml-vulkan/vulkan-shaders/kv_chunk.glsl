#if !defined(GGML_KV_CHUNK_GLSL)
#define GGML_KV_CHUNK_GLSL

// Chunked K/V (dual-buffer fork, dynamic KV on Vulkan): a K or V section is a list of ordinary device buffers (chunks) of
// a whole number of K/V rows each. A shader variant built with KV_CHUNKED gets, in place of the tensor's binding, the
// section's chunk table: a header (the section offset the binding starts at, the bytes per chunk and their fastdiv
// values) and the device address of every chunk. Every K/V access goes through kvc_addr() (the one address computation)
// and a buffer reference of the element type. Without KV_CHUNKED nothing here is compiled and the shaders are unchanged.

#if defined(KV_CHUNKED)
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

// h.x: the section offset of the binding (the descriptor offset of an ordinary buffer), h.y: bytes per chunk,
// h.z, h.w: the fastdiv multiplier and shift of h.y
#define KVC_TABLE(BINDING, NAME) \
    layout (binding = BINDING) readonly buffer NAME##_table { uvec4 h; uint64_t a[]; } NAME;

// the device address of byte `off` of a binding (relative to its start, as the index of an ordinary binding)
uint64_t kvc_addr(const uvec4 h, const uint off, const uint chunk, const uint64_t chunk_addr) {
    return chunk_addr + uint64_t(h.x + off - chunk * h.y);
}

// the chunk of byte `off` of a binding
uint kvc_chunk(const uvec4 h, const uint off) {
    const uint s = h.x + off;
    uint hi, lo;
    umulExtended(s, h.z, hi, lo);
    return uint((uint64_t(hi) + uint64_t(s)) >> h.w);
}

#define KVC_ADDR(NAME, OFF) kvc_addr(NAME.h, (OFF), kvc_chunk(NAME.h, (OFF)), NAME.a[kvc_chunk(NAME.h, (OFF))])

// a buffer reference type for elements of type T at addresses aligned to ALIGN
#define KVC_REF(REF, T, ALIGN) \
    layout (buffer_reference, std430, buffer_reference_align = ALIGN) buffer REF { T v; };

// element I (of SIZE bytes) of the binding whose table is NAME, through a reference of type REF
#define KVC_AT(REF, NAME, I, SIZE) REF(KVC_ADDR(NAME, uint(I) * uint(SIZE))).v

// a chunked source and/or destination of an element-wise shader (copy, set_rows, rope, rms_norm): the variant defines
// the binding, the element type, its size and its alignment; KVC_SRC_AT(i) / KVC_DST_AT(i) is element i
#if defined(KVC_SRC_BINDING)
KVC_TABLE(KVC_SRC_BINDING, kvc_src)
KVC_REF(kvc_src_ref, KVC_SRC_T, KVC_SRC_ALIGN)
#define KVC_SRC_AT(I) KVC_AT(kvc_src_ref, kvc_src, I, KVC_SRC_SIZE)
#endif
#if defined(KVC_DST_BINDING)
KVC_TABLE(KVC_DST_BINDING, kvc_dst)
KVC_REF(kvc_dst_ref, KVC_DST_T, KVC_DST_ALIGN)
#define KVC_DST_AT(I) KVC_AT(kvc_dst_ref, kvc_dst, I, KVC_DST_SIZE)
#endif

#endif // KV_CHUNKED

#endif // GGML_KV_CHUNK_GLSL
