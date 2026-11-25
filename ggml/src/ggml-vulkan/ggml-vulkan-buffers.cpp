#include "ggml-vulkan-common.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <map>
#include <numeric>
#include <set>
#include <cstdlib>
#include <mutex>
#include <shared_mutex>

static void * portable_aligned_malloc(size_t align, size_t size) {
#ifdef _WIN32
    return _aligned_malloc(size, align);
#else
    void * ptr = nullptr;
    if (posix_memalign(&ptr, align, size) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

static void portable_aligned_free(void * ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

ggml_backend_buffer_type_i ggml_backend_vk_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_vk_buffer_type_name,
    /* .alloc_buffer        = */ ggml_backend_vk_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ NULL,
    /* .get_alignment       = */ ggml_backend_vk_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_vk_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_vk_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ NULL,
    /* .is_host             = */ NULL,
};

static std::vector<uint32_t> ggml_vk_find_memory_properties(const vk::PhysicalDeviceMemoryProperties* mem_props, vk::MemoryRequirements* mem_req, vk::MemoryPropertyFlags flags) {
    std::vector<uint32_t> indices;

    for (uint32_t i = 0; i < mem_props->memoryTypeCount; ++i) {
        vk::MemoryType memory_type = mem_props->memoryTypes[i];
        if ((mem_req->memoryTypeBits & ((uint64_t)1 << i)) &&
            (flags & memory_type.propertyFlags) == flags &&
            mem_props->memoryHeaps[memory_type.heapIndex].size >= mem_req->size) {
            indices.push_back(i);
        }
    }
    return indices;
}

static vk_buffer ggml_vk_create_buffer(vk_device& device, size_t size, const std::initializer_list<vk::MemoryPropertyFlags> & req_flags_list,
                                       void *import_ptr = nullptr) {
    VK_LOG_DEBUG("ggml_vk_create_buffer(" << device->name << ", " << size << ", " << to_string(req_flags_list.begin()[0]) << ", " << to_string(req_flags_list.begin()[req_flags_list.size()-1]) << ")");
    if (size > device->max_buffer_size) {
        throw vk::OutOfDeviceMemoryError("Requested buffer size exceeds device buffer size limit");
    }

    vk_buffer buf = std::make_shared<vk_buffer_struct>();

    if (size == 0) {
        buf->size = 0;
        return buf;
    }

    vk::BufferUsageFlags usage_flags = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;
    vk::MemoryAllocateFlags mem_flags {};
    if (device->buffer_device_address) {
        usage_flags |= vk::BufferUsageFlagBits::eShaderDeviceAddress;
        mem_flags |= vk::MemoryAllocateFlagBits::eDeviceAddress;
    }

    vk::BufferCreateInfo buffer_create_info{
        vk::BufferCreateFlags(),
        size,
        usage_flags,
        vk::SharingMode::eExclusive,
        0,
        nullptr,
    };

    vk::ExternalMemoryBufferCreateInfo external_memory_bci;
    if (import_ptr) {
        external_memory_bci.handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT;
        buffer_create_info.setPNext(&external_memory_bci);
    }

    buf->buffer = device->device.createBuffer(buffer_create_info);

    vk::MemoryRequirements mem_req = device->device.getBufferMemoryRequirements(buf->buffer);

    vk::PhysicalDeviceMemoryProperties mem_props = device->physical_device.getMemoryProperties();

    const vk::MemoryPriorityAllocateInfoEXT mem_priority_info { 1.0f };

    vk::MemoryAllocateFlagsInfo mem_flags_info { mem_flags };

    if (device->memory_priority) {
        mem_flags_info.setPNext(&mem_priority_info);
    }

    if (import_ptr) {
        vk::MemoryHostPointerPropertiesEXT host_pointer_props;
        try {
            host_pointer_props = device->device.getMemoryHostPointerPropertiesEXT(vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT, import_ptr);
        } catch (vk::SystemError& e) {
            GGML_LOG_WARN("ggml_vulkan: Failed getMemoryHostPointerPropertiesEXT (%s)\n", e.what());
            device->device.destroyBuffer(buf->buffer);
            return {};
        }
        vk::PhysicalDeviceMemoryProperties mem_props = device->physical_device.getMemoryProperties();

        uint32_t memory_type_idx;
        vk::MemoryPropertyFlags property_flags = *req_flags_list.begin();
        for (memory_type_idx = 0; memory_type_idx < 32; ++memory_type_idx) {
            if (!(host_pointer_props.memoryTypeBits & (1u << memory_type_idx))) {
                continue;
            }
            if (!(mem_req.memoryTypeBits & (1u << memory_type_idx))) {
                continue;
            }

            vk::MemoryType memory_type = mem_props.memoryTypes[memory_type_idx];
            // check for visible+coherent+cached. Other flags (e.g. devicelocal) are allowed
            if ((memory_type.propertyFlags & property_flags) == property_flags) {
                property_flags = memory_type.propertyFlags;
                break;
            }
        }
        if (memory_type_idx == 32) {
            GGML_LOG_WARN("ggml_vulkan: Memory type for host allocation not found\n");
            device->device.destroyBuffer(buf->buffer);
            return {};
        }

        buf->memory_property_flags = mem_props.memoryTypes[memory_type_idx].propertyFlags;
        try {
            vk::ImportMemoryHostPointerInfoEXT import_info;
            import_info.handleType = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT;
            import_info.pHostPointer = import_ptr;
            import_info.setPNext(&mem_flags_info);
            buf->device_memory = device->device.allocateMemory({ size, memory_type_idx, &import_info });
        } catch (const vk::SystemError& e) {
        }
    } else {
        for (auto it = req_flags_list.begin(); it != req_flags_list.end(); it++) {
            const auto & req_flags = *it;

            const std::vector<uint32_t> memory_type_indices = ggml_vk_find_memory_properties(&mem_props, &mem_req, req_flags);

            if (memory_type_indices.empty()) {
                continue;
            }

            bool done = false;

            for (auto mtype_it = memory_type_indices.begin(); mtype_it != memory_type_indices.end(); mtype_it++) {
                try {
                    buf->device_memory = device->device.allocateMemory({ mem_req.size, *mtype_it, &mem_flags_info });
                    buf->memory_property_flags = mem_props.memoryTypes[*mtype_it].propertyFlags;
                    done = true;
                    break;
                } catch (const vk::SystemError& e) {
                    // loop and retry
                    // during last attempt throw the exception
                    if (it + 1 == req_flags_list.end() && mtype_it + 1 == memory_type_indices.end()) {
                        device->device.destroyBuffer(buf->buffer);
                        throw e;
                    }
                }
            }

            if (done) {
                break;
            }
        }
    }

    if (!buf->device_memory) {
        device->device.destroyBuffer(buf->buffer);
        throw vk::OutOfDeviceMemoryError("No suitable memory type found");
    }

    buf->ptr = nullptr;

    if (import_ptr) {
        buf->ptr = import_ptr;
    } else {
        if (buf->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
            buf->ptr = device->device.mapMemory(buf->device_memory, 0, VK_WHOLE_SIZE);
        }
    }

    device->device.bindBufferMemory(buf->buffer, buf->device_memory, 0);

    buf->device = device;
    buf->size = size;

    // driver-allocated host memory (an imported host pointer is the caller's own memory)
    if (!import_ptr && !(buf->memory_property_flags & vk::MemoryPropertyFlagBits::eDeviceLocal)) {
        buf->host_counted = mem_req.size;
        ggml_vk_host_alloc_bytes += (int64_t) mem_req.size;
    }

    if (device->buffer_device_address) {
        const vk::BufferDeviceAddressInfo addressInfo(buf->buffer);
        buf->bda_addr = device->device.getBufferAddress(addressInfo);
    }

    device->memory_logger->log_allocation(buf, size);

    return buf;
}

vk_buffer ggml_vk_create_buffer_check(vk_device& device, size_t size, vk::MemoryPropertyFlags req_flags, vk::MemoryPropertyFlags fallback_flags) {
    try {
        return ggml_vk_create_buffer(device, size, {req_flags, fallback_flags});
    } catch (const vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Memory allocation of size " << size << " failed." << std::endl;
        std::cerr << "ggml_vulkan: " << e.what() << std::endl;
        throw e;
    }
}

vk_buffer ggml_vk_create_buffer_device(vk_device& device, size_t size) {
    vk_buffer buf;
    try {
        if (device->prefer_host_memory) {
            buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                       vk::MemoryPropertyFlagBits::eDeviceLocal});
        } else if (device->uma) {
            // On UMA, prefer host-visible memory so direct tensor borrowing works.
            // If unavailable, fall back to device-local memory.
            buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                       vk::MemoryPropertyFlagBits::eDeviceLocal,
                                                       vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
        } else if (device->disable_host_visible_vidmem) {
            if (device->allow_sysmem_fallback) {
                buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal,
                                                           vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
            } else {
                buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal});
            }
        } else {
            // use rebar if available, otherwise fallback to device only visible memory
            if (device->allow_sysmem_fallback) {
                buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                           vk::MemoryPropertyFlagBits::eDeviceLocal,
                                                           vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
            } else {
                buf = ggml_vk_create_buffer(device, size, {vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                           vk::MemoryPropertyFlagBits::eDeviceLocal});
            }
        }
    } catch (const vk::SystemError& e) {
        std::cerr << "ggml_vulkan: Device memory allocation of size " << size << " failed." << std::endl;
        std::cerr << "ggml_vulkan: " << e.what() << std::endl;
        throw e;
    }

    return buf;
}

void ggml_vk_destroy_buffer(vk_buffer& buf) {
    if (buf == nullptr) {
        return;
    }

    if (buf->device != nullptr) {
        buf->device->memory_logger->log_deallocation(buf);
    }

    buf.reset();
}

// caller must hold host_allocations_mutex
static size_t ggml_vk_host_allocation_find(const void * ptr) {
    for (size_t i = 0; i < vk_instance.host_allocations.size(); i++) {
        const uint8_t * base = (const uint8_t *) vk_instance.host_allocations[i].base;
        if (ptr >= base && ptr < base + vk_instance.host_allocations[i].size) {
            return i;
        }
    }
    return SIZE_MAX;
}

void * ggml_vk_host_malloc(size_t size) {
    VK_LOG_MEMORY("ggml_vk_host_malloc(" << size << ")");

    ggml_vk_instance_init();

    if (vk_instance.multi_device) {
        const size_t align      = std::max<size_t>(vk_instance.host_import_alignment, TENSOR_ALIGNMENT);
        const size_t alloc_size = (size + align - 1) / align * align;

        void * ptr = portable_aligned_malloc(align, alloc_size);
        if (ptr == nullptr) {
            return nullptr;
        }
        vk_host_allocation entry;
        entry.base = ptr;
        entry.size = alloc_size;

        std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
        vk_instance.host_allocations.push_back(std::move(entry));
        return ptr;
    }

    vk_device dev0 = ggml_vk_get_device(0);
    vk_buffer buf;
    try {
        buf = ggml_vk_create_buffer(dev0, size,
            {vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached,
             vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent});
    } catch (vk::SystemError&) {
        return nullptr;
    }

    if (!(buf->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible)) {
        fprintf(stderr, "WARNING: failed to allocate %.2f MB of pinned memory\n", size / 1024.0 / 1024.0);
        ggml_vk_destroy_buffer(buf);
        return nullptr;
    }

    vk_host_allocation entry;
    entry.base             = buf->ptr;
    entry.size             = size;
    entry.views[dev0->idx] = buf;

    std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
    vk_instance.host_allocations.push_back(std::move(entry));
    return buf->ptr;
}

void ggml_vk_host_free(void * ptr) {
    if (ptr == nullptr) {
        return;
    }
    VK_LOG_MEMORY("ggml_vk_host_free(" << ptr << ")");

    vk_host_allocation entry;
    {
        std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);
        const size_t index = ggml_vk_host_allocation_find(ptr);
        if (index == SIZE_MAX) {
            fprintf(stderr, "WARNING: failed to free pinned memory: memory not in map\n");
            return;
        }
        entry = std::move(vk_instance.host_allocations[index]);
        vk_instance.host_allocations.erase(vk_instance.host_allocations.begin() + index);
    }

    for (auto & view : entry.views) {
        ggml_vk_destroy_buffer(view);
    }
    if (vk_instance.multi_device) {
        portable_aligned_free(entry.base);
    }
}

void ggml_vk_host_get(const vk_device& device, const void * ptr, vk_buffer& buf, size_t& buf_offset) {
    buf = nullptr;
    buf_offset = 0;

    std::unique_lock<std::shared_mutex> guard(vk_instance.host_allocations_mutex);

    const size_t index = ggml_vk_host_allocation_find(ptr);
    if (index == SIZE_MAX) {
        return;
    }
    vk_host_allocation & entry = vk_instance.host_allocations[index];

    if (entry.views[device->idx] == nullptr) {
        const uint32_t bit = 1u << device->idx;
        if (!vk_instance.multi_device || !vk_instance.device_host_buffer_import[device->idx] || (vk_instance.host_import_failed & bit)) {
            return;
        }
        GGML_ASSERT(device->external_memory_host);
        // const only propagates from the const context; the pointee is mutable
        entry.views[device->idx] = ggml_vk_buffer_from_host_ptr(const_cast<vk_device&>(device), entry.base, entry.size);
        if (entry.views[device->idx] == nullptr) {
            vk_instance.host_import_failed |= bit;
            GGML_LOG_WARN("ggml_vulkan: %s: host memory import failed, using staging\n", device->name.c_str());
            return;
        }
    }

    buf = entry.views[device->idx];
    buf_offset = (const uint8_t *) ptr - (const uint8_t *) entry.base;
}

void ggml_vk_ensure_sync_staging_buffer(vk_device& device, size_t size) {
    if (device->sync_staging == nullptr || device->sync_staging->size < size) {
        VK_LOG_MEMORY("ggml_vk_ensure_sync_staging_buffer(" << size << ")");
        ggml_vk_destroy_buffer(device->sync_staging);
        device->sync_staging = ggml_vk_create_buffer_check(device, size,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    }
}

void ggml_vk_ensure_sync_staging_buffer(ggml_backend_vk_context * ctx, size_t size) {
    if (ctx->sync_staging == nullptr || ctx->sync_staging->size < size) {
        VK_LOG_MEMORY("ggml_vk_ensure_sync_staging_buffer(" << size << ")");
        ggml_vk_destroy_buffer(ctx->sync_staging);
        ctx->sync_staging = ggml_vk_create_buffer_check(ctx->device, size,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    }
}

static void ggml_vk_buffer_write_nc_async(ggml_backend_vk_context * ctx, vk_context& subctx, vk_buffer& dst, size_t offset, const ggml_tensor * tensor, bool sync_staging = false) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_nc_async(" << tensor << ")");
    GGML_ASSERT(!ggml_is_contiguous(tensor));
    // Buffer is already mapped
    if(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
        std::cerr << "ggml_vulkan: buffer_write_nc_async dst buffer is host_visible. Use synchronous write." << std::endl;
        GGML_ABORT("fatal error");
    }
    // Check if src is pinned memory
    vk_buffer buf = nullptr;
    size_t buf_offset = 0;
    ggml_vk_host_get(ctx->device, tensor->data, buf, buf_offset);

    const uint64_t ne0 = tensor->ne[0];
    const uint64_t ne1 = tensor->ne[1];
    const uint64_t ne2 = tensor->ne[2];
    const uint64_t ne3 = tensor->ne[3];
    const uint64_t nb0 = tensor->nb[0];
    const uint64_t nb1 = tensor->nb[1];
    const uint64_t nb2 = tensor->nb[2];
    const uint64_t nb3 = tensor->nb[3];
    const ggml_type type = tensor->type;
    const uint64_t ts = ggml_type_size(type);
    const uint64_t bs = ggml_blck_size(type);

    const uint64_t dstnb0 = ts;
    const uint64_t dstnb1 = dstnb0*(ne0/bs);
    const uint64_t dstnb2 = dstnb1*ne1;
    const uint64_t dstnb3 = dstnb2*ne2;

    const uint64_t ne = ggml_nelements(tensor);

    if (buf != nullptr) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::BufferCopy> slices;

        for (uint64_t i3 = 0; i3 < ne3; i3++) {
            for (uint64_t i2 = 0; i2 < ne2; i2++) {
                // Find longest contiguous slice
                if (ne1*nb1 == dstnb2) {
                    slices.push_back({ buf_offset + i3*nb3 + i2*nb2, offset + i3*dstnb3 + i2*dstnb2, dstnb2 });
                } else {
                    for (uint64_t i1 = 0; i1 < ne1; i1++) {
                        if (ne0*nb0/bs == dstnb1) {
                            slices.push_back({ buf_offset + i3*nb3 + i2*nb2 + i1*nb1, offset + i3*dstnb3 + i2*dstnb2 + i1*dstnb1, dstnb1 });
                        } else {
                            const uint64_t s_off = buf_offset + i3*nb3 + i2*nb2 + i1*nb1;
                            const uint64_t d_off = offset + i3*dstnb3 + i2*dstnb2 + i1*dstnb1;
                            for (uint64_t i0 = 0; i0 < ne0; i0++) {
                                slices.push_back({ s_off + i0*nb0, d_off + i0*dstnb0, dstnb0 });
                            }
                        }
                    }
                }
            }
        }

        ggml_vk_sync_buffers(ctx, subctx);
        subctx->s->buffer->buf.copyBuffer(buf->buffer, dst->buffer, slices);
        return;
    }

    if (!sync_staging) {
        GGML_ABORT("Asynchronous write to non-pinned memory not supported");
    }

    // Staging buffer required
    vk_buffer& staging = ctx->device->sync_staging;
    const uint64_t copy_size = ts*ne/bs;
    ggml_vk_ensure_sync_staging_buffer(ctx->device, copy_size);
    VkBufferCopy buf_copy{ 0, offset, copy_size };

    ggml_vk_sync_buffers(ctx, subctx);
    vkCmdCopyBuffer(subctx->s->buffer->buf, (VkBuffer)staging->buffer, (VkBuffer)dst->buffer, 1, &buf_copy);

    for (uint64_t i3 = 0; i3 < ne3; i3++) {
        for (uint64_t i2 = 0; i2 < ne2; i2++) {
            // Find longest contiguous slice
            if (ne1*nb1 == dstnb2) {
                deferred_memcpy((uint8_t *)staging->ptr + i3*dstnb3 + i2*dstnb2, (const uint8_t *) tensor->data + buf_offset + i3*nb3 + i2*nb2, dstnb2, &subctx->in_memcpys);
            } else {
                for (uint64_t i1 = 0; i1 < ne1; i1++) {
                    if (ne0*nb0/bs == dstnb1) {
                        deferred_memcpy((uint8_t *)staging->ptr + i3*dstnb3 + i2*dstnb2 + i1*dstnb1, (const uint8_t *) tensor->data + buf_offset + i3*nb3 + i2*nb2 + i1*nb1, dstnb1, &subctx->in_memcpys);
                    } else {
                        const uint64_t s_off = buf_offset + i3*nb3 + i2*nb2 + i1*nb1;
                        const uint64_t d_off = i3*dstnb3 + i2*dstnb2 + i1*dstnb1;
                        for (uint64_t i0 = 0; i0 < ne0; i0++) {
                            deferred_memcpy((uint8_t *)staging->ptr + d_off + i0*dstnb0, (const uint8_t *) tensor->data + s_off + i0*nb0, dstnb0, &subctx->in_memcpys);
                        }
                    }
                }
            }
        }
    }
}

bool ggml_vk_buffer_write_2d_async(vk_context subctx, vk_buffer& dst, size_t offset, const void * src, size_t spitch, size_t dpitch, size_t width, size_t height, bool sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d_async(" << width << ", " << height << ")");
    // Check if src is pinned memory
    vk_buffer buf = nullptr;
    size_t buf_offset = 0;
    ggml_vk_host_get(dst->device, src, buf, buf_offset);

    if (buf != nullptr) {
        // Memory is pinned, use as staging buffer
        std::vector<vk::BufferCopy> slices(1);
        if (width == spitch && width == dpitch) {
            // Only do single write if stride is equal
            slices[0].srcOffset = buf_offset;
            slices[0].dstOffset = offset;
            slices[0].size = width * height;
        } else {
            slices.resize(height);
            for (size_t i = 0; i < height; i++) {
                slices[i].srcOffset = buf_offset + i * spitch;
                slices[i].dstOffset = offset + i * dpitch;
                slices[i].size = width;
            }
        }

        ggml_vk_sync_buffers(nullptr, subctx);
        subctx->s->buffer->buf.copyBuffer(buf->buffer, dst->buffer, slices);
        return true;
    }
    VK_LOG_DEBUG("STAGING");

    if (!sync_staging) {
        // copy was not handled caller needs to fall back
        return false;
    }

    // Staging buffer required
    const size_t staging_size = width * height;
    ggml_vk_ensure_sync_staging_buffer(dst->device, staging_size);

    vk_buffer& staging_buffer = dst->device->sync_staging;

    std::vector<vk::BufferCopy> slices(1);
    if (width == dpitch) {
        slices[0].srcOffset = 0;
        slices[0].dstOffset = offset;
        slices[0].size = staging_size;
    } else {
        slices.resize(height);
        for (size_t i = 0; i < height; i++) {
            slices[i].srcOffset = i * width;
            slices[i].dstOffset = offset + i * dpitch;
            slices[i].size = width;
        }
    }

    ggml_vk_sync_buffers(nullptr, subctx);
    subctx->s->buffer->buf.copyBuffer(staging_buffer->buffer, dst->buffer, slices);

    if (width == spitch) {
        deferred_memcpy((uint8_t *)staging_buffer->ptr, src, staging_size, &subctx->in_memcpys);
    } else {
        for (size_t i = 0; i < height; i++) {
            deferred_memcpy((uint8_t *)staging_buffer->ptr + i * width, (const uint8_t *) src + i * spitch, width, &subctx->in_memcpys);
        }
    }
    return true;
}

bool ggml_vk_buffer_write_async(vk_context subctx, vk_buffer& dst, size_t offset, const void * src, size_t size, bool sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_async(" << size << ")");
    return ggml_vk_buffer_write_2d_async(subctx, dst, offset, src, size, size, size, 1, sync_staging);
}

void ggml_vk_buffer_write_2d(vk_buffer& dst, size_t offset, const void * src, size_t spitch, size_t dpitch, size_t width, size_t height) {
    VK_LOG_DEBUG("ggml_vk_buffer_write_2d(" << width << ", " << height << ")");
    ggml_vk_timing_scope timing(GGML_VK_T_WRITE);
    // Buffer is already mapped
    if(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
        GGML_ASSERT(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent);
        dst->device->submit_epoch++; // dual-buffer fork: a host write invalidates the read-ahead (vk_readahead)

        if (width == spitch && width == dpitch) {
            memcpy((uint8_t *)dst->ptr + offset, src, width * height);
        } else {
            for (size_t i = 0; i < height; i++) {
                memcpy((uint8_t *)dst->ptr + offset + i * dpitch, (const uint8_t *) src + i * spitch, width);
            }
        }
    } else {
        std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);

        // dual-buffer fork: a small write on the compute queue with a spinning wait (ggml_vk_sync_copy_wait)
        vk_context subctx = ggml_vk_create_temporary_context(ggml_vk_sync_copy_queue(dst->device, width * height).cmd_pool);
        ggml_vk_ctx_begin(dst->device, subctx);
        bool ret = ggml_vk_buffer_write_2d_async(subctx, dst, offset, src, spitch, dpitch, width, height, true);
        GGML_ASSERT(ret);
        ggml_vk_ctx_end(subctx);

        for (auto& cpy : subctx->in_memcpys) {
            memcpy(cpy.dst, cpy.src, cpy.n);
        }

        for (auto& mset : subctx->memsets) {
            memset(mset.dst, mset.val, mset.n);
        }

        ggml_vk_submit(subctx, dst->device->fence);
        ggml_vk_sync_copy_wait(dst->device, width * height, "vk_buffer_write_2d waitForFences");
        ggml_vk_queue_command_pools_cleanup(dst->device);
    }
}

void ggml_vk_buffer_write(vk_buffer& dst, size_t offset, const void * src, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_write(" << size << ")");
    ggml_vk_buffer_write_2d(dst, offset, src, size, size, size, 1);
}

bool ggml_vk_buffer_read_2d_async(vk_context subctx, vk_buffer& src, size_t offset, void * dst, size_t spitch, size_t dpitch, size_t width, size_t height, bool sync_staging) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_2d_async(offset=" << offset << ", width=" << width << ", height=" << height << ")");
    GGML_ASSERT(width > 0);
    GGML_ASSERT(height > 0);
    GGML_ASSERT(src != nullptr);

    // TODO: staging_offset is not used

    // Check if dst is pinned memory
    vk_buffer buf = nullptr;
    size_t buf_offset = 0;
    ggml_vk_host_get(src->device, dst, buf, buf_offset);

    std::vector<vk::BufferCopy> slices(1);
    if (width == spitch && width == dpitch) {
        // Only do single write if stride is equal
        slices[0].srcOffset = offset;
        slices[0].dstOffset = buf_offset;
        slices[0].size = width * height;
    } else {
        slices.resize(height);
        for (size_t i = 0; i < height; i++) {
            slices[i].srcOffset = offset + i * spitch;
            slices[i].dstOffset = buf_offset + i * dpitch;
            slices[i].size = width;
        }
    }

    if (buf != nullptr) {
        // Memory is pinned, use as staging buffer
        ggml_vk_sync_buffers(nullptr, subctx);
        subctx->s->buffer->buf.copyBuffer(src->buffer, buf->buffer, slices);

        return true;
    }
    VK_LOG_DEBUG("STAGING");

    if (!sync_staging) {
        // copy was not handled caller needs to fall back
        return false;
    }

    // Fall back to staging buffer
    const size_t staging_size = width * height;
    ggml_vk_ensure_sync_staging_buffer(src->device, staging_size);

    vk_buffer& staging_buffer = src->device->sync_staging;

    std::vector<vk::BufferCopy> staging_slices(1);
    if (width == spitch) {
        staging_slices[0].srcOffset = offset;
        staging_slices[0].dstOffset = 0;
        staging_slices[0].size = staging_size;
    } else {
        staging_slices.resize(height);
        for (size_t i = 0; i < height; i++) {
            staging_slices[i].srcOffset = offset + i * spitch;
            staging_slices[i].dstOffset = i * width;
            staging_slices[i].size = width;
        }
    }

    ggml_vk_sync_buffers(nullptr, subctx);
    subctx->s->buffer->buf.copyBuffer(src->buffer, staging_buffer->buffer, staging_slices);

    if (width == dpitch) {
        deferred_memcpy(dst, staging_buffer->ptr, staging_size, &subctx->out_memcpys);
    } else {
        for (size_t i = 0; i < height; i++) {
            deferred_memcpy((uint8_t *) dst + i * dpitch, (const uint8_t *) staging_buffer->ptr + i * width, width, &subctx->out_memcpys);
        }
    }
    return true;
}

static bool ggml_vk_buffer_read_async(vk_context subctx, vk_buffer& src, size_t offset, void * dst, size_t size, bool sync_staging = false) {
    return ggml_vk_buffer_read_2d_async(subctx, src, offset, dst, size, size, size, 1, sync_staging);
}

void ggml_vk_buffer_read_2d(vk_buffer& src, size_t offset, void * dst, size_t spitch, size_t dpitch, size_t width, size_t height) {
    VK_LOG_DEBUG("ggml_vk_buffer_read_2d(" << src->buffer << ", " << offset << ", " << width << ", " << height << ")");
    ggml_vk_timing_scope timing(GGML_VK_T_READ);

    // dual-buffer fork: a range copied by the read-ahead of the last synchronize (vk_readahead)
    if (ggml_vk_readahead_read(src, offset, dst, spitch, dpitch, width, height)) {
        return;
    }

    // If the device is not an UMA device the memory is host-accessible through rebar. While writing
    // through PCIe is sufficient fast reading back data from PCIe is slower than going through
    // the HW device to host copy path.
    // AMD UMA: uncached host-visible memory is write-combined, CPU reads are slow
    const bool slow_host_read = src->device->vendor_id == VK_VENDOR_ID_AMD &&
                                !(src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCached);
    if(src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible && src->device->uma && !slow_host_read) {
        GGML_ASSERT(src->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent);

        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);
        vk_context subctx = ggml_vk_create_temporary_context(src->device->compute_queue->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        subctx->s->buffer->buf.pipelineBarrier(
            vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eTransfer,
            vk::PipelineStageFlagBits::eHost,
            {},
            { { vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite,
                vk::AccessFlagBits::eHostRead } },
            {}, {});
        ggml_vk_ctx_end(subctx);
        ggml_vk_submit(subctx, src->device->fence);
        ggml_vk_sync_copy_wait(src->device, width * height, "vk_buffer_read_2d uma waitForFences");
        ggml_vk_queue_command_pools_cleanup(src->device);

        if (width == spitch && width == dpitch) {
            memcpy(dst, (const uint8_t *) src->ptr + offset, width * height);
        } else {
            for (size_t i = 0; i < height; i++) {
                memcpy((uint8_t *) dst + i * dpitch, (const uint8_t *) src->ptr + offset + i * spitch, width);
            }
        }
    } else {
        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);

        // dual-buffer fork: a small read (a split-boundary input) on the compute queue with a spinning wait
        // (ggml_vk_sync_copy_wait)
        vk_context subctx = ggml_vk_create_temporary_context(ggml_vk_sync_copy_queue(src->device, width * height).cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        bool ret = ggml_vk_buffer_read_2d_async(subctx, src, offset, dst, spitch, dpitch, width, height, true);
        GGML_ASSERT(ret);
        ggml_vk_ctx_end(subctx);

        ggml_vk_submit(subctx, src->device->fence);
        ggml_vk_sync_copy_wait(src->device, width * height, "vk_buffer_read_2d waitForFences");
        ggml_vk_queue_command_pools_cleanup(src->device);

        for (auto& cpy : subctx->out_memcpys) {
            memcpy(cpy.dst, cpy.src, cpy.n);
        }
    }
}

// dual-buffer fork: vk_readahead. Every synchronous read after the synchronize of a small graph is observed (learned for
// that graph by the next graph_compute); a read inside a range of the read-ahead is served from it while no submission
// and no host write to the device's memory has happened since the read-ahead's own submission.
bool ggml_vk_readahead_read(vk_buffer & src, size_t offset, void * dst, size_t spitch, size_t dpitch, size_t width, size_t height) {
    if (!ggml_vk_low_latency_sync()) {
        return false;
    }
    const size_t len = spitch * (height - 1) + width;
    vk_readahead & ra = src->device->readahead;
    std::lock_guard<std::mutex> lock(ra.mutex);
    if (ra.observe_key != 0 && ggml_vk_sync_copy_small(len) && ra.observed.size() < 16) {
        ra.observed.push_back({ src, src->buffer, offset, len, 0 });
    }
    if (ra.ready_epoch != src->device->submit_epoch.load()) {
        return false;
    }
    for (const auto & r : ra.ready) {
        if (r.handle == src->buffer && offset >= r.offset && offset + len <= r.offset + r.size && r.buf.lock() == src) {
            ggml_vk_timing_scope timing(GGML_VK_T_RA_HIT);
            const uint8_t * s = (const uint8_t *) ra.staging->ptr + r.staging_offset + (offset - r.offset);
            if (width == spitch && width == dpitch) {
                memcpy(dst, s, width * height);
            } else {
                for (size_t i = 0; i < height; i++) {
                    memcpy((uint8_t *) dst + i * dpitch, s + i * spitch, width);
                }
            }
            return true;
        }
    }
    return false;
}

void ggml_vk_buffer_read(vk_buffer& src, size_t offset, void * dst, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_read(" << src->buffer << ", " << offset << ", " << size << ")");
    ggml_vk_buffer_read_2d(src, offset, dst, size, size, size, 1);
}

void ggml_vk_buffer_copy_async(vk_context& ctx, vk_buffer& dst, size_t dst_offset, vk_buffer& src, size_t src_offset, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_copy_async(" << size << ")");
    // Make sure both buffers are on same device
    GGML_ASSERT(src->device == dst->device);

    VkBufferCopy bc{ src_offset, dst_offset, size };

    vkCmdCopyBuffer(ctx->s->buffer->buf, (VkBuffer)src->buffer, (VkBuffer)dst->buffer, 1, &bc);
}

void ggml_vk_buffer_copy(vk_buffer& dst, size_t dst_offset, vk_buffer& src, size_t src_offset, size_t size) {
    if (src->device == dst->device) {
        std::lock_guard<std::recursive_mutex> guard(src->device->mutex);
        VK_LOG_DEBUG("ggml_vk_buffer_copy(SINGLE_DEVICE, " << size << ")");
        // Copy within the device
        vk_context subctx = ggml_vk_create_temporary_context(src->device->transfer_queue->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        ggml_vk_buffer_copy_async(subctx, dst, dst_offset, src, src_offset, size);
        ggml_vk_ctx_end(subctx);
        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, true, UINT64_MAX), "vk_buffer_copy waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_queue_command_pools_cleanup(src->device);
    } else {
        VK_LOG_DEBUG("ggml_vk_buffer_copy(MULTI_DEVICE, " << size << ")");
        // Copy device to device
        ggml_vk_ensure_sync_staging_buffer(src->device, size);

        // Copy to src staging buffer
        ggml_vk_buffer_copy(src->device->sync_staging, 0, src, src_offset, size);
        // Copy to dst buffer
        ggml_vk_buffer_write(dst, dst_offset, src->device->sync_staging->ptr, size);
    }
}

void ggml_vk_buffer_memset_async(vk_context& ctx, vk_buffer& dst, size_t offset, uint32_t c, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_memset_async(" << offset << ", " << c << ", " << size << ")");

    if (dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible &&
        dst->device->uma) {
        deferred_memset((uint8_t*)dst->ptr + offset, c, size, &ctx->memsets);
        return;
    }

    // Fall back to GPU fillBuffer for non-UMA or non-host-visible buffers
    ctx->s->buffer->buf.fillBuffer(dst->buffer, offset, size, c);
}

void ggml_vk_buffer_memset(vk_buffer& dst, size_t offset, uint32_t c, size_t size) {
    VK_LOG_DEBUG("ggml_vk_buffer_memset(" << offset << ", " << c << ", " << size << ")");

    if (dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible &&
        dst->device->uma) {
        dst->device->submit_epoch++; // dual-buffer fork: a host write invalidates the read-ahead (vk_readahead)
        memset((uint8_t*)dst->ptr + offset, c, size);
        return;
    }

    std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);
    vk_context subctx = ggml_vk_create_temporary_context(dst->device->transfer_queue->cmd_pool);
    ggml_vk_ctx_begin(dst->device, subctx);
    subctx->s->buffer->buf.fillBuffer(dst->buffer, offset, size, c);
    ggml_vk_ctx_end(subctx);

    ggml_vk_submit(subctx, dst->device->fence);
    VK_CHECK(dst->device->device.waitForFences({ dst->device->fence }, true, UINT64_MAX), "vk_memset waitForFences", dst->device);
    dst->device->device.resetFences({ dst->device->fence });
    ggml_vk_queue_command_pools_cleanup(dst->device);
}

ggml_backend_buffer_i ggml_backend_vk_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_vk_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_vk_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_vk_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_vk_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_vk_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_vk_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_vk_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_vk_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_vk_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_vk_buffer_clear,
    /* .reset           = */ NULL,
    /* .set_tensor_n    = */ NULL,
    /* .get_tensor_n    = */ NULL,
    /* .cpy_tensor_n    = */ NULL,
};

vk_buffer ggml_vk_buffer_from_host_ptr(vk_device & device, void * ptr, size_t size) {
    if (!device->external_memory_host) {
        return {};
    }

    uintptr_t uptr = reinterpret_cast<uintptr_t>(ptr);
    if (uptr & (device->min_imported_host_pointer_alignment - 1)) {
        return {};
    }
    if (size & (device->min_imported_host_pointer_alignment - 1)) {
        return {};
    }

    const vk::MemoryPropertyFlags property_flags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached;

    vk_buffer buf {};
    try {
        buf = ggml_vk_create_buffer(device, size, { property_flags }, ptr);
    } catch (vk::SystemError& e) {
        GGML_LOG_WARN("ggml_vulkan: Failed ggml_vk_create_buffer (%s)\n", e.what());
    }

    return buf;
}


// Chunked K/V buffers (dual-buffer fork, dynamic KV): the interface of the ggml-cuda VMM arenas (looked up with
// get_proc_address) on ordinary device buffers. A section (one K or V) is a list of chunks, each an ordinary device buffer
// of a whole number of K/V rows (a multiple of 256 rows): it grows by new chunks at the end, the chunks it has stay where
// they are (no copy, no second allocation), and it shrinks by freeing its top chunks (vkFreeMemory: the memory comes back).
// The shaders reach the chunks by their device addresses: a shader that reads or writes the K/V has a KV_CHUNKED build
// that gets the section's chunk table in place of the tensor's binding (a header and the chunk addresses, one table per
// binding offset) and computes every address with kvc_addr() (kv_chunk.glsl); the host reads and writes chunk by chunk.
// The chunk follows the rule of the ggml-cuda VMM (granularity, 2 MiB alignment, the measured booking unit) plus the
// Vulkan bounds (the release unit, the allocation count limit) and is rounded to the section's rows when its tensor is
// placed. Needs bufferDeviceAddress and shaderInt64; without them there is no arena and the KV grows by copy (a log line
// names the missing capability). GGML_VK_KV_CHUNK_KIB sets the device chunk, GGML_VK_KV_CHUNK_DEBUG=1 logs every change
// (test aids).

struct vk_kvc_section;

struct vk_kvc_arena {
    vk_device                       device;
    ggml_backend_buffer_type_t      buft;
    size_t                          chunk = 0; // the device chunk: a section's chunk is the smallest whole number of its rows above it
    size_t                          unit  = 0; // the allocation unit a section's chunk should fill without waste
    std::vector<vk_kvc_section *>   pending;   // sections with new chunks, not zeroed yet
    int                             refs = 1;
    std::mutex                      mtx;
};

struct vk_kvc_table {
    size_t    base; // the section offset the binding starts at (the descriptor offset of an ordinary buffer)
    vk_buffer buf;  // uvec4 header + one device address per chunk
};

struct vk_kvc_section : ggml_backend_vk_buffer_context {
    vk_kvc_arena *            arena;
    size_t                    section;      // max bytes
    size_t                    chunk;        // bytes per chunk
    size_t                    row = 0;      // bytes per K/V row (once the tensor is placed)
    size_t                    logical = 0;  // the ggml buffer size
    std::vector<vk_buffer>    chunks;       // the chunks from the start
    size_t                    n_pending = 0; // the last n_pending chunks are not zeroed yet
    std::vector<vk_kvc_table> tables;
    std::recursive_mutex      mtx;

    vk_kvc_section(vk_device_ref dev, std::string & name, vk_kvc_arena * arena, size_t section, size_t chunk) :
        ggml_backend_vk_buffer_context(dev, vk_buffer(), name), arena(arena), section(section), chunk(chunk) {}

    size_t mapped() const { return chunks.size() * chunk; }
    size_t n_entries() const { return (section + chunk - 1) / chunk; }
};

static bool vk_kvc_debug() {
    static const bool on = getenv("GGML_VK_KV_CHUNK_DEBUG") && atoi(getenv("GGML_VK_KV_CHUNK_DEBUG")) != 0;
    return on;
}

static std::atomic<int64_t> vk_kvc_stat_bytes{0};

static std::mutex                 vk_kvc_reg_mtx;
static std::set<vk_kvc_section *> vk_kvc_live;

static void vk_kvc_arena_unref(vk_kvc_arena * a) {
    int left;
    {
        std::lock_guard<std::mutex> lock(a->mtx);
        left = --a->refs;
    }
    if (left == 0) {
        delete a;
    }
}

// the heap usage of a memory type by the memory budget (VK_EXT_memory_budget); false without it
static bool vk_heap_usage(vk_device & device, uint32_t mem_type, size_t & usage) {
    if (device->idx >= vk_instance.device_supports_membudget.size() || !vk_instance.device_supports_membudget[device->idx]) {
        return false;
    }
    vk::PhysicalDeviceMemoryBudgetPropertiesEXT budget;
    vk::PhysicalDeviceMemoryProperties2 mp;
    mp.pNext = &budget;
    device->physical_device.getMemoryProperties2(&mp);
    const uint32_t heap = mp.memoryProperties.memoryTypes[mem_type].heapIndex;
    usage = (size_t) budget.heapUsage[heap];
    return true;
}

// the unit the driver books one allocation of `size` bytes with: the heap usage change of one allocation, the same twice and a
// multiple of the size; 0 if it cannot be measured (note says why). Once per device.
static size_t vk_kvc_booking_unit(vk_device & device, uint32_t mem_type, size_t size, std::string & note) {
    static std::mutex mtx;
    static std::map<size_t, std::pair<size_t, std::string>> done;
    std::lock_guard<std::mutex> lock(mtx);
    auto it = done.find(device->idx);
    if (it != done.end()) {
        note = it->second.second;
        return it->second.first;
    }
    size_t unit = 0;
    size_t d[2] = { 0, 1 };
    bool measured = true;
    for (int i = 0; i < 2 && measured; ++i) {
        size_t u0 = 0, u1 = 0;
        if (!vk_heap_usage(device, mem_type, u0)) {
            measured = false;
            break;
        }
        vk::DeviceMemory m;
        try {
            m = device->device.allocateMemory(vk::MemoryAllocateInfo(size, mem_type));
        } catch (const vk::SystemError &) {
            measured = false;
            break;
        }
        vk_heap_usage(device, mem_type, u1);
        device->device.freeMemory(m);
        d[i] = u1 > u0 ? u1 - u0 : 0;
    }
    if (!measured) {
        note = " (unclear: no VK_EXT_memory_budget to measure it)";
    } else if (d[0] > 0 && d[0] == d[1] && d[0] % size == 0) {
        unit = d[0];
        note = " (measured)";
    } else {
        note = " (unclear: the heap usage changed by " + std::to_string(d[0]) + " / " + std::to_string(d[1]) + " bytes)";
    }
    done[device->idx] = { unit, note };
    return unit;
}

// the smallest allocation size (from `base`, doubled, a multiple of it) whose memory the driver gives back when freed: a few
// allocations of it, then freed, must bring the heap usage back to where it was (a driver can serve small allocations from
// larger blocks it keeps); 0 if it cannot be measured (no memory budget) or no size up to the limit does. Once per device.
static size_t vk_kvc_release_unit(vk_device & device, uint32_t mem_type, size_t base, std::string & note) {
    static std::mutex mtx;
    static std::map<size_t, std::pair<size_t, std::string>> done;
    std::lock_guard<std::mutex> lock(mtx);
    auto it = done.find(device->idx);
    if (it != done.end()) {
        note = it->second.second;
        return it->second.first;
    }
    const size_t limit = (size_t) 64 << 20;
    const int    n     = 8;
    size_t res = 0;
    note = " (unclear: no VK_EXT_memory_budget to measure it)";
    size_t u_probe = 0;
    if (vk_heap_usage(device, mem_type, u_probe)) {
        note = " (unclear: no size up to " + std::to_string(limit >> 20) + " MiB was given back in full)";
        for (size_t s = base; s <= limit && res == 0; s *= 2) {
            size_t u0 = 0, u1 = 0, u2 = 0;
            vk_heap_usage(device, mem_type, u0);
            std::vector<vk::DeviceMemory> ms;
            try {
                for (int i = 0; i < n; ++i) {
                    ms.push_back(device->device.allocateMemory(vk::MemoryAllocateInfo(s, mem_type)));
                }
            } catch (const vk::SystemError &) {
                for (auto & m : ms) {
                    device->device.freeMemory(m);
                }
                break;
            }
            vk_heap_usage(device, mem_type, u1);
            for (auto & m : ms) {
                device->device.freeMemory(m);
            }
            vk_heap_usage(device, mem_type, u2);
            if (u1 >= u0 + n*s && u2 <= u0) {
                res  = s;
                note = " (measured: " + std::to_string(n) + " allocations of it freed bring the heap usage back)";
            }
        }
    }
    done[device->idx] = { res, note };
    return res;
}

// a section's chunk for rows of `row` bytes: the smallest whole number of 256 rows at or above the device chunk whose
// size the allocation unit holds with little waste (at most 1/16), else the smallest; 0 if it does not fit 31 bits
static size_t vk_kvc_chunk_for_row(const vk_kvc_arena * a, size_t row) {
    const size_t step = 256 * std::max<size_t>(row, 1);
    const size_t k0   = std::max<size_t>(1, (a->chunk + step - 1) / step);
    const size_t unit = std::max<size_t>(a->unit, 1);
    size_t res = k0 * step;
    for (size_t k = k0; k <= 64 * k0; ++k) {
        const size_t s = k * step;
        const size_t waste = unit * ((s + unit - 1) / unit) - s;
        if (waste * 16 <= s) {
            res = s;
            break;
        }
    }
    return res < ((size_t) 1 << 31) ? res : 0;
}

static vk_buffer vk_kvc_alloc_chunk(vk_kvc_arena * a, size_t bytes) {
    try {
        return ggml_vk_create_buffer(a->device, bytes, { vk::MemoryPropertyFlagBits::eDeviceLocal });
    } catch (const vk::SystemError & e) {
        if (vk_kvc_debug()) {
            GGML_LOG_INFO("vk_kvc_alloc_chunk: a chunk of %zu bytes: %s\n", bytes, e.what());
        }
        return nullptr;
    }
}

// the fastdiv values of d (kv_chunk.glsl): q = (mulhi(n, mp) + n) >> L
static void vk_kvc_fastdiv(uint32_t d, uint32_t & mp, uint32_t & L) {
    L = 0;
    while (L < 32 && ((uint64_t) 1 << L) < d) {
        L++;
    }
    mp = (uint32_t) ((((uint64_t) 1 << 32) * (((uint64_t) 1 << L) - d)) / d + 1);
}

// write a table: the header and the address of every chunk (the ones not there point at chunk 0, never read)
static void vk_kvc_table_fill(vk_kvc_section * s, vk_kvc_table & t) {
    const size_t n = s->n_entries();
    std::vector<uint64_t> data(2 + n);
    uint32_t mp = 0, L = 0;
    vk_kvc_fastdiv((uint32_t) s->chunk, mp, L);
    const uint32_t h[4] = { (uint32_t) t.base, (uint32_t) s->chunk, mp, L };
    memcpy(data.data(), h, sizeof(h));
    for (size_t i = 0; i < n; ++i) {
        data[2 + i] = i < s->chunks.size() ? s->chunks[i]->bda_addr : s->chunks[0]->bda_addr;
    }
    if (t.buf->ptr) {
        memcpy(t.buf->ptr, data.data(), data.size() * sizeof(uint64_t));
    } else {
        ggml_vk_buffer_write(t.buf, 0, data.data(), data.size() * sizeof(uint64_t));
    }
}

static void vk_kvc_tables_fill(vk_kvc_section * s) {
    for (auto & t : s->tables) {
        vk_kvc_table_fill(s, t);
    }
}

static void * ggml_backend_vk_vmm_arena_new(ggml_backend_buffer_type_t buft, size_t size) {
    if (buft->iface.get_name != ggml_backend_vk_buffer_type_name) {
        return nullptr;
    }
    vk_device & device = ((ggml_backend_vk_buffer_type_context *) buft->context)->device;
    static std::mutex log_mtx;
    static std::set<std::string> logged;
    auto no = [&](const std::string & why) -> void * {
        std::lock_guard<std::mutex> lock(log_mtx);
        if (logged.insert(device->name).second) {
            GGML_LOG_WARN("ggml_backend_vk_vmm_arena_new: %s: no growable (chunked) KV buffers: %s; the KV grows by copy\n", device->name.c_str(), why.c_str());
        }
        return nullptr;
    };
    if (!device->buffer_device_address) {
        return no("the device has no bufferDeviceAddress (the shaders reach the K/V chunks by their device addresses)");
    }
    if (!device->shader_int64) {
        return no("the device has no shaderInt64 (64-bit chunk addresses in the shaders)");
    }

    // the granularity: the alignment of an ordinary device buffer (probed with a small one) and its memory type
    vk::MemoryRequirements req;
    try {
        vk::BufferCreateInfo bci({}, 1u << 20, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc |
            vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eShaderDeviceAddress, vk::SharingMode::eExclusive);
        vk::Buffer probe = device->device.createBuffer(bci);
        req = device->device.getBufferMemoryRequirements(probe);
        device->device.destroyBuffer(probe);
    } catch (const vk::SystemError & e) {
        return no(std::string("creating a device buffer failed: ") + e.what());
    }
    const size_t block = std::max<size_t>(req.alignment, 1);
    int mem_type = -1;
    {
        const vk::PhysicalDeviceMemoryProperties mp0 = device->physical_device.getMemoryProperties();
        for (uint32_t i = 0; i < mp0.memoryTypeCount; ++i) {
            if ((req.memoryTypeBits & (1u << i)) && (mp0.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal)) {
                mem_type = (int) i;
                break;
            }
        }
    }
    if (mem_type < 0) {
        return no("no device local memory type");
    }
    // the chunk, by the rule of the ggml-cuda VMM: the granularity, the 2 MiB section alignment (GGML_CUDA_VMM_MIN_ALIGN:
    // whole large-page fragments for the K/V reads), the unit the driver books an allocation with (measured: the memory
    // budget's heap usage before and after one allocation, twice; at most 32 alignments); then the Vulkan-only bounds: the
    // release unit (a freed chunk must give its memory back) and the allocation count limit (the chunks of one KV use at
    // most a quarter of maxMemoryAllocationCount); GGML_VK_KV_CHUNK_KIB overrides it (test aid)
    static constexpr size_t min_align = 2ull << 20;
    const size_t align = block * ((std::max(block, min_align) + block - 1) / block);
    std::string unit_note;
    const size_t unit = vk_kvc_booking_unit(device, (uint32_t) mem_type, block, unit_note);
    size_t chunk = align;
    if (unit > 0 && std::lcm(align, unit) <= 32*align) {
        chunk = std::lcm(align, unit);
    }
    std::string release_note;
    const size_t release = vk_kvc_release_unit(device, (uint32_t) mem_type, chunk, release_note);
    chunk = std::max(chunk, release);
    const uint32_t alloc_limit = device->properties.limits.maxMemoryAllocationCount;
    const size_t   max_chunks  = std::max<size_t>(1, alloc_limit / 4);
    const size_t   count_min   = (std::max<size_t>(size, 1) + max_chunks - 1) / max_chunks;
    if (chunk < count_min) {
        chunk = chunk * ((count_min + chunk - 1) / chunk);
    }
    const char * env_chunk = getenv("GGML_VK_KV_CHUNK_KIB");
    if (env_chunk && atoll(env_chunk) > 0) {
        chunk = block * std::max<size_t>(1, (((size_t) atoll(env_chunk) << 10) + block - 1) / block);
    }
    {
        std::lock_guard<std::mutex> lock(log_mtx);
        if (logged.insert(device->name + "/chunk").second) {
            GGML_LOG_INFO("ggml_backend_vk_vmm_arena_new: %s: chunked KV buffers: granularity %zu; section alignment %zu; driver booking unit %zu%s; release unit %zu%s; "
                "allocation limit %u (chunk >= %zu for this arena); chunk %zu%s, a section's chunk is a whole number of its rows (a multiple of 256) at or above it\n",
                device->name.c_str(), block, align, unit, unit_note.c_str(), release, release_note.c_str(), alloc_limit, count_min, chunk,
                env_chunk ? " (GGML_VK_KV_CHUNK_KIB)" : "");
        }
    }

    auto * a = new vk_kvc_arena;
    a->device = device;
    a->buft   = buft;
    a->chunk  = chunk;
    a->unit   = std::max(unit, block);
    if (vk_kvc_debug()) {
        GGML_LOG_INFO("ggml_backend_vk_vmm_arena_new: arena for %zu bytes of sections, chunk %zu\n", size, chunk);
    }
    return a;
}

static size_t ggml_backend_vk_vmm_arena_alignment(void * arena) {
    return ((vk_kvc_arena *) arena)->chunk;
}

static void ggml_backend_vk_vmm_arena_free(void * arena) {
    vk_kvc_arena_unref((vk_kvc_arena *) arena);
}

static void ggml_backend_vk_kvc_free_buffer(ggml_backend_buffer_t buffer);
static void * ggml_backend_vk_kvc_get_base(ggml_backend_buffer_t buffer);
static enum ggml_status ggml_backend_vk_kvc_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor);
static void ggml_backend_vk_kvc_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size);
static void ggml_backend_vk_kvc_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size);
static void ggml_backend_vk_kvc_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size);
static void ggml_backend_vk_kvc_set_tensor_2d(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset,
                                              size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
static void ggml_backend_vk_kvc_get_tensor_2d(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset,
                                              size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);
static void ggml_backend_vk_kvc_clear(ggml_backend_buffer_t buffer, uint8_t value);

static ggml_backend_buffer_i ggml_backend_vk_kvc_interface = {
    /* .free_buffer     = */ ggml_backend_vk_kvc_free_buffer,
    /* .get_base        = */ ggml_backend_vk_kvc_get_base,
    /* .init_tensor     = */ ggml_backend_vk_kvc_init_tensor,
    /* .memset_tensor   = */ ggml_backend_vk_kvc_memset_tensor,
    /* .set_tensor      = */ ggml_backend_vk_kvc_set_tensor,
    /* .get_tensor      = */ ggml_backend_vk_kvc_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_vk_kvc_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_vk_kvc_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_vk_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_vk_kvc_clear,
    /* .reset           = */ NULL,
    /* .set_tensor_n    = */ NULL,
    /* .get_tensor_n    = */ NULL,
    /* .cpy_tensor_n    = */ NULL,
};

static vk_kvc_section * vk_kvc_ctx(ggml_backend_buffer_t buffer) {
    return buffer && buffer->iface.free_buffer == ggml_backend_vk_kvc_free_buffer ? (vk_kvc_section *) buffer->context : nullptr;
}

bool ggml_vk_buffer_chunked(ggml_backend_buffer_t buffer) {
    return vk_kvc_ctx(buffer) != nullptr;
}

bool ggml_vk_tensor_chunked(const ggml_tensor * t) {
    if (!t) {
        return false;
    }
    ggml_backend_buffer_t b = t->buffer ? t->buffer : (t->view_src ? t->view_src->buffer : nullptr);
    return vk_kvc_ctx(b) != nullptr;
}

// allocate the chunks up to size (zeroed by the next commit)
static bool vk_kvc_map_to(vk_kvc_section * s, size_t size) {
    const size_t want = std::min(s->n_entries(), (std::max<size_t>(size, 1) + s->chunk - 1) / s->chunk);
    std::lock_guard<std::recursive_mutex> lock(s->mtx);
    bool added = false;
    while (s->chunks.size() < want) {
        vk_buffer c = vk_kvc_alloc_chunk(s->arena, s->chunk);
        if (!c) {
            return false;
        }
        s->chunks.push_back(c);
        s->n_pending++;
        vk_kvc_stat_bytes += (int64_t) s->chunk;
        added = true;
    }
    if (added) {
        std::lock_guard<std::mutex> alock(s->arena->mtx);
        if (std::find(s->arena->pending.begin(), s->arena->pending.end(), s) == s->arena->pending.end()) {
            s->arena->pending.push_back(s);
        }
    }
    if (!s->dev_buffer && !s->chunks.empty()) {
        s->dev_buffer = s->chunks[0]; // the section's identity for the overlap checks (offsets are section offsets)
    }
    return true;
}

// zero the new chunks of a section and point its tables at them; nothing may access them before
static void vk_kvc_commit_section(vk_kvc_section * s) {
    std::lock_guard<std::recursive_mutex> lock(s->mtx);
    for (size_t i = s->chunks.size() - s->n_pending; i < s->chunks.size(); ++i) {
        ggml_vk_buffer_memset(s->chunks[i], 0, 0, s->chunk);
    }
    s->n_pending = 0;
    vk_kvc_tables_fill(s);
}

static bool ggml_backend_vk_vmm_arena_commit(void * arena) {
    vk_kvc_arena * a = (vk_kvc_arena *) arena;
    std::vector<vk_kvc_section *> todo;
    {
        std::lock_guard<std::mutex> lock(a->mtx);
        todo.swap(a->pending);
    }
    const int64_t t0 = ggml_time_us();
    for (vk_kvc_section * s : todo) {
        vk_kvc_commit_section(s);
    }
    if (vk_kvc_debug() && !todo.empty()) {
        GGML_LOG_INFO("ggml_backend_vk_vmm_arena_commit: new chunks of %zu sections zeroed (%.2f ms)\n", todo.size(), (ggml_time_us() - t0) / 1000.0);
    }
    return true;
}

static ggml_backend_buffer_t ggml_backend_vk_vmm_arena_buffer(void * arena, size_t offset, size_t size, size_t max_size) {
    GGML_UNUSED(offset); // a section of its own
    vk_kvc_arena * a = (vk_kvc_arena *) arena;
    const size_t section = std::max(size, max_size);
    if (section >= ((size_t) 1 << 32)) {
        GGML_LOG_WARN("ggml_backend_vk_vmm_arena_buffer: %s: a K/V section of %zu MiB: the shaders address a section with 32-bit offsets; it grows by copy\n",
            a->device->name.c_str(), section >> 20);
        return nullptr;
    }
    auto * btctx = (ggml_backend_vk_buffer_type_context *) a->buft->context;
    {
        std::lock_guard<std::mutex> lock(a->mtx);
        a->refs++;
    }
    // the device chunk until the tensor is placed (its rows set the section's chunk, ggml_backend_vk_kvc_init_tensor)
    auto * s = new vk_kvc_section(a->device, btctx->name, a, section, a->chunk);
    {
        std::lock_guard<std::mutex> lock(vk_kvc_reg_mtx);
        vk_kvc_live.insert(s);
    }
    ggml_backend_buffer_t buf = ggml_backend_buffer_init(a->buft, ggml_backend_vk_kvc_interface, s, size);
    if (!vk_kvc_map_to(s, size)) {
        GGML_LOG_WARN("ggml_backend_vk_vmm_arena_buffer: %s: no memory for a K/V section of %zu MiB\n", a->device->name.c_str(), size >> 20);
        ggml_backend_buffer_free(buf);
        return nullptr;
    }
    s->logical = size;
    return buf;
}

// the tensor of a section sets its chunk: a whole number of its rows, so that no row (no element, no block, no shader tile
// of up to 256 rows) is split between two chunks; the chunks made before (zeroed, no data yet) are made again if needed
static enum ggml_status ggml_backend_vk_kvc_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    vk_kvc_section * s = vk_kvc_ctx(buffer);
    if (tensor->view_src == nullptr && s->row == 0) {
        std::lock_guard<std::recursive_mutex> lock(s->mtx);
        GGML_ASSERT(vk_tensor_offset(tensor) == 0 && "a chunked K/V section holds one tensor at its start");
        s->row = tensor->nb[1];
        const size_t chunk = vk_kvc_chunk_for_row(s->arena, s->row);
        if (chunk == 0) {
            GGML_LOG_ERROR("%s: a K/V row of %zu bytes: no chunk of a whole number of rows fits 31 bits\n", __func__, s->row);
            return GGML_STATUS_FAILED;
        }
        if (chunk != s->chunk) {
            const size_t n = s->chunks.size();
            vk_kvc_stat_bytes -= (int64_t) (n * s->chunk);
            s->chunks.clear();
            s->dev_buffer = nullptr;
            s->n_pending = 0;
            s->tables.clear();
            s->chunk = chunk;
            if (!vk_kvc_map_to(s, s->logical)) {
                GGML_LOG_ERROR("%s: no memory for the chunks of a K/V section of %zu MiB\n", __func__, s->logical >> 20);
                return GGML_STATUS_ALLOC_FAILED;
            }
            vk_kvc_commit_section(s);
            std::lock_guard<std::mutex> alock(s->arena->mtx);
            s->arena->pending.erase(std::remove(s->arena->pending.begin(), s->arena->pending.end(), s), s->arena->pending.end());
        }
        if (vk_kvc_debug()) {
            GGML_LOG_INFO("%s: K/V section of %zu bytes: rows of %zu bytes, chunk %zu (%zu rows), %zu chunks\n", __func__, s->section, s->row, s->chunk,
                s->chunk / s->row, s->chunks.size());
        }
    }
    return ggml_backend_vk_buffer_init_tensor(buffer, tensor);
}

static bool ggml_backend_vk_buffer_grow(ggml_backend_buffer_t buffer, size_t size) {
    vk_kvc_section * s = vk_kvc_ctx(buffer);
    if (!s) {
        return false;
    }
    if (size <= buffer->size) {
        return true;
    }
    if (size > s->section || !vk_kvc_map_to(s, size)) {
        return false;
    }
    buffer->size = size;
    s->logical   = size;
    return true;
}

static bool ggml_backend_vk_buffer_grow_commit(ggml_backend_buffer_t buffer) {
    vk_kvc_section * s = vk_kvc_ctx(buffer);
    return s && ggml_backend_vk_vmm_arena_commit(s->arena);
}

// free the chunks of a section from byte keep on (keep is rounded up to whole chunks, at least one stays)
static void vk_kvc_unmap_from(vk_kvc_section * s, size_t keep_bytes) {
    std::lock_guard<std::recursive_mutex> lock(s->mtx);
    const size_t keep = std::max<size_t>(1, (keep_bytes + s->chunk - 1) / s->chunk);
    if (s->chunks.size() <= keep) {
        return;
    }
    s->arena->device->device.waitIdle(); // no queue may use a chunk when it goes
    const size_t n_free = s->chunks.size() - keep;
    s->n_pending -= std::min(s->n_pending, n_free);
    for (size_t i = keep; i < s->chunks.size(); ++i) {
        ggml_vk_destroy_buffer(s->chunks[i]);
    }
    s->chunks.resize(keep);
    vk_kvc_stat_bytes -= (int64_t) (n_free * s->chunk);
    vk_kvc_tables_fill(s);
}

static bool ggml_backend_vk_buffer_shrink(ggml_backend_buffer_t buffer, size_t size) {
    vk_kvc_section * s = vk_kvc_ctx(buffer);
    if (!s) {
        return false;
    }
    if (size >= buffer->size) {
        return true;
    }
    const size_t old = s->mapped();
    vk_kvc_unmap_from(s, std::max<size_t>(size, 1));
    buffer->size = size;
    s->logical   = size;
    if (vk_kvc_debug()) {
        GGML_LOG_INFO("ggml_backend_vk_buffer_shrink: section of %zu bytes: %zu -> %zu bytes in chunks\n", s->section, old, s->mapped());
    }
    return true;
}

static size_t ggml_backend_vk_buffer_vmm_chunk(ggml_backend_buffer_t buffer) {
    vk_kvc_section * s = vk_kvc_ctx(buffer);
    return s ? s->chunk : 0;
}

// vkFreeMemory gives the memory back at once: an in-place shrink frees it
static bool ggml_backend_vk_vmm_unmap_keeps_memory(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return false;
}

// stats[4]: live arenas (not tracked: 0), bytes in chunks, commit timings (0)
static void ggml_backend_vk_vmm_stats(int64_t * stats) {
    stats[0] = 0;
    stats[1] = vk_kvc_stat_bytes;
    stats[2] = 0;
    stats[3] = 0;
}

static void ggml_backend_vk_kvc_free_buffer(ggml_backend_buffer_t buffer) {
    vk_kvc_section * s = (vk_kvc_section *) buffer->context;
    {
        std::lock_guard<std::mutex> lock(vk_kvc_reg_mtx);
        vk_kvc_live.erase(s);
    }
    {
        std::lock_guard<std::mutex> lock(s->arena->mtx);
        s->arena->pending.erase(std::remove(s->arena->pending.begin(), s->arena->pending.end(), s), s->arena->pending.end());
    }
    vk_kvc_stat_bytes -= (int64_t) s->mapped();
    for (auto & c : s->chunks) {
        ggml_vk_destroy_buffer(c);
    }
    s->chunks.clear();
    s->tables.clear();
    s->dev_buffer = nullptr;
    vk_kvc_arena * a = s->arena;
    delete s;
    vk_kvc_arena_unref(a);
}

static void * ggml_backend_vk_kvc_get_base(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return vk_ptr_base; // a section of its own: the tensor starts at offset 0
}

// the pieces of bytes [offset, offset + size) of a Vulkan buffer: one per chunk of a chunked section, else the whole range
std::vector<vk_kvc_piece> ggml_vk_buffer_pieces(ggml_backend_buffer_t buffer, size_t offset, size_t size) {
    std::vector<vk_kvc_piece> res;
    vk_kvc_section * s = vk_kvc_ctx(buffer);
    if (!s) {
        res.push_back({ ((ggml_backend_vk_buffer_context *) buffer->context)->dev_buffer, offset, size, 0 });
        return res;
    }
    std::lock_guard<std::recursive_mutex> lock(s->mtx);
    GGML_ASSERT(offset + size <= s->mapped());
    size_t done = 0;
    while (done < size) {
        const size_t pos = offset + done;
        const size_t i   = pos / s->chunk;
        const size_t off = pos - i * s->chunk;
        const size_t len = std::min(size - done, s->chunk - off);
        res.push_back({ s->chunks[i], off, len, done });
        done += len;
    }
    return res;
}

// the binding of a chunked tensor that starts at section offset `base`: its chunk table (made on first use)
vk_subbuffer ggml_vk_kvc_subbuffer(const ggml_tensor * t, size_t base) {
    vk_kvc_section * s = vk_kvc_ctx(t->buffer ? t->buffer : t->view_src->buffer);
    GGML_ASSERT(s && base < ((size_t) 1 << 32));
    std::lock_guard<std::recursive_mutex> lock(s->mtx);
    for (auto & tb : s->tables) {
        if (tb.base == base) {
            return { tb.buf, 0, tb.buf->size };
        }
    }
    const size_t bytes = (2 + s->n_entries()) * sizeof(uint64_t);
    vk_kvc_table tb { base, nullptr };
    tb.buf = ggml_vk_create_buffer(s->arena->device, bytes, { vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                              vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                                                              vk::MemoryPropertyFlagBits::eDeviceLocal });
    vk_kvc_table_fill(s, tb);
    s->tables.push_back(tb);
    if (vk_kvc_debug()) {
        GGML_LOG_INFO("ggml_vk_kvc_subbuffer: section of %zu bytes: a chunk table for binding offset %zu (%zu entries)\n", s->section, base, s->n_entries());
    }
    return { tb.buf, 0, tb.buf->size };
}

static void ggml_backend_vk_kvc_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    for (auto & p : ggml_vk_buffer_pieces(buffer, vk_tensor_offset(tensor) + tensor->view_offs + offset, size)) {
        ggml_vk_buffer_memset(p.buf, p.off, (uint32_t) value * 0x01010101, p.len);
    }
}

static void ggml_backend_vk_kvc_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    for (auto & p : ggml_vk_buffer_pieces(buffer, vk_tensor_offset(tensor) + tensor->view_offs + offset, size)) {
        ggml_vk_buffer_write(p.buf, p.off, (const uint8_t *) data + p.done, p.len);
    }
}

static void ggml_backend_vk_kvc_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    for (auto & p : ggml_vk_buffer_pieces(buffer, vk_tensor_offset(tensor) + tensor->view_offs + offset, size)) {
        ggml_vk_buffer_read(p.buf, p.off, (uint8_t *) data + p.done, p.len);
    }
}

static void ggml_backend_vk_kvc_set_tensor_2d(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset,
                                              size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    for (size_t i = 0; i < n_copies; ++i) {
        ggml_backend_vk_kvc_set_tensor(buffer, tensor, (const uint8_t *) data + i * stride_data, offset + i * stride_tensor, size);
    }
}

static void ggml_backend_vk_kvc_get_tensor_2d(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset,
                                              size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    for (size_t i = 0; i < n_copies; ++i) {
        ggml_backend_vk_kvc_get_tensor(buffer, tensor, (uint8_t *) data + i * stride_data, offset + i * stride_tensor, size);
    }
}

static void ggml_backend_vk_kvc_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    vk_kvc_section * s = (vk_kvc_section *) buffer->context;
    if (buffer->size > 0) {
        for (auto & p : ggml_vk_buffer_pieces(buffer, 0, buffer->size)) {
            ggml_vk_buffer_memset(p.buf, p.off, (uint32_t) value * 0x01010101, p.len);
        }
    }
    GGML_UNUSED(s);
}

// a copy between two Vulkan buffers of which at least one is chunked: piece by piece
void ggml_vk_buffer_copy_pieces(ggml_backend_buffer_t dst, size_t dst_offset, ggml_backend_buffer_t src, size_t src_offset, size_t size) {
    for (auto & d : ggml_vk_buffer_pieces(dst, dst_offset, size)) {
        for (auto & sp : ggml_vk_buffer_pieces(src, src_offset + d.done, d.len)) {
            ggml_vk_buffer_copy(d.buf, d.off + sp.done, sp.buf, sp.off, sp.len);
        }
    }
}

// device memory of the chunked K/V sections above their ggml buffer size (whole chunks) and their tables, per device
size_t ggml_vk_kvc_unbooked(size_t device_idx) {
    std::lock_guard<std::mutex> lock(vk_kvc_reg_mtx);
    size_t res = 0;
    for (vk_kvc_section * s : vk_kvc_live) {
        if (s->arena->device->idx != device_idx) {
            continue;
        }
        std::lock_guard<std::recursive_mutex> slock(s->mtx);
        if (s->mapped() > s->logical) {
            res += s->mapped() - s->logical;
        }
        for (const auto & t : s->tables) {
            if (t.buf->memory_property_flags & vk::MemoryPropertyFlagBits::eDeviceLocal) {
                res += t.buf->size;
            }
        }
    }
    return res;
}


// true if ggml_backend_dev_memory measures the free memory (VK_EXT_memory_budget); without it the free memory is the heap size
static bool ggml_backend_vk_dev_memory_measured(ggml_backend_dev_t dev) {
    const auto * ctx = (const ggml_backend_vk_device_context *) dev->context;
    const bool measured = ctx && ctx->device < vk_instance.device_supports_membudget.size() && vk_instance.device_supports_membudget[ctx->device];
    if (!measured && ctx) {
        static std::mutex mtx;
        static std::set<size_t> logged;
        std::lock_guard<std::mutex> lock(mtx);
        if (logged.insert(ctx->device).second) {
            GGML_LOG_WARN("ggml_backend_vk_dev_memory_measured: %s: no VK_EXT_memory_budget%s: the free memory it reports is the heap size, not a measurement\n",
                ctx->name.c_str(), getenv("GGML_VK_DISABLE_MEMORY_BUDGET") ? " (GGML_VK_DISABLE_MEMORY_BUDGET)" : "");
        }
    }
    return measured;
}

// the scheduler reads a split's output into host memory through get_tensor_async + synchronize: a synchronous read is a
// command buffer, a submission and a fence wait of its own after the wait for the graph (per split boundary in a GPU/CPU
// split); with it the read is recorded behind the graph and goes with its fence: one wait per boundary, as the HIP path
// (async copy + stream synchronize). On by default; GGML_VK_ASYNC_READBACK=0 turns it off (test aid). A device without async
// support has no get_tensor_async, the scheduler then keeps the synchronous read.
bool ggml_vk_low_latency_sync();

static bool ggml_backend_vk_dev_async_readback(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    static const bool on = (!getenv("GGML_VK_ASYNC_READBACK") || atoi(getenv("GGML_VK_ASYNC_READBACK")) != 0) && ggml_vk_low_latency_sync();
    return on;
}

size_t ggml_vk_internal_device_bytes(size_t device_idx);

// device memory the backend holds outside the ggml buffers' sizes (its own graph buffers, the sparse chunk rounding): the
// zones booking adds it to the compute buffers, so the booked free memory follows the measured one
static size_t ggml_backend_vk_dev_internal_memory(ggml_backend_dev_t dev) {
    const auto * ctx = (const ggml_backend_vk_device_context *) dev->context;
    return ctx ? ggml_vk_internal_device_bytes((size_t) ctx->device) : 0;
}

std::atomic<int64_t> ggml_vk_host_alloc_bytes{0};

// host memory the backend allocated through the driver (the Vulkan_Host buffers, staging, pinned copies); the caller measures
// whether the process' cgroup sees it (a driver allocation, e.g. the GTT of a discrete card, need not be charged there)
static size_t ggml_backend_vk_dev_host_memory(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return (size_t) std::max<int64_t>(0, ggml_vk_host_alloc_bytes.load());
}

void * ggml_vk_kvc_proc_address(const char * name) {
    if (strcmp(name, "ggml_backend_dev_host_memory") == 0)       { return (void *) ggml_backend_vk_dev_host_memory; }
    if (strcmp(name, "ggml_backend_dev_internal_memory") == 0)   { return (void *) ggml_backend_vk_dev_internal_memory; }
    if (strcmp(name, "ggml_backend_dev_async_readback") == 0)   { return (void *) ggml_backend_vk_dev_async_readback; }
    if (strcmp(name, "ggml_backend_dev_memory_measured") == 0)  { return (void *) ggml_backend_vk_dev_memory_measured; }
    if (strcmp(name, "ggml_backend_vmm_arena_new") == 0)          { return (void *) ggml_backend_vk_vmm_arena_new; }
    if (strcmp(name, "ggml_backend_vmm_arena_alignment") == 0)    { return (void *) ggml_backend_vk_vmm_arena_alignment; }
    if (strcmp(name, "ggml_backend_vmm_arena_buffer") == 0)       { return (void *) ggml_backend_vk_vmm_arena_buffer; }
    if (strcmp(name, "ggml_backend_vmm_arena_commit") == 0)       { return (void *) ggml_backend_vk_vmm_arena_commit; }
    if (strcmp(name, "ggml_backend_vmm_arena_free") == 0)         { return (void *) ggml_backend_vk_vmm_arena_free; }
    if (strcmp(name, "ggml_backend_buffer_grow") == 0)            { return (void *) ggml_backend_vk_buffer_grow; }
    if (strcmp(name, "ggml_backend_buffer_grow_commit") == 0)     { return (void *) ggml_backend_vk_buffer_grow_commit; }
    if (strcmp(name, "ggml_backend_buffer_shrink") == 0)          { return (void *) ggml_backend_vk_buffer_shrink; }
    if (strcmp(name, "ggml_backend_buffer_vmm_chunk") == 0)       { return (void *) ggml_backend_vk_buffer_vmm_chunk; }
    if (strcmp(name, "ggml_backend_vmm_unmap_keeps_memory") == 0) { return (void *) ggml_backend_vk_vmm_unmap_keeps_memory; }
    if (strcmp(name, "ggml_backend_vmm_stats") == 0)              { return (void *) ggml_backend_vk_vmm_stats; }
    return nullptr;
}
