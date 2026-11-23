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
    // Buffer is already mapped
    if(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible) {
        GGML_ASSERT(dst->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent);

        if (width == spitch && width == dpitch) {
            memcpy((uint8_t *)dst->ptr + offset, src, width * height);
        } else {
            for (size_t i = 0; i < height; i++) {
                memcpy((uint8_t *)dst->ptr + offset + i * dpitch, (const uint8_t *) src + i * spitch, width);
            }
        }
    } else {
        std::lock_guard<std::recursive_mutex> guard(dst->device->mutex);

        vk_context subctx = ggml_vk_create_temporary_context(dst->device->transfer_queue->cmd_pool);
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
        VK_CHECK(dst->device->device.waitForFences({ dst->device->fence }, true, UINT64_MAX), "vk_buffer_write_2d waitForFences", dst->device);
        dst->device->device.resetFences({ dst->device->fence });
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
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, true, UINT64_MAX),
                 "vk_buffer_read_2d uma waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
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

        vk_context subctx = ggml_vk_create_temporary_context(src->device->transfer_queue->cmd_pool);
        ggml_vk_ctx_begin(src->device, subctx);
        bool ret = ggml_vk_buffer_read_2d_async(subctx, src, offset, dst, spitch, dpitch, width, height, true);
        GGML_ASSERT(ret);
        ggml_vk_ctx_end(subctx);

        ggml_vk_submit(subctx, src->device->fence);
        VK_CHECK(src->device->device.waitForFences({ src->device->fence }, true, UINT64_MAX), "vk_buffer_read_2d waitForFences", src->device);
        src->device->device.resetFences({ src->device->fence });
        ggml_vk_queue_command_pools_cleanup(src->device);

        for (auto& cpy : subctx->out_memcpys) {
            memcpy(cpy.dst, cpy.src, cpy.n);
        }
    }
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


// growable buffers on sparse VkBuffers (dual-buffer fork, dynamic KV): the same interface as the ggml-cuda VMM arenas (looked
// up with get_proc_address). The arena holds what a relayout's sections on one device share (the chunk, the memory type, the
// pending binds); every section (one K or V) is a sparse VkBuffer of its own: an address range only, device memory is bound
// per chunk when it grows (vkQueueBindSparse) and unbound and freed when it shrinks, so the memory comes back without a copy.
// The address range: the section's max size if the device's sparse address space (sparseAddressSpaceSize, maxBufferSize)
// allows it, else what it needs now; a section that outgrows its range gets a larger one and its memory chunks are bound
// into it (no data copy), the old range goes (the HIP arena reserves its address space per chunk with the mapping: there the
// address space follows the mapped memory). Needs sparseBinding, sparseResidencyBuffer and a compute queue with sparse binding;
// else no arena (the KV grows by copy, with a log line naming the missing capability). GGML_VK_SPARSE=0 turns it off,
// GGML_VK_SPARSE_CHUNK_KIB sets the chunk, GGML_VK_SPARSE_DEBUG=1 logs every bind, GGML_VK_SPARSE_VA_MIB caps the sparse
// address space (test aid for the rebinding).

struct vk_sparse_section;

struct vk_sparse_arena {
    vk_device                       device;
    ggml_backend_buffer_type_t      buft;
    vk::Queue                       queue;    // the compute queue (sparse binding)
    vk_queue_handle *               qh = nullptr;
    vk::BufferCreateInfo            bci;      // a sparse buffer of this device (size set per range)
    size_t                          chunk    = 0;
    uint32_t                        mem_type = 0;
    bool                            on_demand = false; // the sections at their max size do not fit the sparse address space
    std::vector<vk_sparse_section *> pending; // sections with chunks allocated, not bound and zeroed yet
    int                             refs = 1;
    std::mutex                      mtx;
};

struct vk_sparse_section : ggml_backend_vk_buffer_context {
    vk_sparse_arena *             arena;
    size_t                        section;     // max bytes
    size_t                        range = 0;   // bytes of the address range (dev_buffer)
    size_t                        mapped = 0;  // bytes with memory
    size_t                        logical = 0; // the ggml buffer size
    std::vector<vk::DeviceMemory> mem;         // per chunk, null if none
    std::vector<size_t>           pending;     // chunks allocated, not bound and zeroed yet

    vk_sparse_section(vk_device_ref dev, vk_buffer buf, std::string & name, vk_sparse_arena * arena, size_t section) :
        ggml_backend_vk_buffer_context(dev, std::move(buf), name), arena(arena), section(section) {}
};

static bool vk_sparse_debug() {
    static const bool on = getenv("GGML_VK_SPARSE_DEBUG") && atoi(getenv("GGML_VK_SPARSE_DEBUG")) != 0;
    return on;
}

static std::atomic<int64_t> vk_sparse_stat_bound{0};

// the live sections and the sparse address space they hold, per device
static std::mutex                    vk_sparse_reg_mtx;
static std::set<vk_sparse_section *> vk_sparse_live;
static std::map<size_t, size_t>      vk_sparse_va; // device idx -> bytes of sparse address ranges

// the sparse address space a device allows: sparseAddressSpaceSize (GGML_VK_SPARSE_VA_MIB lowers it, test aid)
static size_t vk_sparse_va_limit(const vk_device & device) {
    size_t lim = (size_t) device->properties.limits.sparseAddressSpaceSize;
    if (const char * env = getenv("GGML_VK_SPARSE_VA_MIB"); env && atoll(env) > 0) {
        lim = std::min(lim, (size_t) atoll(env) << 20);
    }
    return lim;
}

static void vk_sparse_arena_unref(vk_sparse_arena * a) {
    int left;
    {
        std::lock_guard<std::mutex> lock(a->mtx);
        left = --a->refs;
    }
    if (left > 0) {
        return;
    }
    if (vk_sparse_debug()) {
        GGML_LOG_INFO("%s: arena freed\n", __func__);
    }
    delete a;
}

// bind (with their memory) or unbind the chunks idx of a section in its range buf, and wait for it
static bool vk_sparse_bind(vk_sparse_section * s, const vk_buffer & buf, const std::vector<size_t> & idx, bool bind) {
    if (idx.empty()) {
        return true;
    }
    vk_sparse_arena * a = s->arena;
    std::vector<vk::SparseMemoryBind> binds;
    binds.reserve(idx.size());
    for (size_t i : idx) {
        binds.push_back(vk::SparseMemoryBind(i * a->chunk, a->chunk, bind ? s->mem[i] : vk::DeviceMemory(), 0));
    }
    vk::SparseBufferMemoryBindInfo buf_bind(buf->buffer, (uint32_t) binds.size(), binds.data());
    vk::BindSparseInfo info;
    info.setBufferBinds(buf_bind);
    vk::Fence fence = a->device->device.createFence({});
    bool ok = true;
    {
        std::lock_guard<std::recursive_mutex> guard(a->device->mutex);
        if (a->qh) {
            a->qh->lock();
        }
        try {
            a->queue.bindSparse(info, fence);
        } catch (const vk::SystemError & e) {
            GGML_LOG_ERROR("%s: %s of %zu chunks failed: %s\n", __func__, bind ? "binding" : "unbinding", idx.size(), e.what());
            ok = false;
        }
        if (a->qh) {
            a->qh->unlock();
        }
    }
    if (ok) {
        ok = a->device->device.waitForFences({ fence }, true, UINT64_MAX) == vk::Result::eSuccess;
    }
    a->device->device.destroyFence(fence);
    return ok;
}

// a sparse address range of bytes (no memory), booked in the device's sparse address space; nullptr if the device does not
// allow it (the limit, maxBufferSize, or the creation fails). extra: bytes of a range that is about to be given back
static vk_buffer vk_sparse_range_new(vk_sparse_arena * a, size_t bytes, std::string * why) {
    const vk_device & device = a->device;
    {
        std::lock_guard<std::mutex> lock(vk_sparse_reg_mtx);
        const size_t used = vk_sparse_va[device->idx];
        if (bytes > device->max_buffer_size || used + bytes > vk_sparse_va_limit(device)) {
            if (why) {
                *why = "sparse address space: " + std::to_string(used >> 20) + " MiB in use + " + std::to_string(bytes >> 20) + " MiB > " +
                    std::to_string(vk_sparse_va_limit(device) >> 20) + " MiB (sparseAddressSpaceSize), max buffer " + std::to_string(device->max_buffer_size >> 20) + " MiB";
            }
            return nullptr;
        }
        vk_sparse_va[device->idx] = used + bytes;
    }
    vk::BufferCreateInfo bci = a->bci;
    bci.size = bytes;
    vk::Buffer vkbuf;
    try {
        vkbuf = device->device.createBuffer(bci);
    } catch (const vk::SystemError & e) {
        std::lock_guard<std::mutex> lock(vk_sparse_reg_mtx);
        vk_sparse_va[device->idx] -= bytes;
        if (why) {
            *why = std::string("creating a sparse buffer of ") + std::to_string(bytes) + " bytes failed: " + e.what();
        }
        return nullptr;
    }
    auto buf = std::make_shared<vk_buffer_struct>();
    buf->buffer = vkbuf;
    buf->memory_property_flags = vk::MemoryPropertyFlagBits::eDeviceLocal;
    buf->ptr    = nullptr;
    buf->size   = bytes;
    buf->device = device;
    if (device->buffer_device_address) {
        buf->bda_addr = device->device.getBufferAddress(vk::BufferDeviceAddressInfo(vkbuf));
    }
    return buf;
}

static void vk_sparse_range_release(const vk_device & device, size_t bytes) {
    std::lock_guard<std::mutex> lock(vk_sparse_reg_mtx);
    vk_sparse_va[device->idx] -= std::min(vk_sparse_va[device->idx], bytes);
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
static size_t vk_sparse_booking_unit(vk_device & device, uint32_t mem_type, size_t size, std::string & note) {
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
static size_t vk_sparse_release_unit(vk_device & device, uint32_t mem_type, size_t base, std::string & note) {
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
            GGML_LOG_WARN("ggml_backend_vk_vmm_arena_new: %s: no growable (sparse) KV buffers: %s; the KV grows by copy\n", device->name.c_str(), why.c_str());
        }
        return nullptr;
    };
    if (const char * env = getenv("GGML_VK_SPARSE"); env && atoi(env) == 0) {
        return no("GGML_VK_SPARSE=0");
    }
    const vk::PhysicalDeviceFeatures feats = device->physical_device.getFeatures();
    if (!feats.sparseBinding) {
        return no("the device has no sparseBinding");
    }
    if (!feats.sparseResidencyBuffer) {
        return no("the device has no sparseResidencyBuffer");
    }
    const auto qprops = device->physical_device.getQueueFamilyProperties();
    const uint32_t qf = device->compute_queue->queue_family_index;
    if (qf >= qprops.size() || !(qprops[qf].queueFlags & vk::QueueFlagBits::eSparseBinding)) {
        return no("the compute queue family " + std::to_string(qf) + " has no VK_QUEUE_SPARSE_BINDING_BIT");
    }

    vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;
    if (device->buffer_device_address) {
        usage |= vk::BufferUsageFlagBits::eShaderDeviceAddress;
    }
    // the sparse block is the memory requirement's alignment: probe it with a small buffer
    vk::BufferCreateInfo bci(vk::BufferCreateFlagBits::eSparseBinding | vk::BufferCreateFlagBits::eSparseResidency, 1u << 20, usage, vk::SharingMode::eExclusive);
    vk::MemoryRequirements req;
    try {
        vk::Buffer probe = device->device.createBuffer(bci);
        req = device->device.getBufferMemoryRequirements(probe);
        device->device.destroyBuffer(probe);
    } catch (const vk::SystemError & e) {
        return no(std::string("creating a sparse buffer failed: ") + e.what());
    }
    const size_t block = std::max<size_t>(req.alignment, 1);
    int probe_type = -1;
    {
        const vk::PhysicalDeviceMemoryProperties mp0 = device->physical_device.getMemoryProperties();
        for (uint32_t i = 0; i < mp0.memoryTypeCount; ++i) {
            if ((req.memoryTypeBits & (1u << i)) && (mp0.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal)) {
                probe_type = (int) i;
                break;
            }
        }
    }
    if (probe_type < 0) {
        return no("no device local memory type for a sparse buffer");
    }
    // the chunk, by the rule of the ggml-cuda VMM: the sparse block (the granularity), the 2 MiB section alignment (as
    // GGML_CUDA_VMM_MIN_ALIGN: whole large-page fragments for the K/V reads), the unit the driver books an allocation with
    // (measured: the memory budget's heap usage before and after one allocation, twice; at most 32 alignments); then the
    // Vulkan-only bounds: the release unit (a freed chunk must give its memory back) and the allocation count limit (the
    // chunks of one KV use at most a quarter of maxMemoryAllocationCount); GGML_VK_SPARSE_CHUNK_KIB overrides it (test aid)
    static constexpr size_t min_align = 2ull << 20;
    const size_t align = block * ((std::max(block, min_align) + block - 1) / block);
    std::string unit_note;
    const size_t unit = vk_sparse_booking_unit(device, (uint32_t) probe_type, block, unit_note);
    size_t chunk = align;
    if (unit > 0 && std::lcm(align, unit) <= 32*align) {
        chunk = std::lcm(align, unit);
    }
    std::string release_note;
    const size_t release = vk_sparse_release_unit(device, (uint32_t) probe_type, chunk, release_note);
    chunk = std::max(chunk, release);
    const uint32_t alloc_limit = device->properties.limits.maxMemoryAllocationCount;
    const size_t   max_chunks  = std::max<size_t>(1, alloc_limit / 4);
    const size_t   count_min   = (std::max<size_t>(size, 1) + max_chunks - 1) / max_chunks;
    if (chunk < count_min) {
        chunk = chunk * ((count_min + chunk - 1) / chunk);
    }
    const char * env_chunk = getenv("GGML_VK_SPARSE_CHUNK_KIB");
    if (env_chunk && atoll(env_chunk) > 0) {
        chunk = block * std::max<size_t>(1, (((size_t) atoll(env_chunk) << 10) + block - 1) / block);
    }
    {
        std::lock_guard<std::mutex> lock(log_mtx);
        if (logged.insert(device->name + "/chunk").second) {
            GGML_LOG_INFO("ggml_backend_vk_vmm_arena_new: %s: sparse block %zu; section alignment %zu; driver booking unit %zu%s; release unit %zu%s; allocation limit %u (chunk >= %zu for this arena); chunk %zu%s; sparse address space %zu MiB, max buffer %zu MiB\n",
                device->name.c_str(), block, align, unit, unit_note.c_str(), release, release_note.c_str(), alloc_limit, count_min, chunk,
                env_chunk ? " (GGML_VK_SPARSE_CHUNK_KIB)" : "", vk_sparse_va_limit(device) >> 20, (size_t) (device->max_buffer_size >> 20));
        }
    }

    auto * a = new vk_sparse_arena;
    a->device   = device;
    a->buft     = buft;
    a->queue    = device->compute_queue->handle->queue;
    a->qh       = device->compute_queue->handle.get();
    a->bci      = bci;
    a->chunk    = chunk;
    a->mem_type = (uint32_t) probe_type;
    // size is the K/V of the arena's sections at the max context (with room for the alignment): if the sparse address space
    // cannot hold it, every section gets the range it needs now and grows by rebinding (not the first ones all of it)
    {
        std::lock_guard<std::mutex> lock(vk_sparse_reg_mtx);
        a->on_demand = vk_sparse_va[device->idx] + size > vk_sparse_va_limit(device);
    }
    if (a->on_demand) {
        std::lock_guard<std::mutex> lock(log_mtx);
        if (logged.insert(device->name + "/va").second) {
            GGML_LOG_INFO("ggml_backend_vk_vmm_arena_new: %s: the K/V sections at the max context (%zu MiB) do not fit the sparse address space (%zu MiB): "
                "a section gets the range it needs and grows by binding its memory into a larger range (no copy)\n", device->name.c_str(), size >> 20,
                vk_sparse_va_limit(device) >> 20);
        }
    }
    {
        std::lock_guard<std::mutex> lock(log_mtx);
        if (logged.insert(device->name + "/ok").second) {
            GGML_LOG_INFO("ggml_backend_vk_vmm_arena_new: %s: sparse KV buffers: block %zu, chunk %zu bytes, memory type %d, a sparse range per section\n", device->name.c_str(),
                block, chunk, probe_type);
        }
    }
    if (vk_sparse_debug()) {
        GGML_LOG_INFO("ggml_backend_vk_vmm_arena_new: arena for %zu bytes of sections\n", size);
    }
    return a;
}

static size_t ggml_backend_vk_vmm_arena_alignment(void * arena) {
    return ((vk_sparse_arena *) arena)->chunk;
}

static void ggml_backend_vk_vmm_arena_free(void * arena) {
    vk_sparse_arena_unref((vk_sparse_arena *) arena);
}

static void ggml_backend_vk_sparse_free_buffer(ggml_backend_buffer_t buffer);
static void * ggml_backend_vk_sparse_get_base(ggml_backend_buffer_t buffer);
static void ggml_backend_vk_sparse_clear(ggml_backend_buffer_t buffer, uint8_t value);

static ggml_backend_buffer_i ggml_backend_vk_sparse_interface = {
    /* .free_buffer     = */ ggml_backend_vk_sparse_free_buffer,
    /* .get_base        = */ ggml_backend_vk_sparse_get_base,
    /* .init_tensor     = */ ggml_backend_vk_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_vk_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_vk_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_vk_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_vk_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_vk_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_vk_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_vk_sparse_clear,
    /* .reset           = */ NULL,
    /* .set_tensor_n    = */ NULL,
    /* .get_tensor_n    = */ NULL,
    /* .cpy_tensor_n    = */ NULL,
};

static vk_sparse_section * vk_sparse_ctx(ggml_backend_buffer_t buffer) {
    return buffer && buffer->iface.free_buffer == ggml_backend_vk_sparse_free_buffer ? (vk_sparse_section *) buffer->context : nullptr;
}

// a larger address range for a section that outgrows its own: the bound chunks are bound into the new range (no data copy),
// then the old range goes; the device is idle meanwhile (nothing may use the old range)
static bool vk_sparse_rebind(vk_sparse_section * s, size_t want) {
    vk_sparse_arena * a = s->arena;
    std::string why;
    // the max size if the address space holds every section at it (no more rebinding), else what it needs
    size_t range = a->on_demand ? want : s->section;
    vk_buffer nb = vk_sparse_range_new(a, range, nullptr);
    if (!nb && want < s->section) {
        range = want;
        nb = vk_sparse_range_new(a, range, &why);
    }
    if (!nb) {
        GGML_LOG_WARN("ggml_backend_vk_buffer_grow: a K/V section needs %zu MiB of address range: %s; it grows by copy\n", want >> 20, why.c_str());
        return false;
    }
    std::vector<size_t> bound;
    for (size_t i = 0; i < s->mem.size(); ++i) {
        if (s->mem[i] && std::find(s->pending.begin(), s->pending.end(), i) == s->pending.end()) {
            bound.push_back(i);
        }
    }
    a->device->device.waitIdle();
    if (!vk_sparse_bind(s, nb, bound, true)) {
        vk_sparse_range_release(a->device, range);
        return false;
    }
    if (vk_sparse_debug()) {
        GGML_LOG_INFO("vk_sparse_rebind: section of %zu bytes: range %zu -> %zu bytes, %zu chunks bound into it\n", s->section, s->range, range, bound.size());
    }
    vk_sparse_range_release(a->device, s->range);
    s->dev_buffer = nb; // the old range goes with its last reference (the descriptor caches see the destroy)
    s->range = range;
    return true;
}

// allocate the memory of the chunks up to size (bound and zeroed by the next commit); a larger range first if needed
static bool vk_sparse_map_to(vk_sparse_section * s, size_t size) {
    vk_sparse_arena * a = s->arena;
    const size_t want = std::min(s->section, a->chunk * ((size + a->chunk - 1) / a->chunk));
    if (want > s->range && !vk_sparse_rebind(s, want)) {
        return false;
    }
    std::lock_guard<std::mutex> lock(a->mtx);
    while (s->mapped < want) {
        const size_t i = s->mapped / a->chunk;
        try {
            vk::MemoryAllocateInfo mai(a->chunk, a->mem_type);
            vk::MemoryAllocateFlagsInfo flags_info;
            if (a->device->buffer_device_address) {
                flags_info.flags = vk::MemoryAllocateFlagBits::eDeviceAddress;
                mai.setPNext(&flags_info);
            }
            s->mem[i] = a->device->device.allocateMemory(mai);
        } catch (const vk::SystemError & e) {
            if (vk_sparse_debug()) {
                GGML_LOG_INFO("vk_sparse_map_to: chunk %zu: allocation failed: %s\n", i, e.what());
            }
            return false;
        }
        vk_sparse_stat_bound += (int64_t) a->chunk;
        if (s->pending.empty()) {
            a->pending.push_back(s);
        }
        s->pending.push_back(i);
        s->mapped += a->chunk;
    }
    return true;
}

// bind the pending chunks and zero them; nothing may access them before
static bool ggml_backend_vk_vmm_arena_commit(void * arena) {
    vk_sparse_arena * a = (vk_sparse_arena *) arena;
    std::vector<std::pair<vk_sparse_section *, std::vector<size_t>>> todo;
    {
        std::lock_guard<std::mutex> lock(a->mtx);
        for (vk_sparse_section * s : a->pending) {
            todo.emplace_back(s, std::move(s->pending));
            s->pending.clear();
        }
        a->pending.clear();
    }
    if (todo.empty()) {
        return true;
    }
    const int64_t t0 = ggml_time_us();
    size_t n = 0;
    for (auto & [s, idx] : todo) {
        if (!vk_sparse_bind(s, s->dev_buffer, idx, true)) {
            return false;
        }
        n += idx.size();
    }
    const int64_t t1 = ggml_time_us();
    for (auto & [s, idx] : todo) {
        for (size_t i : idx) {
            ggml_vk_buffer_memset(s->dev_buffer, i * a->chunk, 0, a->chunk);
        }
    }
    if (vk_sparse_debug()) {
        GGML_LOG_INFO("ggml_backend_vk_vmm_arena_commit: %zu chunks of %zu sections bound (%.2f ms) and zeroed (%.2f ms)\n", n, todo.size(),
            (t1 - t0) / 1000.0, (ggml_time_us() - t1) / 1000.0);
    }
    return true;
}

static ggml_backend_buffer_t ggml_backend_vk_vmm_arena_buffer(void * arena, size_t offset, size_t size, size_t max_size) {
    GGML_UNUSED(offset); // a range of its own per section
    vk_sparse_arena * a = (vk_sparse_arena *) arena;
    const size_t section = a->chunk * ((std::max(size, max_size) + a->chunk - 1) / a->chunk);
    const size_t now     = a->chunk * ((std::max<size_t>(size, 1) + a->chunk - 1) / a->chunk);
    // the range: the max size if the device's sparse address space allows it, else what the section needs now (it grows by
    // binding its memory into a larger range)
    size_t range = a->on_demand ? now : section;
    vk_buffer buf = vk_sparse_range_new(a, range, nullptr);
    if (!buf) {
        std::string why;
        range = now;
        buf = vk_sparse_range_new(a, range, &why);
        if (!buf) {
            GGML_LOG_WARN("ggml_backend_vk_vmm_arena_buffer: %s: no address range for a K/V section of %zu MiB: %s\n", a->device->name.c_str(), now >> 20, why.c_str());
            return nullptr;
        }
    }
    auto * btctx = (ggml_backend_vk_buffer_type_context *) a->buft->context;
    {
        std::lock_guard<std::mutex> lock(a->mtx);
        a->refs++;
    }
    auto * s = new vk_sparse_section(a->device, buf, btctx->name, a, section);
    s->range = range;
    s->mem.assign(section / a->chunk, nullptr);
    {
        std::lock_guard<std::mutex> lock(vk_sparse_reg_mtx);
        vk_sparse_live.insert(s);
    }
    if (!vk_sparse_map_to(s, std::max<size_t>(size, 1))) {
        ggml_backend_buffer_t tmp = ggml_backend_buffer_init(a->buft, ggml_backend_vk_sparse_interface, s, 0);
        ggml_backend_buffer_free(tmp);
        return nullptr;
    }
    s->logical = size;
    return ggml_backend_buffer_init(a->buft, ggml_backend_vk_sparse_interface, s, size);
}

static bool ggml_backend_vk_buffer_grow(ggml_backend_buffer_t buffer, size_t size) {
    vk_sparse_section * s = vk_sparse_ctx(buffer);
    if (!s) {
        return false;
    }
    if (size <= buffer->size) {
        return true;
    }
    if (size > s->section || !vk_sparse_map_to(s, size)) {
        return false;
    }
    buffer->size = size;
    s->logical   = size;
    return true;
}

static bool ggml_backend_vk_buffer_grow_commit(ggml_backend_buffer_t buffer) {
    vk_sparse_section * s = vk_sparse_ctx(buffer);
    return s && ggml_backend_vk_vmm_arena_commit(s->arena);
}

// unbind and free the chunks of a section from byte keep on (keep is rounded up to whole chunks)
static bool vk_sparse_unmap_from(vk_sparse_section * s, size_t keep_bytes) {
    vk_sparse_arena * a = s->arena;
    const size_t keep = a->chunk * ((keep_bytes + a->chunk - 1) / a->chunk);
    std::vector<size_t> bound;
    std::vector<size_t> all;
    {
        std::lock_guard<std::mutex> lock(a->mtx);
        for (size_t m = keep; m < s->mapped; m += a->chunk) {
            const size_t i = m / a->chunk;
            all.push_back(i);
            if (std::find(s->pending.begin(), s->pending.end(), i) == s->pending.end()) {
                bound.push_back(i); // pending chunks were never bound
            }
        }
    }
    if (all.empty()) {
        return true;
    }
    if (!bound.empty()) {
        a->device->device.waitIdle(); // no queue may use a chunk when it is unbound
        if (!vk_sparse_bind(s, s->dev_buffer, bound, false)) {
            return false;
        }
    }
    std::lock_guard<std::mutex> lock(a->mtx);
    for (size_t i : all) {
        s->pending.erase(std::remove(s->pending.begin(), s->pending.end(), i), s->pending.end());
        a->device->device.freeMemory(s->mem[i]);
        s->mem[i] = nullptr;
        vk_sparse_stat_bound -= (int64_t) a->chunk;
    }
    if (s->pending.empty()) {
        a->pending.erase(std::remove(a->pending.begin(), a->pending.end(), s), a->pending.end());
    }
    s->mapped = keep;
    return true;
}

static bool ggml_backend_vk_buffer_shrink(ggml_backend_buffer_t buffer, size_t size) {
    vk_sparse_section * s = vk_sparse_ctx(buffer);
    if (!s) {
        return false;
    }
    if (size >= buffer->size) {
        return true;
    }
    const size_t old = s->mapped;
    if (!vk_sparse_unmap_from(s, std::max<size_t>(size, 1))) {
        return false;
    }
    buffer->size = size;
    s->logical   = size;
    if (vk_sparse_debug()) {
        GGML_LOG_INFO("ggml_backend_vk_buffer_shrink: section of %zu bytes: %zu -> %zu bytes bound\n", s->section, old, s->mapped);
    }
    return true;
}

static size_t ggml_backend_vk_buffer_vmm_chunk(ggml_backend_buffer_t buffer) {
    vk_sparse_section * s = vk_sparse_ctx(buffer);
    return s ? s->arena->chunk : 0;
}

// vkFreeMemory gives the memory back at once: an in-place shrink frees it
static bool ggml_backend_vk_vmm_unmap_keeps_memory(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return false;
}

// stats[4]: live arenas (not tracked: 0), bound bytes, commit timings (0)
static void ggml_backend_vk_vmm_stats(int64_t * stats) {
    stats[0] = 0;
    stats[1] = vk_sparse_stat_bound;
    stats[2] = 0;
    stats[3] = 0;
}

static void ggml_backend_vk_sparse_free_buffer(ggml_backend_buffer_t buffer) {
    vk_sparse_section * s = (vk_sparse_section *) buffer->context;
    vk_sparse_unmap_from(s, 0);
    {
        std::lock_guard<std::mutex> lock(vk_sparse_reg_mtx);
        vk_sparse_live.erase(s);
    }
    vk_sparse_range_release(s->arena->device, s->range);
    vk_sparse_arena * a = s->arena;
    delete s;
    vk_sparse_arena_unref(a);
}

static void * ggml_backend_vk_sparse_get_base(ggml_backend_buffer_t buffer) {
    GGML_UNUSED(buffer);
    return vk_ptr_base; // a range of its own: the tensors start at offset 0
}

static void ggml_backend_vk_sparse_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    vk_sparse_section * s = (vk_sparse_section *) buffer->context;
    if (buffer->size > 0) {
        ggml_vk_buffer_memset(s->dev_buffer, 0, (uint32_t) value * 0x01010101, buffer->size);
    }
}

// device memory of the sparse K/V sections above their ggml buffer size (whole chunks), per device
size_t ggml_vk_sparse_unbooked(size_t device_idx) {
    std::lock_guard<std::mutex> lock(vk_sparse_reg_mtx);
    size_t res = 0;
    for (const vk_sparse_section * s : vk_sparse_live) {
        if (s->arena->device->idx == device_idx && s->mapped > s->logical) {
            res += s->mapped - s->logical;
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
// split); off by default, GGML_VK_ASYNC_READBACK=1 turns it on (measured neutral on an iGPU; to be measured on a discrete card)
static bool ggml_backend_vk_dev_async_readback(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    static const bool on = getenv("GGML_VK_ASYNC_READBACK") && atoi(getenv("GGML_VK_ASYNC_READBACK")) != 0;
    return on;
}

size_t ggml_vk_internal_device_bytes(size_t device_idx);

// device memory the backend holds outside the ggml buffers' sizes (its own graph buffers, the sparse chunk rounding): the
// zones booking adds it to the compute buffers, so the booked free memory follows the measured one
static size_t ggml_backend_vk_dev_internal_memory(ggml_backend_dev_t dev) {
    const auto * ctx = (const ggml_backend_vk_device_context *) dev->context;
    return ctx ? ggml_vk_internal_device_bytes((size_t) ctx->device) : 0;
}

void * ggml_vk_sparse_proc_address(const char * name) {
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
