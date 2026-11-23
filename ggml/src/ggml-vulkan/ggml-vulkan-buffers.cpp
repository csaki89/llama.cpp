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


// growable buffers on a sparse VkBuffer (dual-buffer fork, dynamic KV): the same interface as the ggml-cuda VMM arenas
// (looked up with get_proc_address). An arena is one sparse buffer sized for the max context: address range only; device
// memory is bound per chunk when a section grows (vkQueueBindSparse) and unbound and freed when it shrinks, so the memory
// comes back without a copy. Needs sparseBinding, sparseResidencyBuffer and a compute queue with sparse binding; else no
// arena (the KV grows by copy, with a log line naming the missing capability). GGML_VK_SPARSE=0 turns it off,
// GGML_VK_SPARSE_CHUNK_KIB sets the chunk (default 2 MiB), GGML_VK_SPARSE_DEBUG=1 logs every bind.

struct vk_sparse_arena {
    vk_device                       device;
    ggml_backend_buffer_type_t      buft;
    vk_buffer                       buf;      // the sparse buffer, no memory of its own
    vk::Queue                       queue;    // the compute queue (sparse binding)
    vk_queue_handle *               qh = nullptr;
    size_t                          reserved = 0;
    size_t                          chunk    = 0;
    uint32_t                        mem_type = 0;
    std::vector<vk::DeviceMemory>   mem;      // per chunk, null if unbound
    std::vector<size_t>             pending;  // chunks allocated, not bound and zeroed yet
    int                             refs = 1;
    std::mutex                      mtx;
};

struct vk_sparse_section : ggml_backend_vk_buffer_context {
    vk_sparse_arena * arena;
    size_t            offset;  // in the arena
    size_t            section; // max bytes
    size_t            mapped = 0;

    vk_sparse_section(vk_device_ref dev, vk_buffer buf, std::string & name, vk_sparse_arena * arena, size_t offset, size_t section) :
        ggml_backend_vk_buffer_context(dev, std::move(buf), name), arena(arena), offset(offset), section(section) {}
};

static bool vk_sparse_debug() {
    static const bool on = getenv("GGML_VK_SPARSE_DEBUG") && atoi(getenv("GGML_VK_SPARSE_DEBUG")) != 0;
    return on;
}

static std::atomic<int64_t> vk_sparse_stat_bound{0};

static void vk_sparse_arena_unref(vk_sparse_arena * a) {
    int left;
    {
        std::lock_guard<std::mutex> lock(a->mtx);
        left = --a->refs;
    }
    if (left > 0) {
        return;
    }
    a->device->device.waitIdle();
    for (auto & m : a->mem) {
        if (m) {
            a->device->device.freeMemory(m);
            vk_sparse_stat_bound -= (int64_t) a->chunk;
            m = nullptr;
        }
    }
    if (vk_sparse_debug()) {
        GGML_LOG_INFO("%s: arena of %zu bytes freed\n", __func__, a->reserved);
    }
    a->buf.reset();
    delete a;
}

// bind (with their memory) or unbind the chunks idx of an arena, and wait for it
static bool vk_sparse_bind(vk_sparse_arena * a, const std::vector<size_t> & idx, bool bind) {
    if (idx.empty()) {
        return true;
    }
    std::vector<vk::SparseMemoryBind> binds;
    binds.reserve(idx.size());
    for (size_t i : idx) {
        binds.push_back(vk::SparseMemoryBind(i * a->chunk, a->chunk, bind ? a->mem[i] : vk::DeviceMemory(), 0));
    }
    vk::SparseBufferMemoryBindInfo buf_bind(a->buf->buffer, (uint32_t) binds.size(), binds.data());
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
    // the chunk, from the device like the ggml-cuda VMM: the sparse block (the alignment), the unit the driver books an allocation with
    // (measured: the memory budget's heap usage before and after one block allocation, twice) and the allocation count limit (the
    // chunks of one KV arena use at most a quarter of maxMemoryAllocationCount); GGML_VK_SPARSE_CHUNK_KIB overrides it (test aid)
    std::string unit_note;
    const size_t unit = vk_sparse_booking_unit(device, (uint32_t) probe_type, block, unit_note);
    size_t chunk = block;
    if (unit > 0 && std::lcm(block, unit) <= 32*block) {
        chunk = std::lcm(block, unit);
    }
    // a chunk the driver gives back when it is freed (the shrink must return the memory)
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
            GGML_LOG_INFO("ggml_backend_vk_vmm_arena_new: %s: sparse block %zu; driver booking unit %zu%s; release unit %zu%s; allocation limit %u (chunk >= %zu for this arena); chunk %zu%s\n",
                device->name.c_str(), block, unit, unit_note.c_str(), release, release_note.c_str(), alloc_limit, count_min, chunk, env_chunk ? " (GGML_VK_SPARSE_CHUNK_KIB)" : "");
        }
    }
    const size_t reserved = chunk * ((std::max<size_t>(size, 1) + chunk - 1) / chunk);
    if (reserved > device->physical_device.getProperties().limits.sparseAddressSpaceSize) {
        return no("the KV (" + std::to_string(reserved) + " bytes) is larger than sparseAddressSpaceSize");
    }
    bci.size = reserved;
    vk::Buffer vkbuf;
    try {
        vkbuf = device->device.createBuffer(bci);
    } catch (const vk::SystemError & e) {
        return no(std::string("creating a sparse buffer of ") + std::to_string(reserved) + " bytes failed: " + e.what());
    }
    req = device->device.getBufferMemoryRequirements(vkbuf);
    const vk::PhysicalDeviceMemoryProperties mp = device->physical_device.getMemoryProperties();
    int mem_type = -1;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((req.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal)) {
            mem_type = (int) i;
            break;
        }
    }
    if (mem_type < 0) {
        device->device.destroyBuffer(vkbuf);
        return no("no device local memory type for a sparse buffer");
    }

    auto * a = new vk_sparse_arena;
    a->device   = device;
    a->buft     = buft;
    a->queue    = device->compute_queue->handle->queue;
    a->qh       = device->compute_queue->handle.get();
    a->reserved = reserved;
    a->chunk    = chunk;
    a->mem_type = (uint32_t) mem_type;
    a->mem.assign(reserved / chunk, nullptr);
    a->buf = std::make_shared<vk_buffer_struct>();
    a->buf->buffer = vkbuf;
    a->buf->memory_property_flags = vk::MemoryPropertyFlagBits::eDeviceLocal;
    a->buf->ptr    = nullptr;
    a->buf->size   = reserved;
    a->buf->device = device;
    if (device->buffer_device_address) {
        a->buf->bda_addr = device->device.getBufferAddress(vk::BufferDeviceAddressInfo(vkbuf));
    }
    {
        std::lock_guard<std::mutex> lock(log_mtx);
        if (logged.insert(device->name + "/ok").second) {
            GGML_LOG_INFO("ggml_backend_vk_vmm_arena_new: %s: sparse KV buffers: block %zu, chunk %zu bytes, memory type %d\n", device->name.c_str(),
                (size_t) req.alignment, chunk, mem_type);
        }
    }
    if (vk_sparse_debug()) {
        GGML_LOG_INFO("ggml_backend_vk_vmm_arena_new: arena of %zu bytes (%zu chunks)\n", reserved, reserved / chunk);
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

// allocate the memory of the chunks up to size (bound and zeroed by the next commit)
static bool vk_sparse_map_to(vk_sparse_section * s, size_t size) {
    vk_sparse_arena * a = s->arena;
    const size_t want = std::min(s->section, a->chunk * ((size + a->chunk - 1) / a->chunk));
    std::lock_guard<std::mutex> lock(a->mtx);
    while (s->mapped < want) {
        const size_t i = (s->offset + s->mapped) / a->chunk;
        try {
            vk::MemoryAllocateInfo mai(a->chunk, a->mem_type);
            vk::MemoryAllocateFlagsInfo flags_info;
            if (a->device->buffer_device_address) {
                flags_info.flags = vk::MemoryAllocateFlagBits::eDeviceAddress;
                mai.setPNext(&flags_info);
            }
            a->mem[i] = a->device->device.allocateMemory(mai);
        } catch (const vk::SystemError & e) {
            if (vk_sparse_debug()) {
                GGML_LOG_INFO("vk_sparse_map_to: chunk %zu: allocation failed: %s\n", i, e.what());
            }
            return false;
        }
        vk_sparse_stat_bound += (int64_t) a->chunk;
        a->pending.push_back(i);
        s->mapped += a->chunk;
    }
    return true;
}

// bind the pending chunks and zero them; nothing may access them before
static bool ggml_backend_vk_vmm_arena_commit(void * arena) {
    vk_sparse_arena * a = (vk_sparse_arena *) arena;
    std::vector<size_t> idx;
    {
        std::lock_guard<std::mutex> lock(a->mtx);
        idx.swap(a->pending);
    }
    if (idx.empty()) {
        return true;
    }
    const int64_t t0 = ggml_time_us();
    if (!vk_sparse_bind(a, idx, true)) {
        return false;
    }
    const int64_t t1 = ggml_time_us();
    for (size_t i : idx) {
        ggml_vk_buffer_memset(a->buf, i * a->chunk, 0, a->chunk);
    }
    if (vk_sparse_debug()) {
        GGML_LOG_INFO("ggml_backend_vk_vmm_arena_commit: %zu chunks bound (%.2f ms) and zeroed (%.2f ms)\n", idx.size(), (t1 - t0) / 1000.0, (ggml_time_us() - t1) / 1000.0);
    }
    return true;
}

static ggml_backend_buffer_t ggml_backend_vk_vmm_arena_buffer(void * arena, size_t offset, size_t size, size_t max_size) {
    vk_sparse_arena * a = (vk_sparse_arena *) arena;
    const size_t section = a->chunk * ((std::max(size, max_size) + a->chunk - 1) / a->chunk);
    if (offset % a->chunk != 0 || offset + section > a->reserved) {
        return nullptr;
    }
    auto * btctx = (ggml_backend_vk_buffer_type_context *) a->buft->context;
    {
        std::lock_guard<std::mutex> lock(a->mtx);
        a->refs++;
    }
    auto * s = new vk_sparse_section(a->device, a->buf, btctx->name, a, offset, section);
    if (!vk_sparse_map_to(s, std::max<size_t>(size, 1))) {
        ggml_backend_buffer_t tmp = ggml_backend_buffer_init(a->buft, ggml_backend_vk_sparse_interface, s, 0);
        ggml_backend_buffer_free(tmp);
        return nullptr;
    }
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
            const size_t i = (s->offset + m) / a->chunk;
            all.push_back(i);
            if (std::find(a->pending.begin(), a->pending.end(), i) == a->pending.end()) {
                bound.push_back(i); // pending chunks were never bound
            }
        }
    }
    if (all.empty()) {
        return true;
    }
    if (!bound.empty()) {
        a->device->device.waitIdle(); // no queue may use a chunk when it is unbound
        if (!vk_sparse_bind(a, bound, false)) {
            return false;
        }
    }
    std::lock_guard<std::mutex> lock(a->mtx);
    for (size_t i : all) {
        a->pending.erase(std::remove(a->pending.begin(), a->pending.end(), i), a->pending.end());
        a->device->device.freeMemory(a->mem[i]);
        a->mem[i] = nullptr;
        vk_sparse_stat_bound -= (int64_t) a->chunk;
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
    if (vk_sparse_debug()) {
        GGML_LOG_INFO("ggml_backend_vk_buffer_shrink: section at %zu: %zu -> %zu bytes bound\n", s->offset, old, s->mapped);
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
    vk_sparse_arena * a = s->arena;
    delete s;
    vk_sparse_arena_unref(a);
}

static void * ggml_backend_vk_sparse_get_base(ggml_backend_buffer_t buffer) {
    return (uint8_t *) vk_ptr_base + ((vk_sparse_section *) buffer->context)->offset;
}

static void ggml_backend_vk_sparse_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    vk_sparse_section * s = (vk_sparse_section *) buffer->context;
    if (buffer->size > 0) {
        ggml_vk_buffer_memset(s->dev_buffer, s->offset, (uint32_t) value * 0x01010101, buffer->size);
    }
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

void * ggml_vk_sparse_proc_address(const char * name) {
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
