// dual-buffer fork: the compute power state of an AMD GPU for the Vulkan backend.
//
// amdgpu switches a GPU to its compute power profile (and keeps GFXOFF off) while a KFD process has a user-mode queue on it:
// the first queue of a process on a device registers the process with the device queue manager (register_process ->
// kfd_inc_compute_active -> amdgpu_amdkfd_set_compute_idle(false)), the last one destroyed unregisters it. A ROCm process gets
// this from the queues its runtime creates; Vulkan work (amdgpu CS) does not, and in a GPU/CPU split generation the GPU then
// stays at its lowest clock (the GPU part of every split runs several times slower).
//
// While a Vulkan backend lives on an AMD device that has a KFD node, this creates one empty compute queue on it through the
// KFD interface itself (the stable uapi of /dev/kfd, no ROCm runtime: it works for any GPU the KFD supports, also one the ROCm
// runtime does not), the same mechanism a ROCm process triggers; nothing is written to sysfs. The queue holds no work. It is
// destroyed when the last backend on the device is freed (and by the kernel at process exit).
// GGML_VK_KFD_COMPUTE_PROFILE=0 turns it off.

#include "ggml-vulkan-common.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#if defined(__linux__)
#include <dirent.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {

struct kfd_bo {
    void *   cpu    = nullptr;
    size_t   size   = 0;
    uint64_t handle = 0;
    bool     mapped = false;
};

struct kfd_dev_state {
    int                 refs      = 0;
    bool                active    = false;
    bool                logged    = false; // the reason for no compute state was logged
    bool                vm        = false; // ACQUIRE_VM done (kept for the process lifetime: the KFD process lives with the mm)
    int                 drm_fd    = -1;
    uint32_t            gpu_id    = 0;
    uint32_t            queue_id  = 0;
    uint64_t            va_base   = 0;
    uint64_t            va_limit  = 0;
    std::vector<kfd_bo> bos;
    std::string         where;             // "gpu_id N, PCI ..., renderDN" for the log line
};

std::mutex                        g_mutex;
std::map<size_t, kfd_dev_state>   g_devs;

bool kfd_disabled() {
    const char * e = getenv("GGML_VK_KFD_COMPUTE_PROFILE");
    return e != nullptr && atoi(e) == 0;
}

#if defined(__linux__)

// the stable KFD uapi (include/uapi/linux/kfd_ioctl.h), only what is used here; the build host's headers may be older
struct kfd_get_version_args { uint32_t major_version, minor_version; };
struct kfd_create_queue_args {
    uint64_t ring_base_address, write_pointer_address, read_pointer_address, doorbell_offset;
    uint32_t ring_size, gpu_id, queue_type, queue_percentage, queue_priority, queue_id;
    uint64_t eop_buffer_address, eop_buffer_size, ctx_save_restore_address;
    uint32_t ctx_save_restore_size, ctl_stack_size;
};
struct kfd_destroy_queue_args { uint32_t queue_id, pad; };
struct kfd_process_device_apertures {
    uint64_t lds_base, lds_limit, scratch_base, scratch_limit, gpuvm_base, gpuvm_limit;
    uint32_t gpu_id, pad;
};
struct kfd_get_process_apertures_new_args { uint64_t kfd_process_device_apertures_ptr; uint32_t num_of_nodes, pad; };
struct kfd_acquire_vm_args { uint32_t drm_fd, gpu_id; };
struct kfd_alloc_memory_of_gpu_args { uint64_t va_addr, size, handle, mmap_offset; uint32_t gpu_id, flags; };
struct kfd_free_memory_of_gpu_args { uint64_t handle; };
struct kfd_map_memory_to_gpu_args { uint64_t handle, device_ids_array_ptr; uint32_t n_devices, n_success; };

#define GGML_KFD_IOR(nr, type)  _IOR('K', nr, type)
#define GGML_KFD_IOW(nr, type)  _IOW('K', nr, type)
#define GGML_KFD_IOWR(nr, type) _IOWR('K', nr, type)
#define GGML_KFD_IOC_GET_VERSION                 GGML_KFD_IOR(0x01, kfd_get_version_args)
#define GGML_KFD_IOC_CREATE_QUEUE                GGML_KFD_IOWR(0x02, kfd_create_queue_args)
#define GGML_KFD_IOC_DESTROY_QUEUE               GGML_KFD_IOWR(0x03, kfd_destroy_queue_args)
#define GGML_KFD_IOC_GET_PROCESS_APERTURES_NEW   GGML_KFD_IOWR(0x14, kfd_get_process_apertures_new_args)
#define GGML_KFD_IOC_ACQUIRE_VM                  GGML_KFD_IOW(0x15, kfd_acquire_vm_args)
#define GGML_KFD_IOC_ALLOC_MEMORY_OF_GPU         GGML_KFD_IOWR(0x16, kfd_alloc_memory_of_gpu_args)
#define GGML_KFD_IOC_FREE_MEMORY_OF_GPU          GGML_KFD_IOW(0x17, kfd_free_memory_of_gpu_args)
#define GGML_KFD_IOC_MAP_MEMORY_TO_GPU           GGML_KFD_IOWR(0x18, kfd_map_memory_to_gpu_args)
#define GGML_KFD_IOC_UNMAP_MEMORY_FROM_GPU       GGML_KFD_IOWR(0x19, kfd_map_memory_to_gpu_args)

constexpr uint32_t KFD_ALLOC_USERPTR    = 1u << 2;
constexpr uint32_t KFD_ALLOC_WRITABLE   = 1u << 31;
constexpr uint32_t KFD_ALLOC_EXECUTABLE = 1u << 30;
constexpr uint32_t KFD_QUEUE_TYPE_COMPUTE_PM4 = 0;
constexpr size_t   KFD_PAGE = 4096;

#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0x100000
#endif

int g_kfd_fd = -1; // kept open: the KFD process (and the VMs it acquired) lives as long as the mm

int kfd_ioctl(unsigned long req, void * arg) {
    int r;
    do {
        r = ioctl(g_kfd_fd, req, arg);
    } while (r == -1 && (errno == EINTR || errno == EAGAIN));
    return r;
}

bool read_file(const std::string & path, std::string & out) {
    FILE * f = fopen(path.c_str(), "r");
    if (!f) {
        return false;
    }
    char buf[4096];
    out.clear();
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
        out.append(buf, n);
    }
    fclose(f);
    return true;
}

// a KFD topology node: its gpu_id and its "key value" properties (read only)
struct kfd_node {
    uint32_t                        gpu_id = 0;
    std::map<std::string, uint64_t> props;
    uint64_t prop(const char * k, uint64_t def = 0) const {
        auto it = props.find(k);
        return it == props.end() ? def : it->second;
    }
};

bool kfd_topology(std::vector<kfd_node> & nodes) {
    const std::string root = "/sys/class/kfd/kfd/topology/nodes";
    DIR * d = opendir(root.c_str());
    if (!d) {
        return false;
    }
    while (struct dirent * e = readdir(d)) {
        if (e->d_name[0] == '.') {
            continue;
        }
        kfd_node n;
        std::string s;
        if (!read_file(root + "/" + e->d_name + "/gpu_id", s)) {
            continue;
        }
        n.gpu_id = (uint32_t) strtoul(s.c_str(), nullptr, 10);
        if (n.gpu_id == 0 || !read_file(root + "/" + e->d_name + "/properties", s)) {
            continue; // a CPU node
        }
        size_t pos = 0;
        while (pos < s.size()) {
            size_t eol = s.find('\n', pos);
            if (eol == std::string::npos) {
                eol = s.size();
            }
            const std::string line = s.substr(pos, eol - pos);
            const size_t sp = line.find(' ');
            if (sp != std::string::npos) {
                n.props[line.substr(0, sp)] = strtoull(line.c_str() + sp + 1, nullptr, 10);
            }
            pos = eol + 1;
        }
        nodes.push_back(std::move(n));
    }
    closedir(d);
    return true;
}

void kfd_free_bo(kfd_dev_state & st, kfd_bo & bo) {
    if (bo.mapped) {
        uint32_t ids[1] = { st.gpu_id };
        kfd_map_memory_to_gpu_args a = { bo.handle, (uint64_t) (uintptr_t) ids, 1, 0 };
        kfd_ioctl(GGML_KFD_IOC_UNMAP_MEMORY_FROM_GPU, &a);
    }
    if (bo.handle) {
        kfd_free_memory_of_gpu_args a = { bo.handle };
        kfd_ioctl(GGML_KFD_IOC_FREE_MEMORY_OF_GPU, &a);
    }
    if (bo.cpu) {
        munmap(bo.cpu, bo.size);
    }
    bo = kfd_bo{};
}

// a page-aligned system-memory buffer mapped to the GPU at the same address (userptr), inside the process' GPUVM aperture
bool kfd_alloc_bo(kfd_dev_state & st, size_t size, kfd_bo & bo, std::string & why) {
    size = (size + KFD_PAGE - 1) / KFD_PAGE * KFD_PAGE;
    void * p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    auto inside = [&](void * q) {
        return q != MAP_FAILED && (uint64_t) (uintptr_t) q >= st.va_base && (uint64_t) (uintptr_t) q + size - 1 <= st.va_limit;
    };
    if (p != MAP_FAILED && !inside(p)) {
        munmap(p, size);
        p = MAP_FAILED;
        // a small GPUVM aperture (older GPUs): an address inside it, from its middle down
        for (uint64_t hint = (st.va_base + (st.va_limit - st.va_base) / 2) & ~(uint64_t) 0xfffff;
             hint > st.va_base && hint + size - 1 <= st.va_limit; hint -= (uint64_t) 1 << 30) {
            void * q = mmap((void *) (uintptr_t) hint, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
            if (q == MAP_FAILED) {
                continue;
            }
            if (inside(q)) {
                p = q;
                break;
            }
            munmap(q, size);
        }
    }
    if (p == MAP_FAILED) {
        why = "no system memory address inside the GPUVM aperture";
        return false;
    }
    memset(p, 0, size);
    bo.cpu  = p;
    bo.size = size;
    kfd_alloc_memory_of_gpu_args a = {};
    a.va_addr     = (uint64_t) (uintptr_t) p;
    a.size        = size;
    a.mmap_offset = (uint64_t) (uintptr_t) p;
    a.gpu_id      = st.gpu_id;
    a.flags       = KFD_ALLOC_USERPTR | KFD_ALLOC_WRITABLE | KFD_ALLOC_EXECUTABLE;
    if (kfd_ioctl(GGML_KFD_IOC_ALLOC_MEMORY_OF_GPU, &a) != 0) {
        why = std::string("KFD memory allocation failed: ") + strerror(errno);
        kfd_free_bo(st, bo);
        return false;
    }
    bo.handle = a.handle;
    uint32_t ids[1] = { st.gpu_id };
    kfd_map_memory_to_gpu_args m = { bo.handle, (uint64_t) (uintptr_t) ids, 1, 0 };
    if (kfd_ioctl(GGML_KFD_IOC_MAP_MEMORY_TO_GPU, &m) != 0) {
        why = std::string("KFD GPU mapping failed: ") + strerror(errno);
        kfd_free_bo(st, bo);
        return false;
    }
    bo.mapped = true;
    return true;
}

// the context save area and EOP buffer sizes the KFD checks a compute queue against (the kernel's
// kfd_queue_ctx_save_restore_size, from the topology properties); a few variants for other kernel versions, tried in order
struct kfd_queue_sizes {
    uint32_t ctl_stack = 0, cwsr = 0, total_cwsr = 0, eop = 0;
    bool operator==(const kfd_queue_sizes & o) const {
        return ctl_stack == o.ctl_stack && cwsr == o.cwsr && total_cwsr == o.total_cwsr && eop == o.eop;
    }
};

std::vector<kfd_queue_sizes> kfd_size_candidates(const kfd_node & n) {
    std::vector<kfd_queue_sizes> out;
    auto add = [&](const kfd_queue_sizes & s) {
        for (const auto & o : out) {
            if (o == s) {
                return;
            }
        }
        out.push_back(s);
    };
    const uint32_t gfxv = (uint32_t) n.prop("gfx_target_version");
    if (gfxv < 80001) {
        add({}); // no CWSR, no EOP buffer
        return out;
    }
    auto align = [](uint64_t v, uint64_t a) { return (v + a - 1) / a * a; };
    const uint64_t xcc       = std::max<uint64_t>(1, n.prop("num_xcc", 1));
    const uint64_t simd_cu   = std::max<uint64_t>(1, n.prop("simd_per_cu", 4));
    const uint64_t cu_num    = n.prop("simd_count") / simd_cu / xcc;
    const uint64_t arrays    = n.prop("array_count");
    const uint64_t arr_eng   = std::max<uint64_t>(1, n.prop("simd_arrays_per_engine", 1));
    const uint64_t wave_old  = std::min<uint64_t>(cu_num * 40, arrays / arr_eng * 512);
    const uint64_t wave_fml  = gfxv < 100100 ? wave_old : cu_num * 32;
    uint64_t vgpr_fml = 0x40000;
    if (gfxv / 100 * 100 == 90400 || gfxv == 90010 || gfxv == 90008 || gfxv == 90500) {
        vgpr_fml = 0x80000;
    } else if (gfxv == 110000 || gfxv == 110001 || gfxv == 120000 || gfxv == 120001) {
        vgpr_fml = 0x60000;
    }
    const uint64_t lds   = gfxv == 90500 ? (n.prop("lds_size_in_kb", 64) << 10) : 0x10000;
    const uint32_t eop_f = gfxv == 80002 ? 0x8000 : 4096;
    const uint64_t waves[] = { wave_fml, cu_num * 32, cu_num * 40, wave_old };
    const uint64_t vgprs[] = { vgpr_fml, 0x40000, 0x60000, 0x80000 };
    for (int pass = 0; pass < 2; pass++) {
        for (uint64_t wave : waves) {
            for (uint64_t vgpr : vgprs) {
                if (pass == 0 && (wave != wave_fml || vgpr != vgpr_fml)) {
                    continue; // the formula first
                }
                const uint64_t bpw = gfxv >= 100100 ? 12 : 8;
                uint64_t ctl = align(40 + wave * bpw + 8, KFD_PAGE);
                if (gfxv / 10000 * 10000 == 100000) {
                    ctl = std::min<uint64_t>(ctl, 0x7000);
                }
                const uint64_t wg    = align(cu_num * (vgpr + 0x4000 + lds + 0x1000), KFD_PAGE);
                const uint64_t cwsr  = ctl + wg;
                const uint64_t debug = align(wave * 32, 64);
                for (uint32_t eop : { eop_f, 4096u, 0x8000u }) {
                    add({ (uint32_t) ctl, (uint32_t) cwsr, (uint32_t) align((cwsr + debug) * xcc, KFD_PAGE), eop });
                    add({ (uint32_t) ctl, (uint32_t) cwsr, (uint32_t) align(cwsr * xcc, KFD_PAGE), eop });
                }
            }
        }
    }
    add({}); // a kernel that does not want a context save area
    return out;
}

void kfd_stop(kfd_dev_state & st) {
    if (st.active) {
        kfd_destroy_queue_args a = { st.queue_id, 0 };
        kfd_ioctl(GGML_KFD_IOC_DESTROY_QUEUE, &a);
    }
    for (auto & bo : st.bos) {
        kfd_free_bo(st, bo);
    }
    st.bos.clear();
    st.queue_id = 0;
    st.active   = false;
}

// the empty compute queue on the device's KFD node; false with the reason
bool kfd_start(const vk_device & device, kfd_dev_state & st, std::string & why) {
    // the device's PCI location
    bool has_pci = false;
    for (const auto & e : device->physical_device.enumerateDeviceExtensionProperties()) {
        if (strcmp(e.extensionName, "VK_EXT_pci_bus_info") == 0) {
            has_pci = true;
            break;
        }
    }
    vk::PhysicalDevicePCIBusInfoPropertiesEXT pci = {};
    vk::PhysicalDeviceProperties2 props2 = {};
    if (has_pci) {
        props2.pNext = &pci;
    }
    device->physical_device.getProperties2(&props2);

    std::vector<kfd_node> nodes;
    if (!kfd_topology(nodes)) {
        why = "no KFD topology (/sys/class/kfd): the amdgpu KFD is not loaded or not visible here";
        return false;
    }
    const kfd_node * node = nullptr;
    int n_same_id = 0;
    for (const auto & n : nodes) {
        if (has_pci) {
            const uint64_t loc = ((uint64_t) pci.pciBus << 8) | ((uint64_t) pci.pciDevice << 3) | pci.pciFunction;
            if (n.prop("location_id", UINT64_MAX) == loc && n.prop("domain", 0) == pci.pciDomain) {
                node = &n;
                break;
            }
        } else if (n.prop("vendor_id") == props2.properties.vendorID && n.prop("device_id") == props2.properties.deviceID) {
            node = &n;
            n_same_id++;
        }
    }
    if (!has_pci && n_same_id > 1) {
        why = "no PCI bus info (VK_EXT_pci_bus_info) and several KFD nodes with the same device id";
        return false;
    }
    if (!node) {
        why = "the device has no KFD node (the KFD does not support it here, e.g. no PCIe atomics)";
        return false;
    }
    st.gpu_id = node->gpu_id;
    char where[160];
    snprintf(where, sizeof(where), "gpu_id %u, PCI %04x:%02x:%02x.%x, renderD%u, gfx_target_version %u", st.gpu_id,
        has_pci ? pci.pciDomain : 0, has_pci ? pci.pciBus : 0, has_pci ? pci.pciDevice : 0, has_pci ? pci.pciFunction : 0,
        (unsigned) node->prop("drm_render_minor"), (unsigned) node->prop("gfx_target_version"));
    st.where = where;

    if (g_kfd_fd < 0) {
        g_kfd_fd = open("/dev/kfd", O_RDWR | O_CLOEXEC);
        if (g_kfd_fd < 0) {
            why = std::string("cannot open /dev/kfd: ") + strerror(errno) + " (not passed to the container, or no access)";
            return false;
        }
        kfd_get_version_args v = {};
        if (kfd_ioctl(GGML_KFD_IOC_GET_VERSION, &v) != 0 || v.major_version != 1) {
            why = "unsupported KFD interface version";
            close(g_kfd_fd);
            g_kfd_fd = -1;
            return false;
        }
    }
    if (!st.vm) {
        if (st.drm_fd < 0) {
            const std::string render = "/dev/dri/renderD" + std::to_string(node->prop("drm_render_minor"));
            st.drm_fd = open(render.c_str(), O_RDWR | O_CLOEXEC);
            if (st.drm_fd < 0) {
                why = "cannot open " + render + ": " + strerror(errno);
                return false;
            }
        }
        kfd_acquire_vm_args a = { (uint32_t) st.drm_fd, st.gpu_id };
        if (kfd_ioctl(GGML_KFD_IOC_ACQUIRE_VM, &a) != 0) {
            why = std::string("KFD ACQUIRE_VM failed: ") + strerror(errno) +
                  (errno == EBUSY ? " (another runtime in this process holds the GPU's KFD VM: it has the compute state already)" : "");
            return false;
        }
        st.vm = true;
    }
    if (st.va_limit == 0) {
        kfd_get_process_apertures_new_args q = {};
        if (kfd_ioctl(GGML_KFD_IOC_GET_PROCESS_APERTURES_NEW, &q) != 0 || q.num_of_nodes == 0) {
            why = std::string("KFD apertures query failed: ") + strerror(errno);
            return false;
        }
        std::vector<kfd_process_device_apertures> ap(q.num_of_nodes);
        q.kfd_process_device_apertures_ptr = (uint64_t) (uintptr_t) ap.data();
        if (kfd_ioctl(GGML_KFD_IOC_GET_PROCESS_APERTURES_NEW, &q) != 0) {
            why = std::string("KFD apertures query failed: ") + strerror(errno);
            return false;
        }
        for (uint32_t i = 0; i < q.num_of_nodes && i < ap.size(); i++) {
            if (ap[i].gpu_id == st.gpu_id) {
                st.va_base  = ap[i].gpuvm_base;
                st.va_limit = ap[i].gpuvm_limit;
            }
        }
        if (st.va_limit == 0) {
            why = "no GPUVM aperture for the device";
            return false;
        }
    }

    // ring, read and write pointers (one page each), then the EOP buffer and the context save area per size candidate
    st.bos.resize(5);
    if (!kfd_alloc_bo(st, KFD_PAGE, st.bos[0], why) || !kfd_alloc_bo(st, KFD_PAGE, st.bos[1], why) ||
        !kfd_alloc_bo(st, KFD_PAGE, st.bos[2], why)) {
        kfd_stop(st);
        return false;
    }
    int last_errno = 0;
    for (const kfd_queue_sizes & s : kfd_size_candidates(*node)) {
        kfd_free_bo(st, st.bos[3]);
        kfd_free_bo(st, st.bos[4]);
        if ((s.eop && !kfd_alloc_bo(st, s.eop, st.bos[3], why)) || (s.total_cwsr && !kfd_alloc_bo(st, s.total_cwsr, st.bos[4], why))) {
            kfd_stop(st);
            return false;
        }
        kfd_create_queue_args c = {};
        c.ring_base_address        = (uint64_t) (uintptr_t) st.bos[0].cpu;
        c.ring_size                = (uint32_t) KFD_PAGE;
        c.write_pointer_address    = (uint64_t) (uintptr_t) st.bos[1].cpu;
        c.read_pointer_address     = (uint64_t) (uintptr_t) st.bos[2].cpu;
        c.gpu_id                   = st.gpu_id;
        c.queue_type               = KFD_QUEUE_TYPE_COMPUTE_PM4;
        c.queue_percentage         = 100;
        c.queue_priority           = 7;
        c.eop_buffer_address       = s.eop ? (uint64_t) (uintptr_t) st.bos[3].cpu : 0;
        c.eop_buffer_size          = s.eop;
        c.ctx_save_restore_address = s.total_cwsr ? (uint64_t) (uintptr_t) st.bos[4].cpu : 0;
        c.ctx_save_restore_size    = s.cwsr;
        c.ctl_stack_size           = s.ctl_stack;
        if (kfd_ioctl(GGML_KFD_IOC_CREATE_QUEUE, &c) == 0) {
            st.queue_id = c.queue_id;
            st.active   = true;
            return true;
        }
        last_errno = errno;
        if (last_errno != EINVAL) {
            break; // not a size mismatch
        }
    }
    why = std::string("KFD CREATE_QUEUE failed: ") + strerror(last_errno);
    kfd_stop(st);
    return false;
}

#endif // __linux__

} // namespace

// a backend on the device starts: the first one takes the compute power state
void ggml_vk_kfd_compute_acquire(vk_device & device) {
    std::lock_guard<std::mutex> lock(g_mutex);
    kfd_dev_state & st = g_devs[device->idx];
    if (++st.refs > 1) {
        return;
    }
    std::string why;
    if (kfd_disabled()) {
        why = "turned off (GGML_VK_KFD_COMPUTE_PROFILE=0)";
    } else if (device->vendor_id != 0x1002) {
        why = "not an AMD device";
    } else {
#if defined(__linux__)
        if (kfd_start(device, st, why)) {
            GGML_LOG_INFO("ggml_vulkan: %s: compute power state active: an empty KFD compute queue on the device (%s), as a ROCm process has; "
                "GGML_VK_KFD_COMPUTE_PROFILE=0 turns it off\n", device->name.c_str(), st.where.c_str());
            return;
        }
#else
        why = "no KFD on this operating system";
#endif
    }
    if (!st.logged) {
        st.logged = true;
        GGML_LOG_WARN("ggml_vulkan: %s: no compute power state: %s; the driver may keep the GPU at low clocks while the CPU computes "
            "part of each token, which slows a GPU/CPU split generation (an AMD GPU needs /dev/kfd in the container)\n",
            device->name.c_str(), why.c_str());
    }
}

// a backend on the device is freed: the last one gives the compute power state back
void ggml_vk_kfd_compute_release(vk_device & device) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_devs.find(device->idx);
    if (it == g_devs.end() || it->second.refs == 0) {
        return;
    }
    kfd_dev_state & st = it->second;
    if (--st.refs > 0) {
        return;
    }
#if defined(__linux__)
    if (st.active) {
        kfd_stop(st);
        GGML_LOG_INFO("ggml_vulkan: %s: compute power state released (KFD compute queue destroyed)\n", device->name.c_str());
    }
#endif
}
