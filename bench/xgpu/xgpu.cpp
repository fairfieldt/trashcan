// xgpu: cross-GPU sync latency and copy bandwidth on two Vulkan devices.
// Written for incite (2x FirePro D700 on RADV) to decide how a Vulkan tensor-parallel allreduce should
// synchronize and move data. See README.md for what each test models.
//
// usage: xgpu [test ...]   tests: fence relay proxy xsem spin copy p2p p2p-verify p2p-wall host-write   (default: all)

#include <vulkan/vulkan.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

using clk = std::chrono::steady_clock;

static double us_since(clk::time_point t0) {
    return std::chrono::duration<double, std::micro>(clk::now() - t0).count();
}

struct stats {
    double p10, p50, p90, mean;
};

static stats summarize(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    double sum = 0;
    for (double x : v) {
        sum += x;
    }
    auto at = [&](double q) { return v[std::min(v.size() - 1, (size_t) (q * v.size()))]; };
    return { at(0.1), at(0.5), at(0.9), sum / v.size() };
}

static void print_stats(const char * what, const stats & s) {
    printf("  %-44s p10 %8.1f  p50 %8.1f  p90 %8.1f  mean %8.1f us\n", what, s.p10, s.p50, s.p90, s.mean);
}

static void check(VkResult r, const char * what) {
    if (r != VK_SUCCESS) {
        fprintf(stderr, "%s failed: %d\n", what, (int) r);
        exit(1);
    }
}

struct buffer {
    vk::Buffer       buf;
    vk::DeviceMemory mem;
    void *           ptr    = nullptr;
    size_t           size   = 0;
    bool             mapped = false;
};

struct gpu {
    vk::PhysicalDevice                 pd;
    vk::Device                         dev;
    uint32_t                           qfam = 0;
    vk::Queue                          q;
    vk::CommandPool                    pool;
    vk::PhysicalDeviceMemoryProperties memprops;
    float                              ts_period = 1.0f;
    std::string                        name;

    PFN_vkGetMemoryHostPointerPropertiesEXT get_host_ptr_props = nullptr;
    PFN_vkGetSemaphoreFdKHR                 get_sem_fd         = nullptr;
    PFN_vkImportSemaphoreFdKHR              import_sem_fd      = nullptr;
    PFN_vkGetMemoryFdKHR                    get_mem_fd         = nullptr;
    PFN_vkGetMemoryFdPropertiesKHR          get_mem_fd_props   = nullptr;

    // spin pipeline
    vk::ShaderModule        spin_module[4];
    vk::DescriptorSetLayout dsl;
    vk::PipelineLayout      pl;
    vk::Pipeline            spin_pipeline[4];
    vk::ShaderModule        store_module[2];
    vk::Pipeline            store_pipeline[2];
    vk::DescriptorPool      dpool;
};

static const vk::BufferUsageFlags k_usage =
    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;

static uint32_t find_mem_type(const gpu & g, uint32_t bits, vk::MemoryPropertyFlags want, vk::MemoryPropertyFlags avoid = {}) {
    for (uint32_t i = 0; i < g.memprops.memoryTypeCount; i++) {
        const auto flags = g.memprops.memoryTypes[i].propertyFlags;
        if ((bits & (1u << i)) && (flags & want) == want && !(flags & avoid)) {
            return i;
        }
    }
    fprintf(stderr, "no memory type for bits 0x%x want 0x%x\n", bits, (uint32_t) want);
    exit(1);
}

static std::string mem_type_str(const gpu & g, uint32_t idx) {
    const auto f = g.memprops.memoryTypes[idx].propertyFlags;
    std::string s = "type " + std::to_string(idx) + " heap " + std::to_string(g.memprops.memoryTypes[idx].heapIndex) + " [";
    if (f & vk::MemoryPropertyFlagBits::eDeviceLocal)  s += " DEVICE_LOCAL";
    if (f & vk::MemoryPropertyFlagBits::eHostVisible)  s += " HOST_VISIBLE";
    if (f & vk::MemoryPropertyFlagBits::eHostCoherent) s += " HOST_COHERENT";
    if (f & vk::MemoryPropertyFlagBits::eHostCached)   s += " HOST_CACHED";
    return s + " ]";
}

static gpu make_gpu(vk::Instance inst, vk::PhysicalDevice pd) {
    gpu g;
    g.pd = pd;

    vk::PhysicalDevicePCIBusInfoPropertiesEXT pci;
    vk::PhysicalDeviceProperties2             props2;
    props2.pNext = &pci;
    pd.getProperties2(&props2);
    char buf[256];
    snprintf(buf, sizeof(buf), "%s @ %02x:%02x.%x", props2.properties.deviceName.data(), pci.pciBus, pci.pciDevice, pci.pciFunction);
    g.name      = buf;
    g.ts_period = props2.properties.limits.timestampPeriod;

    // Prefer a compute-only family (async compute engine); fall back to anything with compute.
    const auto fams = pd.getQueueFamilyProperties();
    g.qfam = UINT32_MAX;
    for (uint32_t i = 0; i < fams.size(); i++) {
        if ((fams[i].queueFlags & vk::QueueFlagBits::eCompute) && !(fams[i].queueFlags & vk::QueueFlagBits::eGraphics)) {
            g.qfam = i;
            break;
        }
    }
    if (g.qfam == UINT32_MAX) {
        for (uint32_t i = 0; i < fams.size(); i++) {
            if (fams[i].queueFlags & vk::QueueFlagBits::eCompute) {
                g.qfam = i;
                break;
            }
        }
    }

    const float                 prio = 1.0f;
    vk::DeviceQueueCreateInfo   qci{ {}, g.qfam, 1, &prio };
    std::vector<const char *>   exts = {
        "VK_EXT_external_memory_host", "VK_KHR_external_memory_fd", "VK_EXT_external_memory_dma_buf",
        "VK_KHR_external_semaphore_fd",
    };
    vk::PhysicalDeviceVulkan12Features have12;
    vk::PhysicalDeviceFeatures2        have2;
    have2.pNext = &have12;
    pd.getFeatures2(&have2);
    vk::PhysicalDeviceVulkan12Features f12;
    f12.timelineSemaphore             = VK_TRUE;
    f12.vulkanMemoryModel             = have12.vulkanMemoryModel;
    f12.vulkanMemoryModelDeviceScope  = have12.vulkanMemoryModelDeviceScope;
    vk::DeviceCreateInfo dci{ {}, qci, {}, exts };
    dci.pNext = &f12;
    g.dev     = pd.createDevice(dci);
    g.q       = g.dev.getQueue(g.qfam, 0);
    g.pool    = g.dev.createCommandPool({ vk::CommandPoolCreateFlagBits::eResetCommandBuffer, g.qfam });
    g.memprops = pd.getMemoryProperties();

    g.get_host_ptr_props = (PFN_vkGetMemoryHostPointerPropertiesEXT) g.dev.getProcAddr("vkGetMemoryHostPointerPropertiesEXT");
    g.get_sem_fd         = (PFN_vkGetSemaphoreFdKHR) g.dev.getProcAddr("vkGetSemaphoreFdKHR");
    g.import_sem_fd      = (PFN_vkImportSemaphoreFdKHR) g.dev.getProcAddr("vkImportSemaphoreFdKHR");
    g.get_mem_fd         = (PFN_vkGetMemoryFdKHR) g.dev.getProcAddr("vkGetMemoryFdKHR");
    g.get_mem_fd_props   = (PFN_vkGetMemoryFdPropertiesKHR) g.dev.getProcAddr("vkGetMemoryFdPropertiesKHR");
    (void) inst;
    return g;
}

static buffer make_buffer(gpu & g, size_t size, vk::MemoryPropertyFlags want, bool exportable = false,
                          vk::MemoryPropertyFlags avoid = {}) {
    buffer b;
    b.size = size;
    vk::ExternalMemoryBufferCreateInfo ebci{ vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT };
    vk::BufferCreateInfo               bci{ {}, size, k_usage, vk::SharingMode::eExclusive };
    if (exportable) {
        bci.pNext = &ebci;
    }
    b.buf = g.dev.createBuffer(bci);
    const auto req = g.dev.getBufferMemoryRequirements(b.buf);

    vk::MemoryDedicatedAllocateInfo dedicated{ {}, b.buf };
    vk::ExportMemoryAllocateInfo    emai{ vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT };
    emai.pNext = &dedicated;
    vk::MemoryAllocateInfo mai{ req.size, find_mem_type(g, req.memoryTypeBits, want, avoid) };
    if (exportable) {
        mai.pNext = &emai;
    }
    b.mem = g.dev.allocateMemory(mai);
    g.dev.bindBufferMemory(b.buf, b.mem, 0);
    if (want & vk::MemoryPropertyFlagBits::eHostVisible) {
        b.ptr    = g.dev.mapMemory(b.mem, 0, size);
        b.mapped = true;
    }
    return b;
}

// Wrap a host allocation (page aligned) as a buffer on this device: VK_EXT_external_memory_host.
static buffer import_host(gpu & g, void * ptr, size_t size, bool verbose = false) {
    buffer b;
    b.size = size;
    b.ptr  = ptr;
    vk::ExternalMemoryBufferCreateInfo ebci{ vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT };
    vk::BufferCreateInfo               bci{ {}, size, k_usage, vk::SharingMode::eExclusive };
    bci.pNext = &ebci;
    b.buf     = g.dev.createBuffer(bci);

    VkMemoryHostPointerPropertiesEXT hp{ VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT };
    check(g.get_host_ptr_props(g.dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, ptr, &hp), "vkGetMemoryHostPointerPropertiesEXT");
    const auto req  = g.dev.getBufferMemoryRequirements(b.buf);
    const uint32_t type = find_mem_type(g, hp.memoryTypeBits & req.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible);
    if (verbose) {
        printf("  host import on %s: %s\n", g.name.c_str(), mem_type_str(g, type).c_str());
    }

    vk::ImportMemoryHostPointerInfoEXT imp{ vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT, ptr };
    vk::MemoryAllocateInfo             mai{ size, type };
    mai.pNext = &imp;
    b.mem     = g.dev.allocateMemory(mai);
    g.dev.bindBufferMemory(b.buf, b.mem, 0);
    return b;
}

// Import another device's exported dma-buf. Ownership of fd passes to the driver on success.
static buffer import_dmabuf(gpu & g, int fd, size_t size) {
    buffer b;
    b.size = size;
    vk::ExternalMemoryBufferCreateInfo ebci{ vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT };
    vk::BufferCreateInfo               bci{ {}, size, k_usage, vk::SharingMode::eExclusive };
    bci.pNext = &ebci;
    b.buf     = g.dev.createBuffer(bci);

    VkMemoryFdPropertiesKHR fp{ VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR };
    check(g.get_mem_fd_props(g.dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, fd, &fp), "vkGetMemoryFdPropertiesKHR");
    const auto     req  = g.dev.getBufferMemoryRequirements(b.buf);
    const uint32_t type = find_mem_type(g, fp.memoryTypeBits & req.memoryTypeBits, {});
    printf("  dma-buf import on %s: %s (allowed types 0x%x)\n", g.name.c_str(), mem_type_str(g, type).c_str(), fp.memoryTypeBits);

    vk::MemoryDedicatedAllocateInfo dedicated{ {}, b.buf };
    vk::ImportMemoryFdInfoKHR       imp{ vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT, fd };
    imp.pNext = &dedicated;
    vk::MemoryAllocateInfo mai{ req.size, type };
    mai.pNext = &imp;
    b.mem     = g.dev.allocateMemory(mai);
    g.dev.bindBufferMemory(b.buf, b.mem, 0);
    return b;
}

static int export_dmabuf(gpu & g, const buffer & b) {
    VkMemoryGetFdInfoKHR gi{ VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR, nullptr, b.mem, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT };
    int fd = -1;
    check(g.get_mem_fd(g.dev, &gi, &fd), "vkGetMemoryFdKHR");
    return fd;
}

static void free_buffer(gpu & g, buffer & b) {
    if (b.mapped) {
        g.dev.unmapMemory(b.mem);
    }
    g.dev.destroyBuffer(b.buf);
    g.dev.freeMemory(b.mem);
    b = {};
}

static vk::Semaphore make_timeline(gpu & g, bool exportable) {
    vk::ExportSemaphoreCreateInfo esci{ vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd };
    vk::SemaphoreTypeCreateInfo   stci{ vk::SemaphoreType::eTimeline, 0 };
    if (exportable) {
        stci.pNext = &esci;
    }
    vk::SemaphoreCreateInfo sci{};
    sci.pNext = &stci;
    return g.dev.createSemaphore(sci);
}

static vk::Semaphore import_timeline(gpu & dst, gpu & src, vk::Semaphore s) {
    int fd = -1;
    VkSemaphoreGetFdInfoKHR gi{ VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR, nullptr, s, VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT };
    check(src.get_sem_fd(src.dev, &gi, &fd), "vkGetSemaphoreFdKHR");
    vk::Semaphore d = make_timeline(dst, false);
    VkImportSemaphoreFdInfoKHR ii{ VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR, nullptr, d, 0, VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT, fd };
    check(dst.import_sem_fd(dst.dev, &ii), "vkImportSemaphoreFdKHR");
    return d;
}

struct sem_val {
    vk::Semaphore s;
    uint64_t      v;
};

static void submit(gpu & g, vk::CommandBuffer cb, const std::vector<sem_val> & waits, const std::vector<sem_val> & signals,
                   vk::Fence fence = {}) {
    std::vector<vk::Semaphore>          ws, ss;
    std::vector<uint64_t>               wv, sv;
    std::vector<vk::PipelineStageFlags> wst;
    for (const auto & w : waits) {
        ws.push_back(w.s);
        wv.push_back(w.v);
        wst.push_back(vk::PipelineStageFlagBits::eAllCommands);
    }
    for (const auto & s : signals) {
        ss.push_back(s.s);
        sv.push_back(s.v);
    }
    vk::TimelineSemaphoreSubmitInfo tsi{ wv, sv };
    vk::SubmitInfo                  si{ ws, wst, cb, ss };
    si.pNext = &tsi;
    g.q.submit(si, fence);
}

static uint64_t sem_value(gpu & g, vk::Semaphore s) {
    return g.dev.getSemaphoreCounterValue(s);
}

static void sem_wait(gpu & g, vk::Semaphore s, uint64_t v) {
    vk::SemaphoreWaitInfo wi{ {}, s, v };
    (void) g.dev.waitSemaphores(wi, UINT64_MAX);
}

static vk::CommandBuffer begin_cb(gpu & g) {
    auto cb = g.dev.allocateCommandBuffers({ g.pool, vk::CommandBufferLevel::ePrimary, 1 })[0];
    cb.begin(vk::CommandBufferBeginInfo{});
    return cb;
}

// A command buffer that does a minimal amount of real work (a 4 byte fill), so every submit reaches the GPU.
static vk::CommandBuffer tiny_cb(gpu & g, const buffer & scratch) {
    auto cb = begin_cb(g);
    cb.fillBuffer(scratch.buf, 0, 4, 0x1234);
    cb.end();
    return cb;
}

// ---------------------------------------------------------------------------------------------------------------

// Submit + fence wait round trip on one GPU: the floor for any CPU-mediated handoff.
static void test_fence(std::vector<gpu> & gpus) {
    printf("\n[fence] submit tiny work + vkWaitForFences, per GPU (CPU -> GPU -> CPU round trip)\n");
    for (auto & g : gpus) {
        buffer scratch = make_buffer(g, 4096, vk::MemoryPropertyFlagBits::eDeviceLocal);
        auto   cb      = tiny_cb(g, scratch);
        auto   fence   = g.dev.createFence({});
        std::vector<double> t;
        for (int i = 0; i < 2200; i++) {
            const auto t0 = clk::now();
            submit(g, cb, {}, {}, fence);
            (void) g.dev.waitForFences(fence, VK_TRUE, UINT64_MAX);
            g.dev.resetFences(fence);
            if (i >= 200) {
                t.push_back(us_since(t0));
            }
        }
        print_stats(g.name.c_str(), summarize(t));
        g.dev.destroyFence(fence);
        free_buffer(g, scratch);
    }
}

// CPU relay: run on GPU0, wait for it, then run on GPU1, wait. One hop = half a round trip of the pair.
static void test_relay(std::vector<gpu> & gpus) {
    printf("\n[relay] CPU waits for GPU0 then submits to GPU1, and back (blocking fences)\n");
    buffer s0 = make_buffer(gpus[0], 4096, vk::MemoryPropertyFlagBits::eDeviceLocal);
    buffer s1 = make_buffer(gpus[1], 4096, vk::MemoryPropertyFlagBits::eDeviceLocal);
    auto   cb0 = tiny_cb(gpus[0], s0), cb1 = tiny_cb(gpus[1], s1);
    auto   f0 = gpus[0].dev.createFence({}), f1 = gpus[1].dev.createFence({});
    std::vector<double> t;
    for (int i = 0; i < 2200; i++) {
        const auto t0 = clk::now();
        submit(gpus[0], cb0, {}, {}, f0);
        (void) gpus[0].dev.waitForFences(f0, VK_TRUE, UINT64_MAX);
        gpus[0].dev.resetFences(f0);
        submit(gpus[1], cb1, {}, {}, f1);
        (void) gpus[1].dev.waitForFences(f1, VK_TRUE, UINT64_MAX);
        gpus[1].dev.resetFences(f1);
        if (i >= 200) {
            t.push_back(us_since(t0) / 2);
        }
    }
    print_stats("per hop", summarize(t));
    gpus[0].dev.destroyFence(f0);
    gpus[1].dev.destroyFence(f1);
    free_buffer(gpus[0], s0);
    free_buffer(gpus[1], s1);
}

// The pattern of PR #25051's CPU proxy: all GPU work is submitted up front, each submit waits on a local timeline
// that a CPU thread signals (vkSignalSemaphore) once it sees the other GPU's timeline advance.
static void test_proxy(std::vector<gpu> & gpus) {
    const int rounds = 2000;
    printf("\n[proxy] pre-submitted work, CPU thread polls one GPU's timeline and host-signals the other's\n");
    buffer s0 = make_buffer(gpus[0], 4096, vk::MemoryPropertyFlagBits::eDeviceLocal);
    buffer s1 = make_buffer(gpus[1], 4096, vk::MemoryPropertyFlagBits::eDeviceLocal);
    auto   cb0 = tiny_cb(gpus[0], s0), cb1 = tiny_cb(gpus[1], s1);
    auto   go0 = make_timeline(gpus[0], false), done0 = make_timeline(gpus[0], false);
    auto   go1 = make_timeline(gpus[1], false), done1 = make_timeline(gpus[1], false);

    for (int k = 1; k <= rounds; k++) {
        submit(gpus[0], cb0, { { go0, (uint64_t) k } }, { { done0, (uint64_t) k } });
        submit(gpus[1], cb1, { { go1, (uint64_t) k } }, { { done1, (uint64_t) k } });
    }
    const auto t0 = clk::now();
    for (int k = 1; k <= rounds; k++) {
        gpus[0].dev.signalSemaphore({ go0, (uint64_t) k });
        while (sem_value(gpus[0], done0) < (uint64_t) k) {
        }
        gpus[1].dev.signalSemaphore({ go1, (uint64_t) k });
        while (sem_value(gpus[1], done1) < (uint64_t) k) {
        }
    }
    const double t = us_since(t0);
    printf("  %-44s %8.1f us (%d rounds, polling with getSemaphoreCounterValue)\n", "per hop", t / (2 * rounds), rounds);

    gpus[0].q.waitIdle();
    gpus[1].q.waitIdle();
    for (auto s : { go0, done0 }) gpus[0].dev.destroySemaphore(s);
    for (auto s : { go1, done1 }) gpus[1].dev.destroySemaphore(s);
    free_buffer(gpus[0], s0);
    free_buffer(gpus[1], s1);
}

// Native cross-device timeline semaphores: each GPU waits directly on the other's exported (OPAQUE_FD) timeline.
// No CPU involvement after submission, unless the driver adds it.
static void test_xsem(std::vector<gpu> & gpus) {
    const int rounds = 2000;
    printf("\n[xsem] pre-submitted ping-pong on exported/imported OPAQUE_FD timeline semaphores\n");
    for (int i = 0; i < 2; i++) {
        vk::SemaphoreTypeCreateInfo             stci{ vk::SemaphoreType::eTimeline, 0 };
        vk::PhysicalDeviceExternalSemaphoreInfo esi{ vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd };
        esi.pNext = &stci;
        const auto esp = gpus[i].pd.getExternalSemaphoreProperties(esi);
        printf("  %s: timeline OPAQUE_FD features 0x%x (exportable=%d importable=%d)\n", gpus[i].name.c_str(),
               (uint32_t) esp.externalSemaphoreFeatures,
               !!(esp.externalSemaphoreFeatures & vk::ExternalSemaphoreFeatureFlagBits::eExportable),
               !!(esp.externalSemaphoreFeatures & vk::ExternalSemaphoreFeatureFlagBits::eImportable));
    }

    buffer s0 = make_buffer(gpus[0], 4096, vk::MemoryPropertyFlagBits::eDeviceLocal);
    buffer s1 = make_buffer(gpus[1], 4096, vk::MemoryPropertyFlagBits::eDeviceLocal);
    auto   cb0 = tiny_cb(gpus[0], s0), cb1 = tiny_cb(gpus[1], s1);
    auto   t0s = make_timeline(gpus[0], true), t1s = make_timeline(gpus[1], true);
    auto   t0_on1 = import_timeline(gpus[1], gpus[0], t0s);
    auto   t1_on0 = import_timeline(gpus[0], gpus[1], t1s);

    // Variant A: submit all rounds up front, both GPUs interleaved.
    auto t0 = clk::now();
    for (int k = 1; k <= rounds; k++) {
        submit(gpus[0], cb0, { { t1_on0, (uint64_t) k - 1 } }, { { t0s, (uint64_t) k } });
        submit(gpus[1], cb1, { { t0_on1, (uint64_t) k } }, { { t1s, (uint64_t) k } });
    }
    const double t_submit = us_since(t0);
    sem_wait(gpus[1], t1s, rounds);
    double t = us_since(t0);
    printf("  %-44s %8.1f us (%d rounds; submitting took %.0f us of %.0f us)\n", "per hop, all pre-submitted", t / (2 * rounds),
           rounds, t_submit, t);

    // Variant B: GPU1 submits are issued only after GPU0's matching submit, one round at a time, like a real graph.
    std::vector<double> hops;
    for (int k = rounds + 1; k <= rounds + 500; k++) {
        const auto t1 = clk::now();
        submit(gpus[0], cb0, { { t1_on0, (uint64_t) k - 1 } }, { { t0s, (uint64_t) k } });
        submit(gpus[1], cb1, { { t0_on1, (uint64_t) k } }, { { t1s, (uint64_t) k } });
        sem_wait(gpus[1], t1s, k);
        hops.push_back(us_since(t1));
    }
    print_stats("round (2 submits + cross wait + host wait)", summarize(hops));

    gpus[0].q.waitIdle();
    gpus[1].q.waitIdle();
    gpus[0].dev.destroySemaphore(t0s);
    gpus[0].dev.destroySemaphore(t1_on0);
    gpus[1].dev.destroySemaphore(t1s);
    gpus[1].dev.destroySemaphore(t0_on1);
    free_buffer(gpus[0], s0);
    free_buffer(gpus[1], s1);
}

// ---------------------------------------------------------------------------------------------------------------

static std::vector<uint32_t> read_spv(const char * path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        fprintf(stderr, "cannot read %s\n", path);
        exit(1);
    }
    const size_t n = f.tellg();
    std::vector<uint32_t> code(n / 4);
    f.seekg(0);
    f.read((char *) code.data(), n);
    return code;
}

struct spin_pc {
    uint32_t role_base, rounds, max_spin, stride;
};

static void init_spin(gpu & g, const std::string & dir) {
    std::vector<vk::DescriptorSetLayoutBinding> b = {
        { 0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
        { 1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute },
    };
    g.dsl = g.dev.createDescriptorSetLayout({ {}, b });
    vk::PushConstantRange pcr{ vk::ShaderStageFlagBits::eCompute, 0, sizeof(spin_pc) };
    g.pl = g.dev.createPipelineLayout({ {}, g.dsl, pcr });
    for (int mode = 0; mode < 4; mode++) {
        const auto code = read_spv((dir + "spin" + std::to_string(mode) + ".spv").c_str());
        g.spin_module[mode] = g.dev.createShaderModule({ {}, code.size() * 4, code.data() });
        vk::ComputePipelineCreateInfo cpci{ {}, { {}, vk::ShaderStageFlagBits::eCompute, g.spin_module[mode], "main" }, g.pl };
        g.spin_pipeline[mode] = g.dev.createComputePipeline({}, cpci).value;
    }
    for (int wide = 0; wide < 2; wide++) {
        const auto code = read_spv((dir + "store" + std::to_string(wide) + ".spv").c_str());
        g.store_module[wide] = g.dev.createShaderModule({ {}, code.size() * 4, code.data() });
        vk::ComputePipelineCreateInfo cpci{ {}, { {}, vk::ShaderStageFlagBits::eCompute, g.store_module[wide], "main" }, g.pl };
        g.store_pipeline[wide] = g.dev.createComputePipeline({}, cpci).value;
    }
    vk::DescriptorPoolSize ps{ vk::DescriptorType::eStorageBuffer, 64 };
    g.dpool = g.dev.createDescriptorPool({ vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 32, ps });
}

static vk::CommandBuffer spin_cb(gpu & g, int mode, const buffer & flags, const buffer & result, spin_pc pc, uint32_t n_groups) {
    auto ds = g.dev.allocateDescriptorSets({ g.dpool, g.dsl })[0];
    vk::DescriptorBufferInfo fi{ flags.buf, 0, VK_WHOLE_SIZE }, ri{ result.buf, 0, VK_WHOLE_SIZE };
    std::vector<vk::WriteDescriptorSet> w = {
        { ds, 0, 0, vk::DescriptorType::eStorageBuffer, {}, fi },
        { ds, 1, 0, vk::DescriptorType::eStorageBuffer, {}, ri },
    };
    g.dev.updateDescriptorSets(w, {});
    auto cb = begin_cb(g);
    cb.bindPipeline(vk::PipelineBindPoint::eCompute, g.spin_pipeline[mode]);
    cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, g.pl, 0, ds, {});
    cb.pushConstants(g.pl, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), &pc);
    cb.dispatch(n_groups, 1, 1);
    cb.end();
    return cb;
}

static void report_spin(const char * what, const uint32_t * r, double t_us, uint32_t rounds) {
    const uint32_t fail0 = r[0], spins0 = r[1], fail1 = r[4], spins1 = r[5];
    if (fail0 || fail1) {
        printf("  %-44s FAILED (role0 stuck at round %u, role1 at round %u) after %.0f us\n", what, fail0, fail1, t_us);
    } else {
        printf("  %-44s %8.2f us per hop (%u rounds, avg spins/round %.1f + %.1f)\n", what, t_us / (2.0 * rounds), rounds,
               (double) spins0 / rounds, (double) spins1 / rounds);
    }
}

// Shader ping-pong through flags in memory both GPUs can see. The flag memory is a host allocation imported into
// both devices, or (p2p test) a dma-buf exported by one GPU.
static const char * k_spin_modes[4] = { "volatile", "vmm-atomic-ld/st", "rmw-atomics", "nontemporal" };

static void run_spin(std::vector<gpu> & gpus, int mode, buffer & f0, buffer & f1, volatile uint32_t * host_view, const char * what_base,
                     uint32_t rounds) {
    char what[96];
    snprintf(what, sizeof(what), "%s [%s]", what_base, k_spin_modes[mode]);
    buffer r0 = make_buffer(gpus[0], 4096, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    buffer r1 = make_buffer(gpus[1], 4096, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    const uint32_t max_spin = 2000000;
    const uint32_t stride   = 64;

    auto reset = [&]() {
        if (host_view) {
            for (int i = 0; i < 2 * (int) stride; i++) host_view[i] = 0;
        }
        memset(r0.ptr, 0, 64);
        memset(r1.ptr, 0, 64);
    };
    auto zero_flags_gpu = [&](gpu & g, buffer & f) {
        auto cb    = begin_cb(g);
        cb.fillBuffer(f.buf, 0, 2 * stride * 4, 0);
        cb.end();
        auto fence = g.dev.createFence({});
        submit(g, cb, {}, {}, fence);
        (void) g.dev.waitForFences(fence, VK_TRUE, UINT64_MAX);
        g.dev.destroyFence(fence);
    };

    // Cross-GPU: role 0 on GPU0, role 1 on GPU1.
    reset();
    if (!host_view) {
        zero_flags_gpu(gpus[0], f0);
    }
    {
        auto cb0 = spin_cb(gpus[0], mode, f0, r0, { 0, rounds, max_spin, stride }, 1);
        auto cb1 = spin_cb(gpus[1], mode, f1, r1, { 1, rounds, max_spin, stride }, 1);
        auto fe0 = gpus[0].dev.createFence({}), fe1 = gpus[1].dev.createFence({});

        // A CPU thread watches the flags in host memory: shows whether each GPU's store reaches memory while the
        // shaders run (writer side) independent of whether the other GPU can see it (reader side).
        std::atomic<bool> stop{ false };
        double seen0 = -1, seen1 = -1;
        uint32_t max0 = 0, max1 = 0;
        const auto t0 = clk::now();
        std::thread observer;
        if (host_view) {
            observer = std::thread([&]() {
                while (!stop.load(std::memory_order_relaxed)) {
                    const uint32_t a = host_view[0], b = host_view[stride];
                    if (a && seen0 < 0) seen0 = us_since(t0);
                    if (b && seen1 < 0) seen1 = us_since(t0);
                    max0 = std::max(max0, a);
                    max1 = std::max(max1, b);
                }
            });
        }
        submit(gpus[1], cb1, {}, {}, fe1);
        submit(gpus[0], cb0, {}, {}, fe0);
        (void) gpus[0].dev.waitForFences(fe0, VK_TRUE, UINT64_MAX);
        (void) gpus[1].dev.waitForFences(fe1, VK_TRUE, UINT64_MAX);
        const double t = us_since(t0);
        stop = true;
        if (observer.joinable()) {
            observer.join();
            printf("  %-44s CPU saw GPU0's flag at %.0f us (max %u), GPU1's at %.0f us (max %u)\n", "", seen0, max0, seen1, max1);
        }
        uint32_t r[8];
        memcpy(r, r0.ptr, 16);
        memcpy(r + 4, (uint32_t *) r1.ptr + 4, 16);
        char label[128];
        snprintf(label, sizeof(label), "%s, GPU0 <-> GPU1", what);
        report_spin(label, r, t, rounds);
        gpus[0].dev.destroyFence(fe0);
        gpus[1].dev.destroyFence(fe1);
    }

    // Loopback on each GPU: both roles in one dispatch, same memory.
    for (int i = 0; i < 2; i++) {
        reset();
        buffer & f = i == 0 ? f0 : f1;
        buffer & r = i == 0 ? r0 : r1;
        if (!host_view) {
            zero_flags_gpu(gpus[i], f);
        }
        auto cb = spin_cb(gpus[i], mode, f, r, { 0, rounds, max_spin, stride }, 2);
        auto fe = gpus[i].dev.createFence({});
        const auto t0 = clk::now();
        submit(gpus[i], cb, {}, {}, fe);
        (void) gpus[i].dev.waitForFences(fe, VK_TRUE, UINT64_MAX);
        const double t = us_since(t0);
        uint32_t rr[8];
        memcpy(rr, r.ptr, 32);
        char label[128];
        snprintf(label, sizeof(label), "%s, loopback on GPU%d", what, i);
        report_spin(label, rr, t, rounds);
        gpus[i].dev.destroyFence(fe);
    }
    free_buffer(gpus[0], r0);
    free_buffer(gpus[1], r1);
}

static void test_spin(std::vector<gpu> & gpus) {
    printf("\n[spin] shader ping-pong on a host allocation imported into both GPUs (VK_EXT_external_memory_host)\n");
    void * host = aligned_alloc(4096, 4096);
    memset(host, 0, 4096);
    buffer f0 = import_host(gpus[0], host, 4096, true);
    buffer f1 = import_host(gpus[1], host, 4096, true);
    for (int mode = 0; mode < 4; mode++) {
        run_spin(gpus, mode, f0, f1, (volatile uint32_t *) host, "host memory", 20000);
    }
    free_buffer(gpus[0], f0);
    free_buffer(gpus[1], f1);
    free(host);
}

// Same ping-pong, but the flags live in GTT memory that RADV allocates as write-combined (not snooped, not
// HOST_CACHED) on GPU0 and shares with GPU1 as a dma-buf. Host-pointer imports can only use the snooped, cached
// type, and GCN 1.0's L2 may treat the two differently.
static void test_spin_wc(std::vector<gpu> & gpus) {
    printf("\n[spin-wc] shader ping-pong on write-combined GTT memory from GPU0, shared with GPU1 as a dma-buf\n");
    buffer f0 = make_buffer(gpus[0], 4096, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, true,
                            vk::MemoryPropertyFlagBits::eHostCached | vk::MemoryPropertyFlagBits::eDeviceLocal);
    printf("  GPU0 allocation: %s\n", mem_type_str(gpus[0], find_mem_type(gpus[0], gpus[0].dev.getBufferMemoryRequirements(f0.buf).memoryTypeBits,
           vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
           vk::MemoryPropertyFlagBits::eHostCached | vk::MemoryPropertyFlagBits::eDeviceLocal)).c_str());
    buffer f1 = import_dmabuf(gpus[1], export_dmabuf(gpus[0], f0), 4096);
    for (int mode = 0; mode < 4; mode++) {
        run_spin(gpus, mode, f0, f1, (volatile uint32_t *) f0.ptr, "WC GTT", 20000);
    }
    free_buffer(gpus[1], f1);
    free_buffer(gpus[0], f0);
}

// ---------------------------------------------------------------------------------------------------------------

// GPU-timed copy: returns microseconds for one vkCmdCopyBuffer of `size` bytes, median of reps.
// writeback: put a transfer-write -> host-read barrier before the closing timestamp, which makes RADV write back L2
// first, so the timestamp covers the data actually leaving the GPU.
static double time_copy(gpu & g, const buffer & src, const buffer & dst, size_t size, int reps = 15, bool writeback = false) {
    auto qp = g.dev.createQueryPool({ {}, vk::QueryType::eTimestamp, 2 });
    std::vector<double> t;
    auto fence = g.dev.createFence({});
    for (int i = 0; i < reps + 2; i++) {
        auto cb = begin_cb(g);
        cb.resetQueryPool(qp, 0, 2);
        cb.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, qp, 0);
        cb.copyBuffer(src.buf, dst.buf, vk::BufferCopy{ 0, 0, size });
        if (writeback) {
            const vk::MemoryBarrier mb{ vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eHostRead | vk::AccessFlagBits::eMemoryRead };
            cb.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost | vk::PipelineStageFlagBits::eBottomOfPipe,
                               {}, mb, {}, {});
        }
        cb.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, qp, 1);
        cb.end();
        submit(g, cb, {}, {}, fence);
        (void) g.dev.waitForFences(fence, VK_TRUE, UINT64_MAX);
        g.dev.resetFences(fence);
        uint64_t ts[2];
        (void) g.dev.getQueryPoolResults(qp, 0, 2, sizeof(ts), ts, sizeof(uint64_t),
                                         vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
        if (i >= 2) {
            t.push_back((ts[1] - ts[0]) * g.ts_period / 1000.0);
        }
        g.dev.freeCommandBuffers(g.pool, cb);
    }
    g.dev.destroyFence(fence);
    g.dev.destroyQueryPool(qp);
    return summarize(t).p50;
}

static void print_copy(const char * what, size_t size, double us) {
    printf("  %-44s %8zu KiB %9.1f us %8.2f GB/s\n", what, size / 1024, us, size / us / 1000.0);
}

static const std::vector<size_t> k_sizes = { 16 << 10, 256 << 10, 1 << 20, 8 << 20 };

static void test_copy(std::vector<gpu> & gpus) {
    printf("\n[copy] GPU-timed vkCmdCopyBuffer between VRAM and a host allocation imported into both GPUs\n");
    const size_t maxsz = 8 << 20;
    void * host = aligned_alloc(4096, maxsz);
    memset(host, 0, maxsz);
    for (int i = 0; i < 2; i++) {
        gpu &  g    = gpus[i];
        buffer vram = make_buffer(g, maxsz, vk::MemoryPropertyFlagBits::eDeviceLocal, false, vk::MemoryPropertyFlagBits::eHostVisible);
        buffer hb   = import_host(g, host, maxsz, i == 0);
        for (size_t sz : k_sizes) {
            char label[96];
            snprintf(label, sizeof(label), "GPU%d VRAM -> host", i);
            print_copy(label, sz, time_copy(g, vram, hb, sz));
            snprintf(label, sizeof(label), "GPU%d host -> VRAM", i);
            print_copy(label, sz, time_copy(g, hb, vram, sz));
        }
        free_buffer(g, vram);
        free_buffer(g, hb);
    }
    free(host);
}

// Export a VRAM buffer from one GPU as a dma-buf and import it on the other. With PCI P2PDMA it can stay in the
// exporter's VRAM (BAR); without, amdgpu migrates it to GTT (system memory) when the importer maps it.
static void test_p2p(std::vector<gpu> & gpus) {
    printf("\n[p2p] dma-buf: GPU1 exports a DEVICE_LOCAL buffer, GPU0 imports it\n");
    const size_t maxsz = 8 << 20;
    buffer exp1 = make_buffer(gpus[1], maxsz, vk::MemoryPropertyFlagBits::eDeviceLocal, true, vk::MemoryPropertyFlagBits::eHostVisible);
    const int fd = export_dmabuf(gpus[1], exp1);
    buffer imp0 = import_dmabuf(gpus[0], fd, maxsz);
    buffer vram0 = make_buffer(gpus[0], maxsz, vk::MemoryPropertyFlagBits::eDeviceLocal, false, vk::MemoryPropertyFlagBits::eHostVisible);
    buffer vram1 = make_buffer(gpus[1], maxsz, vk::MemoryPropertyFlagBits::eDeviceLocal, false, vk::MemoryPropertyFlagBits::eHostVisible);
    for (size_t sz : k_sizes) {
        print_copy("GPU0 VRAM -> GPU1 buffer (GPU0 writes)", sz, time_copy(gpus[0], vram0, imp0, sz));
        print_copy("GPU1 buffer -> GPU0 VRAM (GPU0 reads)", sz, time_copy(gpus[0], imp0, vram0, sz));
        print_copy("GPU1 local copy of its own export", sz, time_copy(gpus[1], exp1, vram1, sz));
    }
    printf("  (local copy speed tells where the exported buffer lives now: VRAM-speed means it stayed in VRAM)\n");
    for (int mode = 0; mode < 4; mode++) {
        run_spin(gpus, mode, imp0, exp1, nullptr, "flags in GPU1 export", 20000);
    }
    free_buffer(gpus[0], imp0);
    free_buffer(gpus[0], vram0);
    free_buffer(gpus[1], vram1);
    free_buffer(gpus[1], exp1);
}

// Data check for the p2p path. Bandwidth alone can't tell a peer write that lands in the right place from one that
// lands somewhere else, so each GPU in turn imports the other's exported VRAM buffer, writes into it and reads from
// it over PCIe (full size and at random sub-ranges), and the CPU checks every word through host-visible staging.
static uint32_t pattern_word(uint64_t seed, size_t i) {
    uint64_t z = seed * 0x9e3779b97f4a7c15ull + i;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return (uint32_t) (z ^ (z >> 31));
}

static void copy_wait(gpu & g, const buffer & src, const buffer & dst, vk::BufferCopy region) {
    auto cb = begin_cb(g);
    cb.copyBuffer(src.buf, dst.buf, region);
    cb.end();
    auto fence = g.dev.createFence({});
    submit(g, cb, {}, {}, fence);
    (void) g.dev.waitForFences(fence, VK_TRUE, UINT64_MAX);
    g.dev.destroyFence(fence);
    g.dev.freeCommandBuffers(g.pool, cb);
}

static void test_p2p_verify(std::vector<gpu> & gpus) {
    const size_t   sz = 32 << 20, words = sz / 4;
    const int      rounds = 16, subranges = 4;
    bool           all_ok = true;
    printf("\n[p2p-verify] dma-buf P2P data check: %zu MiB buffers, %d rounds of full write, full read and %d sub-range writes\n",
           sz >> 20, rounds, subranges);
    for (int init = 0; init < 2; init++) {
        gpu & gi = gpus[init];
        gpu & ge = gpus[1 - init];
        buffer exp  = make_buffer(ge, sz, vk::MemoryPropertyFlagBits::eDeviceLocal, true, vk::MemoryPropertyFlagBits::eHostVisible);
        buffer imp  = import_dmabuf(gi, export_dmabuf(ge, exp), sz);
        buffer v_i  = make_buffer(gi, sz, vk::MemoryPropertyFlagBits::eDeviceLocal, false, vk::MemoryPropertyFlagBits::eHostVisible);
        buffer v_e  = make_buffer(ge, sz, vk::MemoryPropertyFlagBits::eDeviceLocal, false, vk::MemoryPropertyFlagBits::eHostVisible);
        buffer st_i = make_buffer(gi, sz, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        buffer st_e = make_buffer(ge, sz, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        uint32_t * pi = (uint32_t *) st_i.ptr;
        uint32_t * pe = (uint32_t *) st_e.ptr;
        const vk::BufferCopy full{ 0, 0, sz };

        size_t   checked = 0, bad = 0;
        long     first_bad = -1;
        uint32_t first_got = 0, first_want = 0;
        auto compare = [&](const uint32_t * got, auto want) {
            for (size_t i = 0; i < words; i++) {
                const uint32_t w = want(i);
                if (got[i] != w) {
                    if (first_bad < 0) {
                        first_bad = (long) i, first_got = got[i], first_want = w;
                    }
                    bad++;
                }
            }
            checked += words;
        };
        uint64_t rng = 0x5eed0000ull + init;
        auto next = [&]() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; };

        for (int r = 0; r < rounds; r++) {
            const uint64_t s_write = 3 * (uint64_t) r + 1 + 1000 * init, s_read = s_write + 1, s_sub = s_write + 2;

            // P2P write: initiator VRAM -> peer's exported VRAM; the exporter reads it back locally.
            for (size_t i = 0; i < words; i++) pi[i] = pattern_word(s_write, i);
            copy_wait(gi, st_i, v_i, full);
            copy_wait(gi, v_i, imp, full);
            copy_wait(ge, exp, st_e, full);
            compare(pe, [&](size_t i) { return pattern_word(s_write, i); });

            // P2P read: the exporter fills its buffer locally; the initiator reads it across into its own VRAM.
            for (size_t i = 0; i < words; i++) pe[i] = pattern_word(s_read, i);
            copy_wait(ge, st_e, exp, full);
            copy_wait(gi, imp, v_i, full);
            copy_wait(gi, v_i, st_i, full);
            compare(pi, [&](size_t i) { return pattern_word(s_read, i); });

            // Sub-range P2P writes at random word offsets: bytes outside each range must keep the previous contents.
            for (size_t i = 0; i < words; i++) pi[i] = pattern_word(s_sub, i);
            copy_wait(gi, st_i, v_i, full);
            std::vector<std::pair<size_t, size_t>> ranges;
            for (int k = 0; k < subranges; k++) {
                const size_t off = next() % words, len = 1 + next() % std::min<size_t>(words - off, 1 << 20);
                ranges.push_back({ off, len });
                copy_wait(gi, v_i, imp, vk::BufferCopy{ off * 4, off * 4, len * 4 });
            }
            copy_wait(ge, exp, st_e, full);
            compare(pe, [&](size_t i) {
                for (const auto & [off, len] : ranges) {
                    if (i >= off && i < off + len) return pattern_word(s_sub, i);
                }
                return pattern_word(s_read, i);
            });
        }

        // Where the export lives after all that: VRAM-speed local copies mean the importer really went peer-to-peer.
        const double local_us = time_copy(ge, exp, v_e, sz, 5);
        const double gbs      = sz / local_us / 1000.0;
        printf("  GPU%d writes/reads GPU%d's VRAM: %zu MiB checked, %zu bad words", init, 1 - init, checked * 4 >> 20, bad);
        if (bad) {
            printf(" (first at word %ld: got 0x%08x want 0x%08x)", first_bad, first_got, first_want);
        }
        printf("; export local copy %.1f GB/s (%s)\n", gbs, gbs > 30 ? "in VRAM" : "NOT in VRAM");
        all_ok = all_ok && bad == 0 && gbs > 30;

        free_buffer(gi, imp);
        free_buffer(gi, v_i);
        free_buffer(gi, st_i);
        free_buffer(ge, v_e);
        free_buffer(ge, st_e);
        free_buffer(ge, exp);
    }
    printf("  p2p-verify: %s\n", all_ok ? "PASS" : "FAIL");
}

// GPU timestamps undercount copies whose destination is outside local VRAM: the bottom-of-pipe timestamp can land
// while up to 768 KiB of written lines still sit in Tahiti's L2, before the end-of-submission writeback. This times
// submit-to-fence on the CPU instead (which includes that writeback), minus an empty submission, at sizes well past L2.
static double wall_copy(gpu & g, const buffer & src, const buffer & dst, size_t size, int reps = 9) {
    auto fence = g.dev.createFence({});
    auto once  = [&](size_t n) {
        auto cb = begin_cb(g);
        if (n) {
            cb.copyBuffer(src.buf, dst.buf, vk::BufferCopy{ 0, 0, n });
        }
        cb.end();
        const auto t0 = clk::now();
        submit(g, cb, {}, {}, fence);
        (void) g.dev.waitForFences(fence, VK_TRUE, UINT64_MAX);
        const double t = us_since(t0);
        g.dev.resetFences(fence);
        g.dev.freeCommandBuffers(g.pool, cb);
        return t;
    };
    std::vector<double> empty, full;
    for (int i = 0; i < reps + 2; i++) {
        const double e = once(0), f = once(size);
        if (i >= 2) {
            empty.push_back(e);
            full.push_back(f);
        }
    }
    g.dev.destroyFence(fence);
    return summarize(full).p50 - summarize(empty).p50;
}

static void test_p2p_wall(std::vector<gpu> & gpus) {
    printf("\n[p2p-wall] submit-to-fence copy time minus an empty submit, GPU0 <-> GPU1's exported VRAM, and host memory\n");
    const size_t maxsz = 64 << 20;
    buffer exp1  = make_buffer(gpus[1], maxsz, vk::MemoryPropertyFlagBits::eDeviceLocal, true, vk::MemoryPropertyFlagBits::eHostVisible);
    buffer imp0  = import_dmabuf(gpus[0], export_dmabuf(gpus[1], exp1), maxsz);
    buffer vram0 = make_buffer(gpus[0], maxsz, vk::MemoryPropertyFlagBits::eDeviceLocal, false, vk::MemoryPropertyFlagBits::eHostVisible);
    buffer vram1 = make_buffer(gpus[1], maxsz, vk::MemoryPropertyFlagBits::eDeviceLocal, false, vk::MemoryPropertyFlagBits::eHostVisible);
    buffer host0 = make_buffer(gpus[0], maxsz, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, false,
                               vk::MemoryPropertyFlagBits::eDeviceLocal);
    printf("  GPU0 host buffer: %s\n", mem_type_str(gpus[0], find_mem_type(gpus[0], gpus[0].dev.getBufferMemoryRequirements(host0.buf).memoryTypeBits,
           vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, vk::MemoryPropertyFlagBits::eDeviceLocal)).c_str());
    printf("  P2P write, three ways of timing the same copy:\n");
    for (size_t sz : { (size_t) 64 << 10, (size_t) 256 << 10, (size_t) 512 << 10, (size_t) 1 << 20, (size_t) 2 << 20, (size_t) 8 << 20, (size_t) 32 << 20 }) {
        print_copy("  GPU timestamps", sz, time_copy(gpus[0], vram0, imp0, sz));
        print_copy("  GPU timestamps + L2 writeback barrier", sz, time_copy(gpus[0], vram0, imp0, sz, 15, true));
        print_copy("  CPU submit-to-fence", sz, wall_copy(gpus[0], vram0, imp0, sz));
    }
    for (size_t sz : { (size_t) 1 << 20, (size_t) 8 << 20, (size_t) 32 << 20, (size_t) 64 << 20 }) {
        print_copy("GPU0 VRAM -> GPU1 VRAM (P2P write)", sz, wall_copy(gpus[0], vram0, imp0, sz));
        print_copy("GPU1 VRAM -> GPU0 VRAM (P2P read)", sz, wall_copy(gpus[0], imp0, vram0, sz));
        print_copy("GPU0 VRAM -> host (write)", sz, wall_copy(gpus[0], vram0, host0, sz));
        print_copy("host -> GPU0 VRAM (read)", sz, wall_copy(gpus[0], host0, vram0, sz));
        print_copy("GPU1 local copy of its own export", sz, wall_copy(gpus[1], exp1, vram1, sz));
    }
    free_buffer(gpus[0], imp0);
    free_buffer(gpus[0], vram0);
    free_buffer(gpus[0], host0);
    free_buffer(gpus[1], vram1);
    free_buffer(gpus[1], exp1);
}

// Submit-to-fence wall time of one recorded command buffer, minus an empty submission (median of reps).
static double wall_record(gpu & g, const std::function<void(vk::CommandBuffer)> & record, int reps = 7) {
    auto fence = g.dev.createFence({});
    auto once  = [&](bool work) {
        auto cb = begin_cb(g);
        if (work) {
            record(cb);
        }
        cb.end();
        const auto t0 = clk::now();
        submit(g, cb, {}, {}, fence);
        (void) g.dev.waitForFences(fence, VK_TRUE, UINT64_MAX);
        const double t = us_since(t0);
        g.dev.resetFences(fence);
        g.dev.freeCommandBuffers(g.pool, cb);
        return t;
    };
    std::vector<double> empty, full;
    for (int i = 0; i < reps + 1; i++) {
        const double e = once(false), f = once(true);
        if (i >= 1) {
            empty.push_back(e);
            full.push_back(f);
        }
    }
    g.dev.destroyFence(fence);
    return summarize(full).p50 - summarize(empty).p50;
}

// GPU writes into different destinations by different methods: RADV's copy and fill, and store.comp with 4- or
// 16-byte stores, either contiguous or one store per 64-byte line (partial lines). Reports bytes stored per second
// and 64-byte lines touched per second.
static void test_host_write(std::vector<gpu> & gpus) {
    printf("\n[host-write] GPU0 writes, submit-to-fence wall time, 32 MiB region\n");
    gpu &        g  = gpus[0];
    const size_t sz = 32 << 20;
    buffer src   = make_buffer(g, sz, vk::MemoryPropertyFlagBits::eDeviceLocal, false, vk::MemoryPropertyFlagBits::eHostVisible);
    buffer vram  = make_buffer(g, sz, vk::MemoryPropertyFlagBits::eDeviceLocal, false, vk::MemoryPropertyFlagBits::eHostVisible);
    buffer exp1  = make_buffer(gpus[1], sz, vk::MemoryPropertyFlagBits::eDeviceLocal, true, vk::MemoryPropertyFlagBits::eHostVisible);
    buffer peer  = import_dmabuf(g, export_dmabuf(gpus[1], exp1), sz);
    buffer wc    = make_buffer(g, sz, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, false,
                               vk::MemoryPropertyFlagBits::eHostCached | vk::MemoryPropertyFlagBits::eDeviceLocal);
    buffer cached = make_buffer(g, sz, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent |
                                vk::MemoryPropertyFlagBits::eHostCached);
    void * host  = aligned_alloc(4096, sz);
    memset(host, 0, sz);
    buffer hostb = import_host(g, host, sz);
    struct dst { const char * name; buffer * b; };
    const dst dsts[] = { { "local VRAM", &vram }, { "peer VRAM (P2P)", &peer }, { "WC GTT", &wc }, { "cached GTT", &cached },
                         { "imported malloc", &hostb } };
    printf("  %-18s %-34s %10s %9s %11s\n", "destination", "method", "us", "GB/s", "Mlines/s");
    auto row = [&](const char * dname, const char * method, double us, size_t bytes, size_t lines) {
        printf("  %-18s %-34s %10.1f %9.2f %11.1f\n", dname, method, us, bytes / us / 1000.0, lines / us);
    };
    auto ds = g.dev.allocateDescriptorSets({ g.dpool, g.dsl })[0];
    for (const auto & d : dsts) {
        row(d.name, "vkCmdCopyBuffer from VRAM", wall_record(g, [&](vk::CommandBuffer cb) { cb.copyBuffer(src.buf, d.b->buf, vk::BufferCopy{ 0, 0, sz }); }),
            sz, sz / 64);
        row(d.name, "vkCmdFillBuffer", wall_record(g, [&](vk::CommandBuffer cb) { cb.fillBuffer(d.b->buf, 0, sz, 0x12345678); }), sz, sz / 64);
        vk::DescriptorBufferInfo bi{ d.b->buf, 0, VK_WHOLE_SIZE };
        std::vector<vk::WriteDescriptorSet> w = { { ds, 0, 0, vk::DescriptorType::eStorageBuffer, {}, bi },
                                                  { ds, 1, 0, vk::DescriptorType::eStorageBuffer, {}, bi } };
        g.dev.updateDescriptorSets(w, {});
        struct variant { int wide; uint32_t stride; const char * name; };
        const variant vs[] = { { 0, 1, "store 4 B/lane, contiguous" }, { 1, 1, "store 16 B/lane, contiguous" },
                               { 0, 16, "store 4 B per 64 B line" }, { 1, 4, "store 16 B per 64 B line" } };
        for (const auto & v : vs) {
            const uint32_t elem = v.wide ? 16 : 4;
            const uint32_t n    = (uint32_t) (sz / (elem * v.stride));
            const uint32_t pc[4] = { n, v.stride, 7, 0 };
            const double us = wall_record(g, [&](vk::CommandBuffer cb) {
                cb.bindPipeline(vk::PipelineBindPoint::eCompute, g.store_pipeline[v.wide]);
                cb.bindDescriptorSets(vk::PipelineBindPoint::eCompute, g.pl, 0, ds, {});
                cb.pushConstants(g.pl, vk::ShaderStageFlagBits::eCompute, 0, sizeof(pc), pc);
                cb.dispatch(512, 1, 1);
            });
            row(d.name, v.name, us, (size_t) n * elem, (size_t) n * elem * v.stride / 64);
        }
    }
    g.dev.freeDescriptorSets(g.dpool, ds);
    free_buffer(g, hostb);
    free(host);
    free_buffer(g, cached);
    free_buffer(g, wc);
    free_buffer(g, peer);
    free_buffer(gpus[1], exp1);
    free_buffer(g, vram);
    free_buffer(g, src);
}

int main(int argc, char ** argv) {
    std::vector<std::string> tests;
    for (int i = 1; i < argc; i++) {
        tests.push_back(argv[i]);
    }
    if (tests.empty()) {
        tests = { "fence", "relay", "proxy", "xsem", "spin", "spin-wc", "copy", "p2p", "p2p-verify", "p2p-wall", "host-write" };
    }

    vk::ApplicationInfo ai{ "xgpu", 1, nullptr, 0, VK_API_VERSION_1_3 };
    vk::Instance        inst = vk::createInstance({ {}, &ai });
    std::vector<gpu>    gpus;
    for (auto pd : inst.enumeratePhysicalDevices()) {
        if (pd.getProperties().deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
            gpus.push_back(make_gpu(inst, pd));
        }
    }
    if (gpus.size() < 2) {
        fprintf(stderr, "need two discrete GPUs, found %zu\n", gpus.size());
        return 1;
    }
    gpus.resize(2);
    for (int i = 0; i < 2; i++) {
        printf("GPU%d: %s, queue family %u, timestamp period %.2f ns\n", i, gpus[i].name.c_str(), gpus[i].qfam, gpus[i].ts_period);
    }

    std::string exe = argv[0];
    const std::string dir = exe.substr(0, exe.find_last_of('/') + 1);
    for (auto & g : gpus) {
        init_spin(g, dir);
    }

    for (const auto & t : tests) {
        if (t == "fence") test_fence(gpus);
        else if (t == "relay") test_relay(gpus);
        else if (t == "proxy") test_proxy(gpus);
        else if (t == "xsem") test_xsem(gpus);
        else if (t == "spin") test_spin(gpus);
        else if (t == "spin-wc") test_spin_wc(gpus);
        else if (t == "copy") test_copy(gpus);
        else if (t == "p2p") test_p2p(gpus);
        else if (t == "p2p-verify") test_p2p_verify(gpus);
        else if (t == "p2p-wall") test_p2p_wall(gpus);
        else if (t == "host-write") test_host_write(gpus);
        else fprintf(stderr, "unknown test %s\n", t.c_str());
        fflush(stdout);
    }
    return 0;
}
