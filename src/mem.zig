const std = @import("std");

pub const Info = struct {
    model_mib: u64,
    available_mib: u64,
    limit_mib: u64, // 80% of available (GPU needs headroom for activations/KV cache)
    fit: bool,
};

pub fn checkGpuModelFitsInMemory(path: []const u8) ?Info {
    const model_bytes = estimateModelSize(path) orelse return null;
    const available_bytes = getAvailableGpuMemory() orelse return null;

    const model_mib = model_bytes / (1024 * 1024);
    const available_mib = available_bytes / (1024 * 1024);
    const limit_mib = available_mib / 5 * 4; // 80%

    return .{
        .model_mib = model_mib,
        .available_mib = available_mib,
        .limit_mib = limit_mib,
        .fit = model_mib < limit_mib,
    };
}

fn estimateModelSize(path: []const u8) ?u64 {
    // Try as single file (GGUF)
    if (std.fs.cwd().openFile(path, .{})) |file| {
        defer file.close();
        const stat = file.stat() catch return null;
        if (stat.kind == .file) {
            return if (stat.size > 0) stat.size else null;
        }
    } else |_| {}

    // Try as directory (HuggingFace with .safetensors files)
    var dir = std.fs.cwd().openDir(path, .{ .iterate = true }) catch return null;
    defer dir.close();

    var total: u64 = 0;
    var iter = dir.iterate();
    while (iter.next() catch return null) |entry| {
        if (entry.kind != .file) continue;
        if (std.mem.endsWith(u8, entry.name, ".safetensors")) {
            const file = dir.openFile(entry.name, .{}) catch continue;
            defer file.close();
            const stat = file.stat() catch continue;
            total += stat.size;
        }
    }

    return if (total > 0) total else null;
}

// Vulkan type definitions for dynamic loading (no vulkan-zig dependency needed)
const VK_SUCCESS: c_int = 0;
const VK_API_VERSION_1_0: u32 = 1 << 22; // version 1.0.0
const VK_MEMORY_HEAP_DEVICE_LOCAL_BIT: u32 = 0x00000001;
const VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT: u32 = 0x00000001;
const VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT: u32 = 0x00000002;
const VK_MAX_MEMORY_HEAPS = 16;
const VK_MAX_MEMORY_TYPES = 32;

const VkInstance = *opaque {};
const VkPhysicalDevice = *opaque {};

const VkApplicationInfo = extern struct {
    sType: u32 = 0, // VK_STRUCTURE_TYPE_APPLICATION_INFO
    pNext: ?*const anyopaque = null,
    pApplicationName: ?[*:0]const u8 = null,
    applicationVersion: u32 = 0,
    pEngineName: ?[*:0]const u8 = null,
    engineVersion: u32 = 0,
    apiVersion: u32 = VK_API_VERSION_1_0,
};

const VkInstanceCreateInfo = extern struct {
    sType: u32 = 1, // VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO
    pNext: ?*const anyopaque = null,
    flags: u32 = 0,
    pApplicationInfo: ?*const VkApplicationInfo = null,
    enabledLayerCount: u32 = 0,
    ppEnabledLayerNames: ?[*]const [*:0]const u8 = null,
    enabledExtensionCount: u32 = 0,
    ppEnabledExtensionNames: ?[*]const [*:0]const u8 = null,
};

const VkMemoryHeap = extern struct {
    size: u64,
    flags: u32,
};

const VkMemoryType = extern struct {
    propertyFlags: u32,
    heapIndex: u32,
};

const VkPhysicalDeviceMemoryProperties = extern struct {
    memoryTypeCount: u32,
    memoryTypes: [VK_MAX_MEMORY_TYPES]VkMemoryType,
    memoryHeapCount: u32,
    memoryHeaps: [VK_MAX_MEMORY_HEAPS]VkMemoryHeap,
};

const PfnVkCreateInstance = *const fn (*const VkInstanceCreateInfo, ?*const anyopaque, *?VkInstance) callconv(.c) c_int;
const PfnVkDestroyInstance = *const fn (VkInstance, ?*const anyopaque) callconv(.c) void;
const PfnVkEnumeratePhysicalDevices = *const fn (VkInstance, *u32, ?[*]VkPhysicalDevice) callconv(.c) c_int;
const PfnVkGetPhysicalDeviceMemoryProperties = *const fn (VkPhysicalDevice, *VkPhysicalDeviceMemoryProperties) callconv(.c) void;

fn getAvailableGpuMemory() ?u64 {
    const builtin = @import("builtin");
    if (builtin.os.tag != .linux) return null;

    var lib = std.DynLib.open("libvulkan.so.1") catch return null;
    defer lib.close();

    const createInstance = lib.lookup(PfnVkCreateInstance, "vkCreateInstance") orelse return null;
    const destroyInstance = lib.lookup(PfnVkDestroyInstance, "vkDestroyInstance") orelse return null;
    const enumeratePhysicalDevices = lib.lookup(PfnVkEnumeratePhysicalDevices, "vkEnumeratePhysicalDevices") orelse return null;
    const getMemProps = lib.lookup(PfnVkGetPhysicalDeviceMemoryProperties, "vkGetPhysicalDeviceMemoryProperties") orelse return null;

    const app_info = VkApplicationInfo{ .pApplicationName = "infer-memcheck" };
    const create_info = VkInstanceCreateInfo{ .pApplicationInfo = &app_info };

    var instance: ?VkInstance = null;
    if (createInstance(&create_info, null, &instance) != VK_SUCCESS) return null;
    const inst = instance orelse return null;
    defer destroyInstance(inst, null);

    var device_count: u32 = 0;
    if (enumeratePhysicalDevices(inst, &device_count, null) != VK_SUCCESS) return null;
    if (device_count == 0) return null;

    var devices: [8]VkPhysicalDevice = undefined;
    var count: u32 = @min(device_count, 8);
    if (enumeratePhysicalDevices(inst, &count, &devices) != VK_SUCCESS) return null;

    // Two passes: first look for discrete GPUs with dedicated VRAM (heaps
    // referenced by DEVICE_LOCAL-only memory types, not HOST_VISIBLE). Only
    // fall back to integrated GPU shared memory if no discrete GPU is found.
    var best_dedicated: u64 = 0;
    var best_shared: u64 = 0;

    for (devices[0..count]) |device| {
        var mem_props: VkPhysicalDeviceMemoryProperties = undefined;
        getMemProps(device, &mem_props);

        // Find heaps referenced by memory types that are DEVICE_LOCAL but NOT
        // HOST_VISIBLE — these are dedicated VRAM, not system RAM exposed via
        // resizable BAR or integrated GPU shared memory.
        var vram_heap_mask: u32 = 0;
        for (mem_props.memoryTypes[0..mem_props.memoryTypeCount]) |mem_type| {
            if (mem_type.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT != 0 and
                mem_type.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT == 0)
            {
                vram_heap_mask |= @as(u32, 1) << @intCast(mem_type.heapIndex);
            }
        }

        var dedicated: u64 = 0;
        for (mem_props.memoryHeaps[0..mem_props.memoryHeapCount], 0..) |heap, i| {
            if (vram_heap_mask & (@as(u32, 1) << @intCast(i)) != 0) {
                dedicated += heap.size;
            }
        }

        if (dedicated > 0) {
            if (dedicated > best_dedicated) best_dedicated = dedicated;
        } else {
            // Integrated GPU: all device-local memory is host-visible.
            // Use the largest device-local heap.
            var shared: u64 = 0;
            for (mem_props.memoryHeaps[0..mem_props.memoryHeapCount]) |heap| {
                if (heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT != 0) {
                    if (heap.size > shared) shared = heap.size;
                }
            }
            if (shared > best_shared) best_shared = shared;
        }
    }

    const vram = if (best_dedicated > 0) best_dedicated else best_shared;
    return if (vram > 0) vram else null;
}

test "getAvailableGpuMemory returns value or null" {
    // On systems with Vulkan + GPU this returns a value; otherwise null.
    // Either outcome is valid — we just verify it doesn't crash.
    _ = getAvailableGpuMemory();
}
