const std = @import("std");
pub const vk = @import("vulkan");
const log = std.log.scoped(.vulkan);

// Re-export vulkan types for use in other modules
pub const Buffer = vk.Buffer;
pub const DeviceMemory = vk.DeviceMemory;
pub const DeviceSize = vk.DeviceSize;
pub const CommandBuffer = vk.CommandBuffer;
pub const Pipeline = vk.Pipeline;
pub const PipelineLayout = vk.PipelineLayout;
pub const DescriptorSet = vk.DescriptorSet;
pub const DescriptorSetLayout = vk.DescriptorSetLayout;
pub const DescriptorPool = vk.DescriptorPool;
pub const ShaderModule = vk.ShaderModule;
pub const BufferUsageFlags = vk.BufferUsageFlags;
pub const MemoryBarrier = vk.MemoryBarrier;

pub const GpuBuffer = struct {
    buffer: vk.Buffer,
    memory: vk.DeviceMemory,
    size: vk.DeviceSize,
    mapped: ?[*]u8,
};

pub const CoopPreference = enum { prefer_nv, prefer_khr };

pub const InitOptions = struct {
    coop_preference: CoopPreference = .prefer_nv,
};

pub const VulkanDevice = struct {
    allocator: std.mem.Allocator,

    // Dispatch tables
    vki: vk.InstanceWrapper,
    vkd: vk.DeviceWrapper,

    // Core handles
    instance: vk.Instance,
    physical_device: vk.PhysicalDevice,
    device: vk.Device,
    compute_queue: vk.Queue,
    queue_family_index: u32,

    // Command infrastructure
    command_pool: vk.CommandPool,
    descriptor_pool: vk.DescriptorPool,

    // Memory type indices
    device_local_memory_type: u32,
    host_visible_memory_type: u32,
    host_cached_memory_type: u32,

    // Device limits
    max_workgroup_size: u32,

    // Extension support
    has_coop_matrix: bool,
    has_coop_matrix_int8: bool,
    has_integer_dot_product: bool,
    timestamp_period: f32,

    // GPU portability
    compute_units: u32,
    subgroup_size: u32,
    min_subgroup_size: u32,
    has_subgroup_size_control: bool,

    // Memory tracking
    allocated_bytes: usize,

    // Persistent upload staging buffer, grown to the high-water-mark of all
    // uploadToBuffer calls. Avoids the allocate/destroy churn of creating
    // a fresh host-visible staging buffer per tensor (40+ cycles for a 4B
    // model init) and keeps peak-during-upload bounded by the largest
    // single tensor instead of (cumulative device-local + current staging).
    upload_staging: ?GpuBuffer,

    // Dynamic library handle
    vk_lib: std.DynLib,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, options: InitOptions) !*Self {
        // Dynamically load libvulkan
        var vk_lib = std.DynLib.open("libvulkan.so.1") catch {
            log.err("failed to load libvulkan.so.1", .{});
            return error.VulkanUnavailable;
        };
        errdefer vk_lib.close();

        const get_instance_proc_addr = vk_lib.lookup(vk.PfnGetInstanceProcAddr, "vkGetInstanceProcAddr") orelse {
            log.err("vkGetInstanceProcAddr not found in libvulkan", .{});
            return error.VulkanUnavailable;
        };

        // Load base dispatch (global Vulkan functions)
        const vkb = vk.BaseWrapper.load(get_instance_proc_addr);

        // Create instance
        const app_info = vk.ApplicationInfo{
            .api_version = @bitCast(vk.API_VERSION_1_1),
            .application_version = 0,
            .engine_version = 0,
            .p_application_name = "infer",
            .p_engine_name = "infer",
        };
        const instance = vkb.createInstance(&.{
            .p_application_info = &app_info,
        }, null) catch {
            log.err("failed to create Vulkan instance", .{});
            return error.VulkanUnavailable;
        };

        // Load instance dispatch
        const vki = vk.InstanceWrapper.load(instance, get_instance_proc_addr);

        // Enumerate physical devices
        var device_count: u32 = 0;
        _ = vki.enumeratePhysicalDevices(instance, &device_count, null) catch return error.VulkanUnavailable;
        if (device_count == 0) {
            log.err("no Vulkan physical devices found", .{});
            return error.VulkanUnavailable;
        }

        const devices = try allocator.alloc(vk.PhysicalDevice, device_count);
        defer allocator.free(devices);
        _ = vki.enumeratePhysicalDevices(instance, &device_count, devices.ptr) catch return error.VulkanUnavailable;

        // Select best device (prefer discrete GPU)
        var selected_device = devices[0];
        var selected_props = vki.getPhysicalDeviceProperties(selected_device);

        for (devices[1..device_count]) |dev| {
            const props = vki.getPhysicalDeviceProperties(dev);
            if (props.device_type == .discrete_gpu and selected_props.device_type != .discrete_gpu) {
                selected_device = dev;
                selected_props = props;
            }
        }

        const device_name: [*:0]const u8 = @ptrCast(&selected_props.device_name);
        log.info("Vulkan device: {s}", .{device_name});

        // Find compute queue family
        var queue_family_count: u32 = 0;
        vki.getPhysicalDeviceQueueFamilyProperties(selected_device, &queue_family_count, null);
        const queue_families = try allocator.alloc(vk.QueueFamilyProperties, queue_family_count);
        defer allocator.free(queue_families);
        vki.getPhysicalDeviceQueueFamilyProperties(selected_device, &queue_family_count, queue_families.ptr);

        var compute_family: ?u32 = null;
        for (queue_families[0..queue_family_count], 0..) |qf, i| {
            if (qf.queue_flags.compute_bit) {
                compute_family = @intCast(i);
                break;
            }
        }

        const queue_family_index = compute_family orelse {
            log.err("no compute queue family found", .{});
            return error.VulkanUnavailable;
        };

        // Check for device extensions
        var has_nv_coop_matrix = false;
        var has_khr_coop_matrix = false;
        var has_16bit_storage = false;
        var has_khr_integer_dot_product = false;
        var has_subgroup_size_control_ext = false;
        var has_amd_shader_core = false;
        {
            var ext_count: u32 = 0;
            _ = vki.enumerateDeviceExtensionProperties(selected_device, null, &ext_count, null) catch {};
            if (ext_count > 0) {
                if (allocator.alloc(vk.ExtensionProperties, ext_count)) |exts| {
                    defer allocator.free(exts);
                    _ = vki.enumerateDeviceExtensionProperties(selected_device, null, &ext_count, exts.ptr) catch {};
                    for (exts[0..ext_count]) |ext| {
                        const name: [*:0]const u8 = @ptrCast(&ext.extension_name);
                        const name_str = std.mem.sliceTo(name, 0);
                        if (std.mem.eql(u8, name_str, "VK_NV_cooperative_matrix")) {
                            has_nv_coop_matrix = true;
                        }
                        if (std.mem.eql(u8, name_str, "VK_KHR_cooperative_matrix")) {
                            has_khr_coop_matrix = true;
                        }
                        if (std.mem.eql(u8, name_str, "VK_KHR_16bit_storage")) {
                            has_16bit_storage = true;
                        }
                        if (std.mem.eql(u8, name_str, "VK_KHR_shader_integer_dot_product")) {
                            has_khr_integer_dot_product = true;
                        }
                        if (std.mem.eql(u8, name_str, "VK_EXT_subgroup_size_control")) {
                            has_subgroup_size_control_ext = true;
                        }
                        if (std.mem.eql(u8, name_str, "VK_AMD_shader_core_properties")) {
                            has_amd_shader_core = true;
                        }
                    }
                } else |_| {}
            }
        }

        // Query 16-bit storage support (required for FP16 activation buffers)
        if (!has_16bit_storage) {
            log.err("VK_KHR_16bit_storage not supported — required for FP16 activations", .{});
            return error.VulkanUnavailable;
        }

        // Query cooperative matrix properties to check for fp16 support.
        // The coop shaders use fp16 inputs with f32 accumulators (FP16 tensor cores).
        var has_coop_matrix = false;
        var has_coop_matrix_int8 = false;
        if (has_khr_coop_matrix) {
            var prop_count: u32 = 0;
            _ = vki.getPhysicalDeviceCooperativeMatrixPropertiesKHR(selected_device, &prop_count, null) catch {};
            if (prop_count > 0) {
                if (allocator.alloc(vk.CooperativeMatrixPropertiesKHR, prop_count)) |props| {
                    defer allocator.free(props);
                    for (props) |*p| p.* = .{
                        .m_size = 0,
                        .n_size = 0,
                        .k_size = 0,
                        .a_type = .float16_khr,
                        .b_type = .float16_khr,
                        .c_type = .float16_khr,
                        .result_type = .float16_khr,
                        .saturating_accumulation = 0,
                        .scope = .subgroup_khr,
                    };
                    _ = vki.getPhysicalDeviceCooperativeMatrixPropertiesKHR(selected_device, &prop_count, props.ptr) catch {};
                    for (props[0..prop_count]) |prop| {
                        log.info("cooperative matrix: A={} B={} C={} R={} M={} N={} K={} scope={}", .{
                            @intFromEnum(prop.a_type), @intFromEnum(prop.b_type),
                            @intFromEnum(prop.c_type), @intFromEnum(prop.result_type),
                            prop.m_size,               prop.n_size,
                            prop.k_size,               @intFromEnum(prop.scope),
                        });
                        if (prop.a_type == .float16_khr and prop.b_type == .float16_khr) {
                            has_coop_matrix = true;
                        }
                        if (prop.a_type == .sint8_khr and prop.b_type == .sint8_khr) {
                            has_coop_matrix_int8 = true;
                        }
                    }
                } else |_| {}
            }
        }
        if (has_nv_coop_matrix and !has_coop_matrix) {
            log.info("cooperative matrix available but no fp16 support, using scalar path", .{});
        }

        // Check integer dot product support
        var has_integer_dot_product = false;
        if (has_khr_integer_dot_product) {
            // Query whether int8 packed dot product is hardware-accelerated
            var idp_features = vk.PhysicalDeviceShaderIntegerDotProductFeatures{
                .shader_integer_dot_product = vk.FALSE,
            };
            var features2 = vk.PhysicalDeviceFeatures2{
                .p_next = @ptrCast(&idp_features),
                .features = std.mem.zeroes(vk.PhysicalDeviceFeatures),
            };
            vki.getPhysicalDeviceFeatures2(selected_device, &features2);
            if (idp_features.shader_integer_dot_product == vk.TRUE) {
                has_integer_dot_product = true;
                log.info("integer dot product (dotPacked4x8) supported", .{});
            }
        }

        // Query subgroup properties (Vulkan 1.1 core)
        var subgroup_props = vk.PhysicalDeviceSubgroupProperties{
            .subgroup_size = 0,
            .supported_stages = .{},
            .supported_operations = .{},
            .quad_operations_in_all_stages = vk.FALSE,
        };
        var subgroup_size_ctrl_props = vk.PhysicalDeviceSubgroupSizeControlProperties{
            .p_next = @ptrCast(&subgroup_props),
            .min_subgroup_size = 0,
            .max_subgroup_size = 0,
            .max_compute_workgroup_subgroups = 0,
            .required_subgroup_size_stages = .{},
        };
        var amd_core_props = vk.PhysicalDeviceShaderCorePropertiesAMD{
            .p_next = if (has_subgroup_size_control_ext) @ptrCast(&subgroup_size_ctrl_props) else @ptrCast(&subgroup_props),
            .shader_engine_count = 0,
            .shader_arrays_per_engine_count = 0,
            .compute_units_per_shader_array = 0,
            .simd_per_compute_unit = 0,
            .wavefronts_per_simd = 0,
            .wavefront_size = 0,
            .sgprs_per_simd = 0,
            .min_sgpr_allocation = 0,
            .max_sgpr_allocation = 0,
            .sgpr_allocation_granularity = 0,
            .vgprs_per_simd = 0,
            .min_vgpr_allocation = 0,
            .max_vgpr_allocation = 0,
            .vgpr_allocation_granularity = 0,
        };
        var props2 = vk.PhysicalDeviceProperties2{
            .p_next = if (has_amd_shader_core) @ptrCast(&amd_core_props) else if (has_subgroup_size_control_ext) @ptrCast(&subgroup_size_ctrl_props) else @ptrCast(&subgroup_props),
            .properties = std.mem.zeroes(vk.PhysicalDeviceProperties),
        };
        vki.getPhysicalDeviceProperties2(selected_device, &props2);
        const subgroup_size = subgroup_props.subgroup_size;
        const min_subgroup_size = if (has_subgroup_size_control_ext) subgroup_size_ctrl_props.min_subgroup_size else subgroup_size;

        // Check subgroup size control feature support
        var has_subgroup_size_control = false;
        if (has_subgroup_size_control_ext) {
            var ssc_features = vk.PhysicalDeviceSubgroupSizeControlFeatures{
                .subgroup_size_control = vk.FALSE,
                .compute_full_subgroups = vk.FALSE,
            };
            var ssc_features2 = vk.PhysicalDeviceFeatures2{
                .p_next = @ptrCast(&ssc_features),
                .features = std.mem.zeroes(vk.PhysicalDeviceFeatures),
            };
            vki.getPhysicalDeviceFeatures2(selected_device, &ssc_features2);
            has_subgroup_size_control = ssc_features.subgroup_size_control == vk.TRUE;
        }

        // Auto-detect compute units per vendor
        const vendor_id = selected_props.vendor_id;
        var compute_units: u32 = blk: {
            if (vendor_id == 0x10DE) { // NVIDIA
                break :blk 48;
            } else if (vendor_id == 0x1002) { // AMD
                if (has_amd_shader_core) {
                    const cu = amd_core_props.shader_engine_count *
                        amd_core_props.shader_arrays_per_engine_count *
                        amd_core_props.compute_units_per_shader_array;
                    if (cu > 0) break :blk cu;
                }
                break :blk 32;
            } else if (vendor_id == 0x8086) { // Intel
                break :blk 32;
            } else {
                break :blk if (selected_props.device_type == .discrete_gpu) @as(u32, 32) else @as(u32, 16);
            }
        };

        // Environment variable override
        if (std.posix.getenv("INFER_COMPUTE_UNITS")) |val| {
            compute_units = std.fmt.parseInt(u32, std.mem.sliceTo(val, 0), 10) catch compute_units;
        }

        const vendor_name: []const u8 = if (vendor_id == 0x10DE) "NVIDIA" else if (vendor_id == 0x1002) "AMD" else if (vendor_id == 0x8086) "Intel" else "unknown";
        log.info("vendor: {s} (0x{X:0>4}), compute_units: {}, subgroup_size: {}, min_subgroup_size: {}, subgroup_size_control: {}", .{
            vendor_name, vendor_id, compute_units, subgroup_size, min_subgroup_size, has_subgroup_size_control,
        });

        // Create logical device
        // NV and KHR cooperative matrix extensions conflict on NVIDIA drivers (~4× perf regression),
        // so enable only one based on caller preference.
        const enable_khr_coop = switch (options.coop_preference) {
            .prefer_khr => has_coop_matrix_int8 and has_khr_coop_matrix,
            .prefer_nv => has_coop_matrix_int8 and has_khr_coop_matrix and !has_nv_coop_matrix,
        };
        const enable_nv_coop = switch (options.coop_preference) {
            .prefer_nv => has_coop_matrix and has_nv_coop_matrix,
            .prefer_khr => has_coop_matrix and has_nv_coop_matrix and !enable_khr_coop,
        };
        var ext_names: [6][*:0]const u8 = undefined;
        var ext_count: u32 = 0;
        ext_names[ext_count] = "VK_KHR_16bit_storage";
        ext_count += 1;
        if (enable_nv_coop) {
            ext_names[ext_count] = "VK_NV_cooperative_matrix";
            ext_count += 1;
        }
        if (enable_khr_coop) {
            ext_names[ext_count] = "VK_KHR_cooperative_matrix";
            ext_count += 1;
        }
        if (has_integer_dot_product) {
            ext_names[ext_count] = "VK_KHR_shader_integer_dot_product";
            ext_count += 1;
        }
        if (has_subgroup_size_control) {
            ext_names[ext_count] = "VK_EXT_subgroup_size_control";
            ext_count += 1;
        }

        // Chain feature structs into pNext
        var khr_coop_matrix_features = vk.PhysicalDeviceCooperativeMatrixFeaturesKHR{
            .p_next = null,
            .cooperative_matrix = if (enable_khr_coop) vk.TRUE else vk.FALSE,
            .cooperative_matrix_robust_buffer_access = vk.FALSE,
        };
        var coop_matrix_features = vk.PhysicalDeviceCooperativeMatrixFeaturesNV{
            .p_next = if (enable_khr_coop) @ptrCast(&khr_coop_matrix_features) else null,
            .cooperative_matrix = if (enable_nv_coop) vk.TRUE else vk.FALSE,
            .cooperative_matrix_robust_buffer_access = vk.FALSE,
        };
        var idp_enable_features = vk.PhysicalDeviceShaderIntegerDotProductFeatures{
            .p_next = if (enable_nv_coop) @ptrCast(&coop_matrix_features) else if (enable_khr_coop) @ptrCast(&khr_coop_matrix_features) else null,
            .shader_integer_dot_product = if (has_integer_dot_product) vk.TRUE else vk.FALSE,
        };
        var storage_16bit_features = vk.PhysicalDevice16BitStorageFeatures{
            .p_next = if (has_integer_dot_product or enable_nv_coop) @ptrCast(&idp_enable_features) else null,
            .storage_buffer_16_bit_access = vk.TRUE,
            .uniform_and_storage_buffer_16_bit_access = vk.FALSE,
            .storage_push_constant_16 = vk.FALSE,
            .storage_input_output_16 = vk.FALSE,
        };
        var subgroup_size_ctrl_features = vk.PhysicalDeviceSubgroupSizeControlFeatures{
            .p_next = @ptrCast(&storage_16bit_features),
            .subgroup_size_control = if (has_subgroup_size_control) vk.TRUE else vk.FALSE,
            .compute_full_subgroups = vk.FALSE,
        };

        // Enable shaderFloat64 for the f64 sum-of-squares / softmax accumulators
        // in rmsNorm and softmax shaders (matches llama.cpp's `ggml_float`).
        // Without explicitly enabling this device feature, drivers may fall back
        // to a slow software-emulation path that uses much more device memory.
        var device_features2 = vk.PhysicalDeviceFeatures2{
            .p_next = @ptrCast(&subgroup_size_ctrl_features),
            .features = blk: {
                var f = std.mem.zeroes(vk.PhysicalDeviceFeatures);
                f.shader_float_64 = vk.TRUE;
                break :blk f;
            },
        };

        const queue_priority = [_]f32{1.0};
        const device = vki.createDevice(selected_device, &.{
            .p_next = @ptrCast(&device_features2),
            .queue_create_info_count = 1,
            .p_queue_create_infos = &[_]vk.DeviceQueueCreateInfo{.{
                .queue_family_index = queue_family_index,
                .queue_count = 1,
                .p_queue_priorities = &queue_priority,
            }},
            .enabled_extension_count = ext_count,
            .pp_enabled_extension_names = &ext_names,
        }, null) catch return error.VulkanUnavailable;

        // Load device dispatch
        const vkd = vk.DeviceWrapper.load(device, vki.dispatch.vkGetDeviceProcAddr.?);

        // Get compute queue
        const compute_queue = vkd.getDeviceQueue(device, queue_family_index, 0);

        // Create command pool
        const command_pool = vkd.createCommandPool(device, &.{
            .flags = .{ .reset_command_buffer_bit = true },
            .queue_family_index = queue_family_index,
        }, null) catch return error.VulkanUnavailable;

        // Create descriptor pool
        const pool_sizes = [_]vk.DescriptorPoolSize{
            .{ .type = .storage_buffer, .descriptor_count = 4096 },
        };
        const descriptor_pool = vkd.createDescriptorPool(device, &.{
            .flags = .{ .free_descriptor_set_bit = true },
            .max_sets = 2048,
            .pool_size_count = pool_sizes.len,
            .p_pool_sizes = &pool_sizes,
        }, null) catch return error.VulkanUnavailable;

        // Find memory types
        const mem_props = vki.getPhysicalDeviceMemoryProperties(selected_device);

        const device_local_type = findMemoryType(&mem_props, 0xFFFFFFFF, .{ .device_local_bit = true }) orelse {
            log.err("no device-local memory type found", .{});
            return error.VulkanUnavailable;
        };

        const host_visible_type = findMemoryType(&mem_props, 0xFFFFFFFF, .{ .host_visible_bit = true, .host_coherent_bit = true }) orelse {
            log.err("no host-visible coherent memory type found", .{});
            return error.VulkanUnavailable;
        };

        // Prefer cached host memory for readback buffers (avoids uncached PCIe reads)
        const host_cached_type = findMemoryType(&mem_props, 0xFFFFFFFF, .{ .host_visible_bit = true, .host_coherent_bit = true, .host_cached_bit = true }) orelse host_visible_type;

        const self = try allocator.create(Self);
        self.* = .{
            .allocator = allocator,
            .vki = vki,
            .vkd = vkd,
            .instance = instance,
            .physical_device = selected_device,
            .device = device,
            .compute_queue = compute_queue,
            .queue_family_index = queue_family_index,
            .command_pool = command_pool,
            .descriptor_pool = descriptor_pool,
            .device_local_memory_type = device_local_type,
            .host_visible_memory_type = host_visible_type,
            .host_cached_memory_type = host_cached_type,
            .max_workgroup_size = selected_props.limits.max_compute_work_group_size[0],
            .has_coop_matrix = enable_nv_coop,
            .has_coop_matrix_int8 = enable_khr_coop,
            .has_integer_dot_product = has_integer_dot_product,
            .timestamp_period = selected_props.limits.timestamp_period,
            .compute_units = compute_units,
            .subgroup_size = subgroup_size,
            .min_subgroup_size = min_subgroup_size,
            .has_subgroup_size_control = has_subgroup_size_control,
            .allocated_bytes = 0,
            .upload_staging = null,
            .vk_lib = vk_lib,
        };

        return self;
    }

    pub fn deinit(self: *Self) void {
        self.vkd.deviceWaitIdle(self.device) catch {};
        if (self.upload_staging) |s| self.destroyBuffer(s);
        self.vkd.destroyDescriptorPool(self.device, self.descriptor_pool, null);
        self.vkd.destroyCommandPool(self.device, self.command_pool, null);
        self.vkd.destroyDevice(self.device, null);
        self.vki.destroyInstance(self.instance, null);
        self.vk_lib.close();
        self.allocator.destroy(self);
    }

    // --- Buffer management ---

    pub fn createBuffer(self: *Self, size: vk.DeviceSize, usage: vk.BufferUsageFlags, device_local: bool) !GpuBuffer {
        const buffer = try self.vkd.createBuffer(self.device, &.{
            .size = size,
            .usage = usage,
            .sharing_mode = .exclusive,
        }, null);
        errdefer self.vkd.destroyBuffer(self.device, buffer, null);

        const mem_requirements = self.vkd.getBufferMemoryRequirements(self.device, buffer);

        const preferred_type: u32 = if (device_local) self.device_local_memory_type else self.host_visible_memory_type;
        const mem_type_index = findMemoryType2(&mem_requirements, preferred_type) orelse preferred_type;

        const memory = try self.vkd.allocateMemory(self.device, &.{
            .allocation_size = mem_requirements.size,
            .memory_type_index = mem_type_index,
        }, null);
        errdefer self.vkd.freeMemory(self.device, memory, null);

        try self.vkd.bindBufferMemory(self.device, buffer, memory, 0);

        // Map host-visible memory
        var mapped: ?[*]u8 = null;
        if (!device_local) {
            const raw_ptr = try self.vkd.mapMemory(self.device, memory, 0, size, .{});
            mapped = @ptrCast(raw_ptr);
        }

        self.allocated_bytes += size;

        return GpuBuffer{
            .buffer = buffer,
            .memory = memory,
            .size = size,
            .mapped = mapped,
        };
    }

    /// Creates a host-cached buffer for GPU→CPU readback (cached reads, avoids uncached PCIe penalties).
    pub fn createReadbackBuffer(self: *Self, size: vk.DeviceSize) !GpuBuffer {
        const buffer = try self.vkd.createBuffer(self.device, &.{
            .size = size,
            .usage = .{ .transfer_dst_bit = true },
            .sharing_mode = .exclusive,
        }, null);
        errdefer self.vkd.destroyBuffer(self.device, buffer, null);

        const mem_requirements = self.vkd.getBufferMemoryRequirements(self.device, buffer);
        const mem_type_index = findMemoryType2(&mem_requirements, self.host_cached_memory_type) orelse self.host_cached_memory_type;

        const memory = try self.vkd.allocateMemory(self.device, &.{
            .allocation_size = mem_requirements.size,
            .memory_type_index = mem_type_index,
        }, null);
        errdefer self.vkd.freeMemory(self.device, memory, null);

        try self.vkd.bindBufferMemory(self.device, buffer, memory, 0);

        const raw_ptr = try self.vkd.mapMemory(self.device, memory, 0, size, .{});
        const mapped: [*]u8 = @ptrCast(raw_ptr);

        self.allocated_bytes += size;

        return GpuBuffer{ .buffer = buffer, .memory = memory, .size = size, .mapped = mapped };
    }

    pub fn destroyBuffer(self: *Self, buf: GpuBuffer) void {
        if (buf.mapped != null) {
            self.vkd.unmapMemory(self.device, buf.memory);
        }
        self.vkd.destroyBuffer(self.device, buf.buffer, null);
        self.vkd.freeMemory(self.device, buf.memory, null);
        self.allocated_bytes -|= buf.size;
    }

    pub fn uploadToBuffer(self: *Self, dst: GpuBuffer, data: []const u8) !void {
        const staging = try self.ensureUploadStaging(data.len);

        @memcpy(staging.mapped.?[0..data.len], data);

        const cmd = try self.createCommandBuffer();
        try self.beginCommandBuffer(cmd);
        self.vkd.cmdCopyBuffer(cmd, staging.buffer, dst.buffer, 1, &[_]vk.BufferCopy{.{
            .src_offset = 0,
            .dst_offset = 0,
            .size = data.len,
        }});
        try self.endCommandBuffer(cmd);
        try self.submitAndWait(cmd);
        self.freeCommandBuffer(cmd);
    }

    // Returns a persistent host-visible staging buffer at least `size`
    // bytes. Reuses across `uploadToBuffer` calls so weight upload for a
    // multi-GB model doesn't thrash the Vulkan allocator with 40+ tiny
    // alloc/free cycles, and the peak-during-upload footprint stays
    // bounded by the largest single tensor instead of cumulative.
    fn ensureUploadStaging(self: *Self, size: vk.DeviceSize) !GpuBuffer {
        if (self.upload_staging) |existing| {
            if (existing.size >= size) return existing;
            // Grow: wait for any in-flight copies before freeing.
            self.vkd.deviceWaitIdle(self.device) catch {};
            self.destroyBuffer(existing);
            self.upload_staging = null;
        }
        const fresh = try self.createBuffer(size, .{ .transfer_src_bit = true }, false);
        self.upload_staging = fresh;
        return fresh;
    }

    pub fn downloadFromBuffer(self: *Self, src: GpuBuffer, dst: []u8) !void {
        const staging = try self.createBuffer(
            dst.len,
            .{ .transfer_dst_bit = true },
            false,
        );
        defer self.destroyBuffer(staging);

        const cmd = try self.createCommandBuffer();
        try self.beginCommandBuffer(cmd);
        self.vkd.cmdCopyBuffer(cmd, src.buffer, staging.buffer, 1, &[_]vk.BufferCopy{.{
            .src_offset = 0,
            .dst_offset = 0,
            .size = dst.len,
        }});
        try self.endCommandBuffer(cmd);
        try self.submitAndWait(cmd);
        self.freeCommandBuffer(cmd);

        @memcpy(dst, staging.mapped.?[0..dst.len]);
    }

    // --- Command buffer helpers ---

    pub fn createCommandBuffer(self: *Self) !vk.CommandBuffer {
        var cmd: [1]vk.CommandBuffer = undefined;
        try self.vkd.allocateCommandBuffers(self.device, &.{
            .command_pool = self.command_pool,
            .level = .primary,
            .command_buffer_count = 1,
        }, &cmd);
        return cmd[0];
    }

    pub fn freeCommandBuffer(self: *Self, cmd: vk.CommandBuffer) void {
        self.vkd.freeCommandBuffers(self.device, self.command_pool, 1, &[_]vk.CommandBuffer{cmd});
    }

    pub fn beginCommandBuffer(self: *Self, cmd: vk.CommandBuffer) !void {
        try self.vkd.beginCommandBuffer(cmd, &.{
            .flags = .{ .one_time_submit_bit = true },
        });
    }

    pub fn endCommandBuffer(self: *Self, cmd: vk.CommandBuffer) !void {
        try self.vkd.endCommandBuffer(cmd);
    }

    pub fn createFence(self: *Self, signaled: bool) !vk.Fence {
        return self.vkd.createFence(self.device, &.{
            .flags = if (signaled) .{ .signaled_bit = true } else .{},
        }, null);
    }

    pub fn destroyFence(self: *Self, fence: vk.Fence) void {
        self.vkd.destroyFence(self.device, fence, null);
    }

    pub fn submitWithFence(self: *Self, cmd: vk.CommandBuffer, fence: vk.Fence) !void {
        const cmds = [_]vk.CommandBuffer{cmd};
        try self.vkd.resetFences(self.device, 1, &[_]vk.Fence{fence});
        try self.vkd.queueSubmit(self.compute_queue, 1, &[_]vk.SubmitInfo{.{
            .command_buffer_count = 1,
            .p_command_buffers = &cmds,
        }}, fence);
    }

    pub fn waitForFence(self: *Self, fence: vk.Fence) !void {
        // Spin-wait with short timeouts to avoid OS sleep latency (~0.1 ms on Linux)
        const fences = [_]vk.Fence{fence};
        var iterations: u32 = 0;
        while (iterations < 50_000_000) : (iterations += 1) { // ~5s at 100ns
            const result = try self.vkd.waitForFences(self.device, 1, &fences, vk.TRUE, 100); // 100 ns timeout
            if (result == .success) return;
        }
        log.err("waitForFence timed out after ~5 seconds", .{});
        return error.VulkanError;
    }

    pub fn submitAndWait(self: *Self, cmd: vk.CommandBuffer) !void {
        const fence = try self.createFence(false);
        defer self.destroyFence(fence);
        try self.submitWithFence(cmd, fence);
        try self.waitForFence(fence);
    }

    // --- Shader/Pipeline helpers ---

    pub fn createShaderModule(self: *Self, spirv: []const u8) !vk.ShaderModule {
        // SPIR-V embedded via @embedFile may not be u32-aligned.
        // Always copy to an aligned buffer.
        const word_count = spirv.len / 4;
        const buf = self.allocator.alloc(u32, word_count) catch return error.OutOfHostMemory;
        defer self.allocator.free(buf);
        @memcpy(std.mem.sliceAsBytes(buf), spirv[0 .. word_count * 4]);

        return self.vkd.createShaderModule(self.device, &.{
            .code_size = spirv.len,
            .p_code = buf.ptr,
        }, null);
    }

    pub fn createPipelineLayout(self: *Self, set_layouts: []const vk.DescriptorSetLayout, push_constant_size: u32) !vk.PipelineLayout {
        const push_range = vk.PushConstantRange{
            .stage_flags = .{ .compute_bit = true },
            .offset = 0,
            .size = push_constant_size,
        };

        return self.vkd.createPipelineLayout(self.device, &.{
            .set_layout_count = @intCast(set_layouts.len),
            .p_set_layouts = if (set_layouts.len > 0) set_layouts.ptr else null,
            .push_constant_range_count = if (push_constant_size > 0) 1 else 0,
            .p_push_constant_ranges = if (push_constant_size > 0) @ptrCast(&push_range) else null,
        }, null);
    }

    pub fn createComputePipeline(self: *Self, shader_module: vk.ShaderModule, layout: vk.PipelineLayout) !vk.Pipeline {
        var pipeline: [1]vk.Pipeline = undefined;
        _ = try self.vkd.createComputePipelines(self.device, .null_handle, 1, &[_]vk.ComputePipelineCreateInfo{.{
            .stage = .{
                .stage = .{ .compute_bit = true },
                .module = shader_module,
                .p_name = "main",
            },
            .layout = layout,
            .base_pipeline_index = -1,
        }}, null, &pipeline);
        return pipeline[0];
    }

    pub fn createComputePipelineSubgroup32(self: *Self, shader_module: vk.ShaderModule, layout: vk.PipelineLayout) !vk.Pipeline {
        var required_subgroup_size = vk.PipelineShaderStageRequiredSubgroupSizeCreateInfo{
            .required_subgroup_size = 32,
        };
        var pipeline: [1]vk.Pipeline = undefined;
        _ = try self.vkd.createComputePipelines(self.device, .null_handle, 1, &[_]vk.ComputePipelineCreateInfo{.{
            .stage = .{
                .p_next = @ptrCast(&required_subgroup_size),
                .stage = .{ .compute_bit = true },
                .module = shader_module,
                .p_name = "main",
            },
            .layout = layout,
            .base_pipeline_index = -1,
        }}, null, &pipeline);
        return pipeline[0];
    }

    pub fn createDescriptorSetLayout(self: *Self, binding_count: u32) !vk.DescriptorSetLayout {
        var bindings: [8]vk.DescriptorSetLayoutBinding = undefined;
        for (0..binding_count) |i| {
            bindings[i] = .{
                .binding = @intCast(i),
                .descriptor_type = .storage_buffer,
                .descriptor_count = 1,
                .stage_flags = .{ .compute_bit = true },
            };
        }

        return self.vkd.createDescriptorSetLayout(self.device, &.{
            .binding_count = binding_count,
            .p_bindings = &bindings,
        }, null);
    }

    pub fn allocateDescriptorSet(self: *Self, layout: vk.DescriptorSetLayout) !vk.DescriptorSet {
        var ds: [1]vk.DescriptorSet = undefined;
        const layouts = [_]vk.DescriptorSetLayout{layout};
        try self.vkd.allocateDescriptorSets(self.device, &.{
            .descriptor_pool = self.descriptor_pool,
            .descriptor_set_count = 1,
            .p_set_layouts = &layouts,
        }, &ds);
        return ds[0];
    }

    pub fn updateDescriptorSet(self: *Self, descriptor_set: vk.DescriptorSet, binding: u32, buffer: vk.Buffer, size: vk.DeviceSize) void {
        self.updateDescriptorSetOffset(descriptor_set, binding, buffer, 0, size);
    }

    pub fn updateDescriptorSetOffset(self: *Self, descriptor_set: vk.DescriptorSet, binding: u32, buffer: vk.Buffer, offset: vk.DeviceSize, range: vk.DeviceSize) void {
        const buffer_info = [_]vk.DescriptorBufferInfo{.{
            .buffer = buffer,
            .offset = offset,
            .range = range,
        }};
        self.vkd.updateDescriptorSets(self.device, 1, &[_]vk.WriteDescriptorSet{.{
            .dst_set = descriptor_set,
            .dst_binding = binding,
            .dst_array_element = 0,
            .descriptor_count = 1,
            .descriptor_type = .storage_buffer,
            .p_image_info = undefined,
            .p_buffer_info = &buffer_info,
            .p_texel_buffer_view = undefined,
        }}, 0, null);
    }

    // --- Command recording helpers ---

    pub fn cmdBindPipeline(self: *Self, cmd: vk.CommandBuffer, pipeline: vk.Pipeline) void {
        self.vkd.cmdBindPipeline(cmd, .compute, pipeline);
    }

    pub fn cmdBindDescriptorSet(self: *Self, cmd: vk.CommandBuffer, layout: vk.PipelineLayout, set: vk.DescriptorSet) void {
        const sets = [_]vk.DescriptorSet{set};
        self.vkd.cmdBindDescriptorSets(cmd, .compute, layout, 0, 1, &sets, 0, null);
    }

    pub fn cmdBindDescriptorSetAt(self: *Self, cmd: vk.CommandBuffer, layout: vk.PipelineLayout, set_index: u32, set: vk.DescriptorSet) void {
        const sets = [_]vk.DescriptorSet{set};
        self.vkd.cmdBindDescriptorSets(cmd, .compute, layout, set_index, 1, &sets, 0, null);
    }

    pub fn cmdPushConstants(self: *Self, cmd: vk.CommandBuffer, layout: vk.PipelineLayout, data: []const u8) void {
        self.vkd.cmdPushConstants(cmd, layout, .{ .compute_bit = true }, 0, @intCast(data.len), data.ptr);
    }

    pub fn cmdDispatch(self: *Self, cmd: vk.CommandBuffer, group_count_x: u32, group_count_y: u32, group_count_z: u32) void {
        self.vkd.cmdDispatch(cmd, group_count_x, group_count_y, group_count_z);
    }

    pub fn cmdComputeBarrier(self: *Self, cmd: vk.CommandBuffer) void {
        const barrier = vk.MemoryBarrier{
            .src_access_mask = .{ .shader_write_bit = true },
            .dst_access_mask = .{ .shader_read_bit = true },
        };
        self.vkd.cmdPipelineBarrier(
            cmd,
            .{ .compute_shader_bit = true },
            .{ .compute_shader_bit = true },
            .{},
            1,
            @ptrCast(&barrier),
            0,
            null,
            0,
            null,
        );
    }

    pub fn cmdTransferToComputeBarrier(self: *Self, cmd: vk.CommandBuffer) void {
        const barrier = vk.MemoryBarrier{
            .src_access_mask = .{ .transfer_write_bit = true },
            .dst_access_mask = .{ .shader_read_bit = true },
        };
        self.vkd.cmdPipelineBarrier(
            cmd,
            .{ .transfer_bit = true },
            .{ .compute_shader_bit = true },
            .{},
            1,
            @ptrCast(&barrier),
            0,
            null,
            0,
            null,
        );
    }

    pub fn cmdComputeToTransferBarrier(self: *Self, cmd: vk.CommandBuffer) void {
        const barrier = vk.MemoryBarrier{
            .src_access_mask = .{ .shader_write_bit = true },
            .dst_access_mask = .{ .transfer_read_bit = true },
        };
        self.vkd.cmdPipelineBarrier(
            cmd,
            .{ .compute_shader_bit = true },
            .{ .transfer_bit = true },
            .{},
            1,
            @ptrCast(&barrier),
            0,
            null,
            0,
            null,
        );
    }

    pub fn cmdBufferBarrier(self: *Self, cmd: vk.CommandBuffer, buffer: vk.Buffer, size: vk.DeviceSize, src_stage: vk.PipelineStageFlags, dst_stage: vk.PipelineStageFlags, src_access: vk.AccessFlags, dst_access: vk.AccessFlags) void {
        const barrier = vk.BufferMemoryBarrier{
            .src_access_mask = src_access,
            .dst_access_mask = dst_access,
            .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .buffer = buffer,
            .offset = 0,
            .size = size,
        };
        self.vkd.cmdPipelineBarrier(
            cmd,
            src_stage,
            dst_stage,
            .{},
            0,
            null,
            1,
            @ptrCast(&barrier),
            0,
            null,
        );
    }

    pub fn cmdCopyBuffer(self: *Self, cmd: vk.CommandBuffer, src: vk.Buffer, dst: vk.Buffer, size: vk.DeviceSize) void {
        self.vkd.cmdCopyBuffer(cmd, src, dst, 1, &[_]vk.BufferCopy{.{
            .src_offset = 0,
            .dst_offset = 0,
            .size = size,
        }});
    }

    pub fn cmdCopyBufferOffset(self: *Self, cmd: vk.CommandBuffer, src: vk.Buffer, dst: vk.Buffer, src_offset: vk.DeviceSize, dst_offset: vk.DeviceSize, size: vk.DeviceSize) void {
        self.vkd.cmdCopyBuffer(cmd, src, dst, 1, &[_]vk.BufferCopy{.{
            .src_offset = src_offset,
            .dst_offset = dst_offset,
            .size = size,
        }});
    }

    pub fn cmdFillBuffer(self: *Self, cmd: vk.CommandBuffer, buffer: vk.Buffer, size: vk.DeviceSize) void {
        self.vkd.cmdFillBuffer(cmd, buffer, 0, size, 0);
    }

    pub fn cmdFillBufferOffset(self: *Self, cmd: vk.CommandBuffer, buffer: vk.Buffer, offset: vk.DeviceSize, size: vk.DeviceSize) void {
        self.vkd.cmdFillBuffer(cmd, buffer, offset, size, 0);
    }

    // --- Timestamp query helpers ---

    pub fn createTimestampQueryPool(self: *Self, count: u32) !vk.QueryPool {
        return self.vkd.createQueryPool(self.device, &.{
            .query_type = .timestamp,
            .query_count = count,
        }, null);
    }

    pub fn destroyQueryPool(self: *Self, pool: vk.QueryPool) void {
        self.vkd.destroyQueryPool(self.device, pool, null);
    }

    pub fn cmdWriteTimestamp(self: *Self, cmd: vk.CommandBuffer, pool: vk.QueryPool, query: u32) void {
        self.vkd.cmdWriteTimestamp(cmd, .{ .compute_shader_bit = true }, pool, query);
    }

    pub fn cmdResetQueryPool(self: *Self, cmd: vk.CommandBuffer, pool: vk.QueryPool, first: u32, count: u32) void {
        self.vkd.cmdResetQueryPool(cmd, pool, first, count);
    }

    pub fn getTimestampResults(self: *Self, pool: vk.QueryPool, count: u32, results: []u64) void {
        _ = self.vkd.getQueryPoolResults(
            self.device,
            pool,
            0,
            count,
            @intCast(count * @sizeOf(u64)),
            @ptrCast(results.ptr),
            @sizeOf(u64),
            .{ .@"64_bit" = true, .wait_bit = true },
        ) catch {};
    }

    pub fn workgroupCount(total: u32) u32 {
        return (total + 255) / 256;
    }
};

// --- Static helpers ---

fn findMemoryType(mem_props: *const vk.PhysicalDeviceMemoryProperties, type_filter: u32, properties: vk.MemoryPropertyFlags) ?u32 {
    for (0..mem_props.memory_type_count) |i| {
        const idx: u5 = @intCast(i);
        if ((type_filter & (@as(u32, 1) << idx)) != 0) {
            const flags = mem_props.memory_types[i].property_flags;
            const props_int: u32 = @bitCast(properties);
            const flags_int: u32 = @bitCast(flags);
            if ((flags_int & props_int) == props_int) {
                return @intCast(i);
            }
        }
    }
    return null;
}

fn findMemoryType2(requirements: *const vk.MemoryRequirements, preferred_type: u32) ?u32 {
    const idx: u5 = @intCast(preferred_type);
    if ((requirements.memory_type_bits & (@as(u32, 1) << idx)) != 0) {
        return preferred_type;
    }
    return null;
}

pub const VulkanError = error{
    VulkanUnavailable,
    VulkanError,
    OutOfMemory,
};

test "init and deinit" {
    const device = VulkanDevice.init(std.testing.allocator, .{}) catch |err| switch (err) {
        error.VulkanUnavailable => return,
        else => return err,
    };
    defer device.deinit();

    try std.testing.expect(device.subgroup_size > 0);
    try std.testing.expect(device.max_workgroup_size > 0);
}
