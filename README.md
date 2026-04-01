# vulkan

Vulkan compute device abstraction for GPU inference.

Provides a `VulkanDevice` that manages instance creation, physical/logical device selection, compute queue, command pools, and descriptor pools. Includes `GpuBuffer` for GPU memory allocation with optional host-mapped access.

## Features

- Dynamic Vulkan library loading (`libvulkan.so.1`)
- Cooperative matrix extension detection (NV and KHR)
- Integer dot product extension support (`VK_KHR_shader_integer_dot_product`)
- Subgroup size control
- Device-local memory tracking
- Configurable cooperative matrix preference (`prefer_nv` / `prefer_khr`)

## Usage

```bash
zig fetch --save git+https://github.com/infer-zero/vulkan
```

Then in your `build.zig`:

```zig
const vulkan_dep = b.dependency("vulkan_device", .{ .target = target, .optimize = optimize });
my_mod.addImport("vulkan_device", vulkan_dep.module("vulkan_device"));
```

```zig
const vk_mod = @import("vulkan_device");

var device = try vk_mod.VulkanDevice.init(allocator, .{
    .coop_preference = .prefer_khr,
});
defer device.deinit();

var buffer = try device.createBuffer(size, .{ .storage_buffer_bit = true }, .{ .device_local_bit = true });
defer device.destroyBuffer(buffer);
```

## Dependencies

- [vulkan-zig](https://github.com/Snektron/vulkan-zig) — Zig Vulkan bindings
- [Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers) — Vulkan registry

No dependency on other packages in this repository.

## License

MIT
