# vulkan-device

Vulkan compute device abstraction for GPU inference in Zig.

This package wraps Vulkan instance + device + compute queue + descriptor
pool setup behind a single `VulkanDevice` so model variants can focus on
shaders and buffers. It is the common GPU substrate every Vulkan variant
in the `infer` workspace builds on, with no dependency on the runtime
package or any model code.

## Parts

### Device

This is the entry point most callers will touch.

- VulkanDevice: instance + physical/logical device + compute queue +
  command pool + descriptor pool, with dynamic loading of
  `libvulkan.so.1` and feature/extension probing baked in.
- InitOptions: knobs at `init` time, including `coop_preference`
  (`prefer_nv` / `prefer_khr` for the cooperative-matrix path) and the
  optional subgroup-size override.

### Buffers

- GpuBuffer: device-local GPU memory allocation with optional
  host-visible mapping for upload/readback. Created via
  `device.createBuffer(...)`, freed via `device.destroyBuffer(...)`.
- Readback helpers: `createReadbackBuffer` returns a
  `HOST_VISIBLE | HOST_COHERENT | HOST_CACHED` buffer (uncached PCIe reads
  are catastrophic — always use `HOST_CACHED` for logits readback).

### Capabilities

These flags surface what the runtime device actually supports so a
shader pipeline can pick the fastest path it can compile.

- Cooperative matrix detection: NV (`VK_NV_cooperative_matrix`) and KHR
  (`VK_KHR_cooperative_matrix`), reported separately so callers can
  enable one or the other (enabling both regresses ~4×).
- Integer dot product: `VK_KHR_shader_integer_dot_product` for INT8 GEMM.
- Subgroup size + supported coop-matrix shapes for the active backend.

## Usage

Fetch the library:

```bash
zig fetch --save git+https://github.com/infer-zero/vulkan
```

Add the dependency in your `build.zig`:

```zig
const vulkan_dep = b.dependency("vulkan_device", .{ .target = target, .optimize = optimize });
my_mod.addImport("vulkan_device", vulkan_dep.module("vulkan_device"));
```

`vulkan-zig` and the Vulkan headers come transitively — callers do not
need to declare them. The `vk` namespace is re-exported from the package
root for convenience:

```zig
const vk_mod = @import("vulkan_device");
const vk = vk_mod.vk;

var device = try vk_mod.VulkanDevice.init(allocator, .{
    .coop_preference = .prefer_khr,
});
defer device.deinit();

var buffer = try device.createBuffer(
    size,
    .{ .storage_buffer_bit = true },
    .{ .device_local_bit = true },
);
defer device.destroyBuffer(buffer);
```

## AI Usage

- The first full version of this library was hand written.
- Some helpers, fixes and Zig version migrations were AI assisted.
- Comments and docs were AI written and human edited.
- All was human reviewed.
- The design and the device/buffer split is my own.

## License

MIT
