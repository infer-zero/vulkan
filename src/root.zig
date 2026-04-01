const vulkan_device = @import("vulkan_device.zig");
pub const mem = @import("mem.zig");

// Re-export vulkan_device public API
pub const vk = vulkan_device.vk;
pub const VulkanDevice = vulkan_device.VulkanDevice;
pub const GpuBuffer = vulkan_device.GpuBuffer;
pub const CoopPreference = vulkan_device.CoopPreference;
pub const InitOptions = vulkan_device.InitOptions;
pub const VulkanError = vulkan_device.VulkanError;

// Re-export vulkan types
pub const Buffer = vulkan_device.Buffer;
pub const DeviceMemory = vulkan_device.DeviceMemory;
pub const DeviceSize = vulkan_device.DeviceSize;
pub const CommandBuffer = vulkan_device.CommandBuffer;
pub const Pipeline = vulkan_device.Pipeline;
pub const PipelineLayout = vulkan_device.PipelineLayout;
pub const DescriptorSet = vulkan_device.DescriptorSet;
pub const DescriptorSetLayout = vulkan_device.DescriptorSetLayout;
pub const DescriptorPool = vulkan_device.DescriptorPool;
pub const ShaderModule = vulkan_device.ShaderModule;
pub const BufferUsageFlags = vulkan_device.BufferUsageFlags;
pub const MemoryBarrier = vulkan_device.MemoryBarrier;


test {
    _ = vulkan_device;
    _ = mem;
}
