const std = @import("std");

pub const TargetEnv = enum {
    vulkan1_1,
    vulkan1_3,

    fn flag(self: TargetEnv) []const u8 {
        return switch (self) {
            .vulkan1_1 => "--target-env=vulkan1.1",
            .vulkan1_3 => "--target-env=vulkan1.3",
        };
    }
};

pub const ShaderGroups = struct {
    shared: bool = false,
    bf16: bool = false,
    q8_0: bool = false,
    q5_0: bool = false,
    q4_0: bool = false,
    q4_k: bool = false,
    q5_k: bool = false,
    q6_k: bool = false,
    moe_bf16: bool = false,
    moe_q8_0: bool = false,
    moe_q4_k: bool = false,
};

const Shader = struct {
    dir: []const u8,
    source: []const u8,
    output: []const u8,
    target_env: TargetEnv,
    defines: []const []const u8,
};

fn shader(dir: []const u8, name: []const u8) Shader {
    return .{ .dir = dir, .source = name, .output = name, .target_env = .vulkan1_1, .defines = &.{} };
}

fn shaderVk13(dir: []const u8, name: []const u8) Shader {
    return .{ .dir = dir, .source = name, .output = name, .target_env = .vulkan1_3, .defines = &.{} };
}

fn variant(dir: []const u8, source: []const u8, output: []const u8, target_env: TargetEnv, defines: []const []const u8) Shader {
    return .{ .dir = dir, .source = source, .output = output, .target_env = target_env, .defines = defines };
}

const shared_shaders = [_]Shader{
    shader("shaders", "add_vectors"),
    shader("shaders", "copy_embedding"),
    shader("shaders", "flash_attention_causal_tiled"),
    shader("shaders", "flash_attention_decode"),
    shader("shaders", "fused_add_norm"),
    shader("shaders", "fused_add_norm_batch"),
    shader("shaders", "fused_rope_store_bf16"),
    shader("shaders", "fused_rope_store_bf16_batch"),
    shader("shaders", "matmul_f32_logits"),
    shader("shaders", "rms_norm"),
    shader("shaders", "rope"),
    // Variants
    variant("shaders", "copy_embedding", "copy_embedding_batch", .vulkan1_1, &.{"-DBATCH_MODE"}),
    variant("shaders", "add_vectors", "add_vectors_batch", .vulkan1_1, &.{"-DBATCH_MODE"}),
    variant("shaders", "rope", "rope_batch", .vulkan1_1, &.{"-DBATCH_MODE"}),
};

const bf16_shaders = [_]Shader{
    shader("shaders/bf16", "f16_to_bf16"),
    shader("shaders/bf16", "matmul_bf16"),
    shader("shaders/bf16", "matmul_bf16_logits"),
    shader("shaders/bf16", "matmul_silu_hadamard"),
    shader("shaders/bf16", "matmul_bf16_batch"),
    shader("shaders/bf16", "matmul_silu_hadamard_batch"),
    shader("shaders/bf16", "matmul_bf16_batch_coop"),
    shader("shaders/bf16", "matmul_silu_hadamard_batch_coop"),
    // Variants
    variant("shaders/bf16", "f16_to_bf16", "f16_to_bf16_batch", .vulkan1_1, &.{"-DBATCH_MODE"}),
};

const q8_0_shaders = [_]Shader{
    shader("shaders/q8_0", "f16_to_bf16"),
    shader("shaders/q8_0", "matmul_q8_0_logits"),
    shader("shaders/q8_0", "matmul_silu_hadamard_q8_0"),
    shader("shaders/q8_0", "matmul_q8_0_panel_matvec"),
    shader("shaders/q8_0", "matmul_silu_hadamard_q8_0_panel_matvec"),
    shader("shaders/q8_0", "matmul_q8_0_batch"),
    shader("shaders/q8_0", "matmul_silu_hadamard_q8_0_batch"),
    shader("shaders/q8_0", "matmul_q8_0_batch_coop"),
    shader("shaders/q8_0", "matmul_silu_hadamard_q8_0_batch_coop"),
    shader("shaders/q8_0", "rms_norm_batch"),
    shader("shaders/q8_0", "quantize_to_q8"),
    // VK 1.3 (int8 coop)
    shaderVk13("shaders/q8_0", "matmul_q8_0_batch_coop_int8"),
    shaderVk13("shaders/q8_0", "matmul_silu_hadamard_q8_0_batch_coop_int8"),
    shaderVk13("shaders/q8_0", "matmul_q8_0_batch_coop_int8_ksplit"),
    // Variants
    variant("shaders/q8_0", "f16_to_bf16", "f16_to_bf16_batch", .vulkan1_1, &.{"-DBATCH_MODE"}),
    variant("shaders/q8_0", "matmul_q8_0_panel_matvec", "matmul_q8_0_panel_matvec_store_bf16", .vulkan1_1, &.{"-DSTORE_BF16_KV"}),
    variant("shaders/q8_0", "matmul_q8_0_batch", "matmul_q8_0_batch_idp", .vulkan1_3, &.{"-DIDP_MODE"}),
    variant("shaders/q8_0", "matmul_silu_hadamard_q8_0_batch", "matmul_silu_hadamard_q8_0_batch_idp", .vulkan1_3, &.{"-DIDP_MODE"}),
};

const q5_0_shaders = [_]Shader{
    shader("shaders/q5_0", "matmul_q5_0_panel_matvec"),
    shader("shaders/q5_0", "matmul_silu_hadamard_q5_0_panel_matvec"),
    shader("shaders/q5_0", "matmul_q5_0_batch"),
    shader("shaders/q5_0", "matmul_silu_hadamard_q5_0_batch"),
    // Variants
    variant("shaders/q5_0", "matmul_q5_0_panel_matvec", "matmul_q5_0_panel_matvec_store_bf16", .vulkan1_1, &.{"-DSTORE_BF16_KV"}),
};

const q4_0_shaders = [_]Shader{
    shader("shaders/q4_0", "f16_to_bf16"),
    shader("shaders/q4_0", "matmul_q4_0_logits"),
    shader("shaders/q4_0", "matmul_q4_0_panel_matvec"),
    shader("shaders/q4_0", "matmul_silu_hadamard_q4_0_panel_matvec"),
    shader("shaders/q4_0", "matmul_q4_0_batch"),
    shader("shaders/q4_0", "matmul_silu_hadamard_q4_0_batch"),
    shader("shaders/q4_0", "matmul_q4_0_batch_coop"),
    shader("shaders/q4_0", "matmul_silu_hadamard_q4_0_batch_coop"),
    // Variants
    variant("shaders/q4_0", "f16_to_bf16", "f16_to_bf16_batch", .vulkan1_1, &.{"-DBATCH_MODE"}),
    variant("shaders/q4_0", "matmul_q4_0_panel_matvec", "matmul_q4_0_panel_matvec_store_bf16", .vulkan1_1, &.{"-DSTORE_BF16_KV"}),
};

const q4_k_shaders = [_]Shader{
    shader("shaders/q4_k", "f16_to_bf16"),
    shader("shaders/q4_k", "matmul_q4_k_logits"),
    shader("shaders/q4_k", "matmul_q4_k_panel_matvec"),
    shader("shaders/q4_k", "matmul_silu_hadamard_q4_k_panel_matvec"),
    shader("shaders/q4_k", "matmul_q4_k_batch"),
    shader("shaders/q4_k", "matmul_silu_hadamard_q4_k_batch"),
    // Variants
    variant("shaders/q4_k", "f16_to_bf16", "f16_to_bf16_batch", .vulkan1_1, &.{"-DBATCH_MODE"}),
    variant("shaders/q4_k", "matmul_q4_k_panel_matvec", "matmul_q4_k_panel_matvec_store_bf16", .vulkan1_1, &.{"-DSTORE_BF16_KV"}),
};

const q5_k_shaders = [_]Shader{
    shader("shaders/q5_k", "matmul_q5_k_panel_matvec"),
    shader("shaders/q5_k", "matmul_silu_hadamard_q5_k_panel_matvec"),
    shader("shaders/q5_k", "matmul_q5_k_batch"),
    shader("shaders/q5_k", "matmul_silu_hadamard_q5_k_batch"),
    // Variants
    variant("shaders/q5_k", "matmul_q5_k_panel_matvec", "matmul_q5_k_panel_matvec_store_bf16", .vulkan1_1, &.{"-DSTORE_BF16_KV"}),
};

const q6_k_shaders = [_]Shader{
    shader("shaders/q6_k", "matmul_q6_k_panel_matvec"),
    shader("shaders/q6_k", "matmul_q6_k_batch"),
    // Variants
    variant("shaders/q6_k", "matmul_q6_k_panel_matvec", "matmul_q6_k_panel_matvec_store_bf16", .vulkan1_1, &.{"-DSTORE_BF16_KV"}),
};

const moe_bf16_shaders = [_]Shader{
    shader("shaders/moe", "moe_matmul_down"),
    shader("shaders/moe", "moe_matmul_silu_hadamard"),
    shader("shaders/moe", "moe_scaled_add"),
    shader("shaders/moe", "moe_softmax_topk"),
};

const moe_q8_0_shaders = [_]Shader{
    shader("shaders/moe", "moe_matmul_q8_0_panel_matvec"),
    shader("shaders/moe", "moe_matmul_silu_hadamard_q8_0_panel_matvec"),
    shader("shaders/moe", "moe_scaled_add"),
    shader("shaders/moe", "moe_softmax_topk"),
};

const moe_q4_k_shaders = [_]Shader{
    shader("shaders/moe", "moe_matmul_q4_k_panel_matvec"),
    shader("shaders/moe", "moe_matmul_silu_hadamard_q4_k_panel_matvec"),
    shader("shaders/moe", "moe_scaled_add"),
    shader("shaders/moe", "moe_softmax_topk"),
};

/// Add pre-defined shader groups to a module as anonymous SPIR-V imports.
/// Each shader becomes available as `@embedFile(@import("spv_{name}"))`.
pub fn addShaders(
    b: *std.Build,
    mod: *std.Build.Module,
    vk_dep: *std.Build.Dependency,
    groups: ShaderGroups,
) void {
    const all_groups = [_]struct { enabled: bool, shaders: []const Shader }{
        .{ .enabled = groups.shared, .shaders = &shared_shaders },
        .{ .enabled = groups.bf16, .shaders = &bf16_shaders },
        .{ .enabled = groups.q8_0, .shaders = &q8_0_shaders },
        .{ .enabled = groups.q5_0, .shaders = &q5_0_shaders },
        .{ .enabled = groups.q4_0, .shaders = &q4_0_shaders },
        .{ .enabled = groups.q4_k, .shaders = &q4_k_shaders },
        .{ .enabled = groups.q5_k, .shaders = &q5_k_shaders },
        .{ .enabled = groups.q6_k, .shaders = &q6_k_shaders },
        .{ .enabled = groups.moe_bf16, .shaders = &moe_bf16_shaders },
        .{ .enabled = groups.moe_q8_0, .shaders = &moe_q8_0_shaders },
        .{ .enabled = groups.moe_q4_k, .shaders = &moe_q4_k_shaders },
    };

    for (all_groups) |group| {
        if (!group.enabled) continue;
        for (group.shaders) |s| {
            const spv = compileShader(b, vk_dep, b.fmt("{s}/{s}.comp", .{ s.dir, s.source }), s.output, s.target_env, s.defines);
            mod.addAnonymousImport(b.fmt("spv_{s}", .{s.output}), .{ .root_source_file = spv });
        }
    }
}

/// Compile a single GLSL compute shader to SPIR-V.
/// Use this for model-specific shader variants not covered by addShaders.
pub fn compileShader(
    b: *std.Build,
    vk_dep: *std.Build.Dependency,
    source_path: []const u8,
    output_name: []const u8,
    target_env: TargetEnv,
    defines: []const []const u8,
) std.Build.LazyPath {
    const shaderc_dep = vk_dep.builder.dependency("shaderc", .{ .optimize = .ReleaseFast });
    const glslc = shaderc_dep.artifact("glslc");

    const cmd = b.addRunArtifact(glslc);
    cmd.addArgs(&.{ target_env.flag(), "-o" });
    const spv_output = cmd.addOutputFileArg(b.fmt("{s}.spv", .{output_name}));
    for (defines) |def| cmd.addArg(def);
    cmd.addFileArg(vk_dep.path(source_path));
    return spv_output;
}

/// Compile a single shader from any source path and register it as `@import("spv_{name}")`.
/// Use for model-specific shaders or shared shaders with custom defines not covered by addShaders.
pub fn addShader(
    b: *std.Build,
    mod: *std.Build.Module,
    vk_dep: *std.Build.Dependency,
    source: std.Build.LazyPath,
    name: []const u8,
    target_env: TargetEnv,
    defines: []const []const u8,
) void {
    const shaderc_dep = vk_dep.builder.dependency("shaderc", .{ .optimize = .ReleaseFast });
    const glslc = shaderc_dep.artifact("glslc");
    const cmd = b.addRunArtifact(glslc);
    cmd.addArgs(&.{ target_env.flag(), "-o" });
    const spv = cmd.addOutputFileArg(b.fmt("{s}.spv", .{name}));
    for (defines) |def| cmd.addArg(def);
    cmd.addFileArg(source);
    mod.addAnonymousImport(b.fmt("spv_{s}", .{name}), .{ .root_source_file = spv });
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const vulkan_dep = b.dependency("vulkan", .{
        .registry = b.dependency("vulkan_headers", .{}).path("registry/vk.xml"),
    });
    const vulkan = vulkan_dep.module("vulkan-zig");

    const mod = b.addModule("vulkan_device", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    mod.addImport("vulkan", vulkan);

    const tests = b.addTest(.{
        .root_module = mod,
    });

    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_tests.step);

    // Compile all shaders to validate GLSL correctness
    const validate_step = b.step("validate-shaders", "Compile all shaders to SPIR-V");
    const shaderc_dep = b.dependency("shaderc", .{ .optimize = .ReleaseFast });
    const glslc = shaderc_dep.artifact("glslc");
    const all_groups = [_][]const Shader{
        &shared_shaders,
        &bf16_shaders,
        &q8_0_shaders,
        &q5_0_shaders,
        &q4_0_shaders,
        &q4_k_shaders,
        &q6_k_shaders,
        &moe_bf16_shaders,
        &moe_q8_0_shaders,
        &moe_q4_k_shaders,
    };
    for (all_groups) |group| {
        for (group) |s| {
            const cmd = b.addRunArtifact(glslc);
            cmd.addArgs(&.{ s.target_env.flag(), "-o" });
            _ = cmd.addOutputFileArg(b.fmt("{s}.spv", .{s.output}));
            for (s.defines) |def| cmd.addArg(def);
            cmd.addFileArg(b.path(b.fmt("{s}/{s}.comp", .{ s.dir, s.source })));
            validate_step.dependOn(&cmd.step);
        }
    }
}
