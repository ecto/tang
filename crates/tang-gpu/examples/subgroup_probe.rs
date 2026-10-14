//! How a Vulkan driver lays invocations out in subgroups, and whether an
//! XOR shuffle by `off <= 16` stays inside an aligned run of 32 local
//! invocations (what tang's GLSL decode kernels assume). Needs `--features
//! glsl`. Prints, for one 256-wide workgroup: subgroup size, which local
//! ids share a subgroup, and the first local id whose
//! `subgroupShuffleXor(local_id, off)` is not `local_id ^ off`.
#[cfg(feature = "glsl")]
fn main() {
    use wgpu::util::DeviceExt;
    const SRC: &str = r#"#version 450
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_shuffle : require
layout(local_size_x = 256) in;
layout(std430, set = 0, binding = 0) buffer Out { uint o[]; };
void main() {
    uint t = gl_LocalInvocationID.x;
    o[t * 8 + 0] = gl_SubgroupSize;
    o[t * 8 + 1] = gl_SubgroupID;
    o[t * 8 + 2] = gl_SubgroupInvocationID;
    for (uint k = 0; k < 5; k++) {
        o[t * 8 + 3 + k] = subgroupShuffleXor(t, 16u >> k);
    }
}
"#;
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&Default::default())).unwrap();
    println!("{:?}", adapter.get_info().name);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::SPIRV_SHADER_PASSTHROUGH,
        ..Default::default()
    }))
    .unwrap();
    let compiler = shaderc::Compiler::new().unwrap();
    let mut opts = shaderc::CompileOptions::new().unwrap();
    opts.set_target_env(
        shaderc::TargetEnv::Vulkan,
        shaderc::EnvVersion::Vulkan1_1 as u32,
    );
    let spv = compiler
        .compile_into_spirv(
            SRC,
            shaderc::ShaderKind::Compute,
            "p.comp",
            "main",
            Some(&opts),
        )
        .unwrap();
    let module = unsafe {
        device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough::SpirV(
            wgpu::ShaderModuleDescriptorSpirV {
                label: None,
                source: spv.as_binary().into(),
            },
        ))
    };
    let n = 256 * 8;
    let out = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (n * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: &vec![0u8; n * 4],
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: out.as_entire_binding(),
        }],
    });
    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipe);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    enc.copy_buffer_to_buffer(&out, 0, &read, 0, (n * 4) as u64);
    queue.submit([enc.finish()]);
    read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::Wait).unwrap();
    let v: Vec<u32> = bytemuck::cast_slice(&read.slice(..).get_mapped_range()).to_vec();
    let at = |t: usize, k: usize| v[t * 8 + k];
    println!("subgroup size {}", at(0, 0));
    let mut groups: std::collections::BTreeMap<u32, Vec<u32>> = Default::default();
    for t in 0..256 {
        groups.entry(at(t, 1)).or_default().push(t as u32);
    }
    for (g, ts) in &groups {
        let contiguous = ts.windows(2).all(|w| w[1] == w[0] + 1);
        println!(
            "  subgroup {g}: {} invocations, local ids {}..={} contiguous {contiguous}",
            ts.len(),
            ts[0],
            ts[ts.len() - 1]
        );
    }
    for k in 0..5 {
        let off = 16u32 >> k;
        let bad = (0..256u32).find(|&t| at(t as usize, 3 + k) != t ^ off);
        println!(
            "  shuffle xor {off}: {}",
            match bad {
                None => "all lanes match t ^ off".to_string(),
                Some(t) => format!("lane {t} got {} (want {})", at(t as usize, 3 + k), t ^ off),
            }
        );
    }
}

#[cfg(not(feature = "glsl"))]
fn main() {
    eprintln!("subgroup_probe needs --features glsl");
}
