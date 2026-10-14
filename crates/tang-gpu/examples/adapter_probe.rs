//! Print what wgpu's adapter offers: backend, limits that matter for large
//! weight buffers, and subgroup features.
fn main() {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    for a in instance.enumerate_adapters(wgpu::Backends::all()) {
        let i = a.get_info();
        let l = a.limits();
        let f = a.features();
        println!(
            "{} ({:?}, {:?}) driver {} {}",
            i.name, i.backend, i.device_type, i.driver, i.driver_info
        );
        println!("  max_buffer_size {} MiB", l.max_buffer_size >> 20);
        println!(
            "  max_storage_buffer_binding_size {} MiB",
            l.max_storage_buffer_binding_size >> 20
        );
        println!(
            "  max_storage_buffers_per_shader_stage {}",
            l.max_storage_buffers_per_shader_stage
        );
        println!(
            "  max_compute_workgroup_storage_size {} KiB",
            l.max_compute_workgroup_storage_size >> 10
        );
        println!(
            "  max_compute_invocations_per_workgroup {}",
            l.max_compute_invocations_per_workgroup
        );
        println!(
            "  subgroup sizes {}..{}",
            l.min_subgroup_size, l.max_subgroup_size
        );
        println!(
            "  SUBGROUP {}  SHADER_F16 {}  SHADER_INT64 {}",
            f.contains(wgpu::Features::SUBGROUP),
            f.contains(wgpu::Features::SHADER_F16),
            f.contains(wgpu::Features::SHADER_INT64)
        );
        println!(
            "  SPIRV_SHADER_PASSTHROUGH {}  SUBGROUP_BARRIER {}",
            f.contains(wgpu::Features::SPIRV_SHADER_PASSTHROUGH),
            f.contains(wgpu::Features::SUBGROUP_BARRIER)
        );
    }
}
