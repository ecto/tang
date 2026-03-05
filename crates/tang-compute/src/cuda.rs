//! CUDA compute backend via cudarc.

use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use cudarc::driver::{CudaDevice as CudaDeviceInner, CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig};
use cudarc::nvrtc;

use crate::device::{ComputeBuffer, ComputeDevice};
use crate::kernels::{attention_cuda, backward_cuda, matmul_cuda, reduce_cuda};
use tang_expr::codegen::Dialect;
use tang_expr::node::ExprId;
use tang_expr::trace;

// ---- bf16 CPU conversion helpers ----

fn f32_to_bf16(val: f32) -> u16 {
    let bits = val.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7FFF + lsb);
    (rounded >> 16) as u16
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

// ---- Buffer storage ----

enum CudaStorage {
    F32(CudaSlice<f32>),
    Bf16(CudaSlice<u16>),
}

/// CUDA buffer wrapping either a CudaSlice<f32> or CudaSlice<u16> (bf16).
pub struct CudaBuffer {
    storage: CudaStorage,
    len: usize,
}

impl ComputeBuffer for CudaBuffer {
    fn len(&self) -> usize {
        self.len
    }

    fn to_vec(&self) -> Vec<f32> {
        match &self.storage {
            CudaStorage::F32(s) => self.device().dtoh_sync_copy(s).unwrap(),
            CudaStorage::Bf16(s) => {
                let u16_data = self.device().dtoh_sync_copy(s).unwrap();
                u16_data.iter().map(|&b| bf16_to_f32(b)).collect()
            }
        }
    }
}

impl CudaBuffer {
    fn device(&self) -> &Arc<CudaDeviceInner> {
        match &self.storage {
            CudaStorage::F32(s) => s.device(),
            CudaStorage::Bf16(s) => s.device(),
        }
    }

    /// Whether this buffer stores bf16 data.
    pub fn is_bf16(&self) -> bool {
        matches!(&self.storage, CudaStorage::Bf16(_))
    }

    /// Get the underlying f32 slice. Panics if bf16.
    fn f32_data(&self) -> &CudaSlice<f32> {
        match &self.storage {
            CudaStorage::F32(s) => s,
            CudaStorage::Bf16(_) => panic!("expected f32 buffer, got bf16"),
        }
    }

    /// Get the underlying u16 (bf16) slice. Panics if f32.
    fn bf16_data(&self) -> &CudaSlice<u16> {
        match &self.storage {
            CudaStorage::Bf16(s) => s,
            CudaStorage::F32(_) => panic!("expected bf16 buffer, got f32"),
        }
    }
}

/// CUDA compute device.
pub struct CudaComputeDevice {
    device: Arc<CudaDeviceInner>,
    module_cache: RefCell<HashMap<u64, String>>, // hash → module name
    mixed_precision: bool,
}

impl CudaComputeDevice {
    /// Create a new CUDA device (ordinal 0), f32 precision.
    pub fn new() -> Result<Self, cudarc::driver::DriverError> {
        let device = CudaDeviceInner::new(0)?;
        Ok(CudaComputeDevice {
            device,
            module_cache: RefCell::new(HashMap::new()),
            mixed_precision: false,
        })
    }

    /// Create a new CUDA device with bf16 mixed precision.
    /// Weights and activations stored in bf16, compute in f32 internally.
    pub fn new_mixed_precision() -> Result<Self, cudarc::driver::DriverError> {
        let device = CudaDeviceInner::new(0)?;
        Ok(CudaComputeDevice {
            device,
            module_cache: RefCell::new(HashMap::new()),
            mixed_precision: true,
        })
    }

    /// Compile and load a CUDA kernel, returning the module name.
    fn get_module(&self, source: &str, fn_names: &[&str]) -> String {
        let mut hasher = DefaultHasher::new();
        source.hash(&mut hasher);
        let hash = hasher.finish();

        if let Some(name) = self.module_cache.borrow().get(&hash) {
            return name.clone();
        }

        let module_name = format!("m{hash:x}");
        let ptx = nvrtc::compile_ptx(source).expect("Failed to compile CUDA kernel");

        self.device
            .load_ptx(ptx, &module_name, fn_names)
            .expect("Failed to load PTX");

        self.module_cache
            .borrow_mut()
            .insert(hash, module_name.clone());
        module_name
    }

    /// Allocate a zero-initialized f32 buffer regardless of mixed_precision mode.
    /// Used for gradient accumulation buffers that need f32 precision (atomicAdd targets).
    fn alloc_f32(&self, len: usize) -> CudaBuffer {
        let slice = self.device.alloc_zeros::<f32>(len).unwrap();
        CudaBuffer { storage: CudaStorage::F32(slice), len }
    }

    /// Upload data as f32 regardless of mixed_precision mode.
    /// Used for elementwise kernels and other f32-only operations.
    fn upload_f32(&self, data: &[f32]) -> CudaBuffer {
        let slice = self.device.htod_sync_copy(data).unwrap();
        CudaBuffer { storage: CudaStorage::F32(slice), len: data.len() }
    }
}

impl ComputeDevice for CudaComputeDevice {
    type Buffer = CudaBuffer;

    fn dialect(&self) -> Dialect {
        Dialect::Cuda
    }

    fn upload(&self, data: &[f32]) -> CudaBuffer {
        if self.mixed_precision {
            let bf16_data: Vec<u16> = data.iter().map(|&v| f32_to_bf16(v)).collect();
            let slice = self.device.htod_sync_copy(&bf16_data).unwrap();
            CudaBuffer { storage: CudaStorage::Bf16(slice), len: data.len() }
        } else {
            let slice = self.device.htod_sync_copy(data).unwrap();
            CudaBuffer { storage: CudaStorage::F32(slice), len: data.len() }
        }
    }

    fn upload_u32(&self, data: &[u32]) -> CudaBuffer {
        // Always f32 — these are integer IDs stored as f32 bit patterns, not learnable weights
        let f32_data: Vec<f32> = data.iter().map(|&x| f32::from_bits(x)).collect();
        self.upload_f32(&f32_data)
    }

    fn alloc(&self, len: usize) -> CudaBuffer {
        if self.mixed_precision {
            let slice = self.device.alloc_zeros::<u16>(len).unwrap();
            CudaBuffer { storage: CudaStorage::Bf16(slice), len }
        } else {
            let slice = self.device.alloc_zeros::<f32>(len).unwrap();
            CudaBuffer { storage: CudaStorage::F32(slice), len }
        }
    }

    fn download(&self, buf: &CudaBuffer) -> Vec<f32> {
        buf.to_vec()
    }

    fn elementwise(
        &self,
        inputs: &[&CudaBuffer],
        numel: usize,
        f: &dyn Fn(&[ExprId]) -> ExprId,
    ) -> CudaBuffer {
        let n_inputs = inputs.len();

        let (graph, output) = trace(|| {
            let vars: Vec<ExprId> = (0..n_inputs as u16).map(ExprId::var).collect();
            f(&vars)
        });
        let kernel = graph.to_kernel(&[output], n_inputs, Dialect::Cuda);
        let module = self.get_module(&kernel.source, &[kernel.entry_point]);

        // Interleave inputs — download auto-converts bf16→f32
        let mut interleaved = vec![0.0f32; numel * n_inputs];
        for i in 0..numel {
            for (j, inp) in inputs.iter().enumerate() {
                let inp_data = self.download(inp);
                interleaved[i * n_inputs + j] = inp_data[i];
            }
        }

        // Always f32 for tang-expr generated kernels
        let input_buf = self.upload_f32(&interleaved);
        let output_buf = self.alloc_f32(numel);
        let count = numel as u32;

        let func = self
            .device
            .get_func(&module, kernel.entry_point)
            .unwrap();

        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (((numel as u32) + 255) / 256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(cfg, (input_buf.f32_data(), output_buf.f32_data(), count))
                .unwrap();
        }

        // Convert output to bf16 if mixed precision
        if self.mixed_precision {
            let f32_data = self.download(&output_buf);
            self.upload(&f32_data)
        } else {
            output_buf
        }
    }

    fn matmul(&self, a: &CudaBuffer, b: &CudaBuffer, m: usize, k: usize, n: usize) -> CudaBuffer {
        if a.is_bf16() {
            let module = self.get_module(matmul_cuda::MATMUL_BF16_CUDA, &["matmul_bf16"]);
            let func = self.device.get_func(&module, "matmul_bf16").unwrap();
            let output = self.alloc(m * n);

            let cfg = LaunchConfig {
                block_dim: (16, 16, 1),
                grid_dim: (((n as u32) + 15) / 16, ((m as u32) + 15) / 16, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                func.launch(
                    cfg,
                    (a.bf16_data(), b.bf16_data(), output.bf16_data(), m as u32, k as u32, n as u32),
                )
                .unwrap();
            }

            output
        } else {
            let module = self.get_module(matmul_cuda::MATMUL_CUDA, &["matmul"]);
            let func = self.device.get_func(&module, "matmul").unwrap();
            let output = self.alloc(m * n);

            let cfg = LaunchConfig {
                block_dim: (16, 16, 1),
                grid_dim: (((n as u32) + 15) / 16, ((m as u32) + 15) / 16, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                func.launch(
                    cfg,
                    (a.f32_data(), b.f32_data(), output.f32_data(), m as u32, k as u32, n as u32),
                )
                .unwrap();
            }

            output
        }
    }

    fn softmax(&self, data: &CudaBuffer, n_rows: usize, row_len: usize) -> CudaBuffer {
        let tg_size = std::cmp::min(row_len, 256).next_power_of_two();
        let cfg = LaunchConfig {
            block_dim: (tg_size as u32, 1, 1),
            grid_dim: (n_rows as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        if data.is_bf16() {
            let module = self.get_module(reduce_cuda::SOFTMAX_BF16_CUDA, &["softmax_bf16"]);
            let func = self.device.get_func(&module, "softmax_bf16").unwrap();
            let output = self.alloc(data.len);

            unsafe {
                func.launch(
                    cfg,
                    (data.bf16_data(), output.bf16_data(), n_rows as u32, row_len as u32),
                )
                .unwrap();
            }

            output
        } else {
            let module = self.get_module(reduce_cuda::SOFTMAX_CUDA, &["softmax"]);
            let func = self.device.get_func(&module, "softmax").unwrap();
            let output = self.alloc(data.len);

            unsafe {
                func.launch(
                    cfg,
                    (data.f32_data(), output.f32_data(), n_rows as u32, row_len as u32),
                )
                .unwrap();
            }

            output
        }
    }

    fn rms_norm(
        &self,
        data: &CudaBuffer,
        weight: &CudaBuffer,
        n_groups: usize,
        dim: usize,
        eps: f32,
    ) -> CudaBuffer {
        let tg_size = std::cmp::min(dim, 256).next_power_of_two();
        let cfg = LaunchConfig {
            block_dim: (tg_size as u32, 1, 1),
            grid_dim: (n_groups as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        if data.is_bf16() {
            let module = self.get_module(reduce_cuda::RMS_NORM_BF16_CUDA, &["rms_norm_bf16"]);
            let func = self.device.get_func(&module, "rms_norm_bf16").unwrap();
            let output = self.alloc(data.len);

            unsafe {
                func.launch(
                    cfg,
                    (data.bf16_data(), weight.bf16_data(), output.bf16_data(), n_groups as u32, dim as u32, eps),
                )
                .unwrap();
            }

            output
        } else {
            let module = self.get_module(reduce_cuda::RMS_NORM_CUDA, &["rms_norm"]);
            let func = self.device.get_func(&module, "rms_norm").unwrap();
            let output = self.alloc(data.len);

            unsafe {
                func.launch(
                    cfg,
                    (data.f32_data(), weight.f32_data(), output.f32_data(), n_groups as u32, dim as u32, eps),
                )
                .unwrap();
            }

            output
        }
    }

    fn embedding(
        &self,
        weight: &CudaBuffer,
        ids: &CudaBuffer,
        seq_len: usize,
        dim: usize,
    ) -> CudaBuffer {
        // CPU fallback — download auto-converts bf16→f32, upload auto-converts f32→bf16
        let w = self.download(weight);
        let id_data = self.download(ids);
        let mut out = vec![0.0f32; seq_len * dim];
        for i in 0..seq_len {
            let id = id_data[i].to_bits() as usize;
            let src = id * dim;
            let dst = i * dim;
            out[dst..dst + dim].copy_from_slice(&w[src..src + dim]);
        }
        self.upload(&out)
    }

    fn reduce_sum(&self, data: &CudaBuffer, shape: &[usize], axis: usize) -> CudaBuffer {
        // CPU fallback — download auto-converts bf16→f32, upload auto-converts f32→bf16
        let cpu_data = self.download(data);
        let cpu = crate::CpuDevice::new();
        let buf = cpu.upload(&cpu_data);
        let result = cpu.reduce_sum(&buf, shape, axis);
        self.upload(&cpu.download(&result))
    }

    fn causal_attention(
        &self,
        q: &CudaBuffer,
        k: &CudaBuffer,
        v: &CudaBuffer,
        seq_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> CudaBuffer {
        let total_dim = n_heads * head_dim;
        let tg_size = std::cmp::min(seq_len, 256).next_power_of_two();
        let cfg = LaunchConfig {
            block_dim: (tg_size as u32, 1, 1),
            grid_dim: (seq_len as u32, n_heads as u32, 1),
            shared_mem_bytes: 0,
        };

        if q.is_bf16() {
            let module = self.get_module(
                attention_cuda::CAUSAL_ATTENTION_BF16_CUDA,
                &["causal_attention_bf16"],
            );
            let func = self.device.get_func(&module, "causal_attention_bf16").unwrap();
            let output = self.alloc(seq_len * total_dim);

            unsafe {
                func.launch(
                    cfg,
                    (
                        q.bf16_data(), k.bf16_data(), v.bf16_data(), output.bf16_data(),
                        seq_len as u32, n_heads as u32, n_kv_heads as u32, head_dim as u32,
                    ),
                )
                .unwrap();
            }

            output
        } else {
            let module = self.get_module(
                attention_cuda::CAUSAL_ATTENTION_CUDA,
                &["causal_attention"],
            );
            let func = self.device.get_func(&module, "causal_attention").unwrap();
            let output = self.alloc(seq_len * total_dim);

            unsafe {
                func.launch(
                    cfg,
                    (
                        q.f32_data(), k.f32_data(), v.f32_data(), output.f32_data(),
                        seq_len as u32, n_heads as u32, n_kv_heads as u32, head_dim as u32,
                    ),
                )
                .unwrap();
            }

            output
        }
    }

    fn kv_attention(
        &self,
        q: &CudaBuffer,
        k_cache: &CudaBuffer,
        v_cache: &CudaBuffer,
        cache_start: usize,
        q_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> CudaBuffer {
        let total_dim = n_heads * head_dim;

        if q_len == 1 {
            // Single-query decode
            let total_len = cache_start + 1;
            let tg_size = std::cmp::min(total_len, 256).next_power_of_two();
            let cfg = LaunchConfig {
                block_dim: (tg_size as u32, 1, 1),
                grid_dim: (n_heads as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            if q.is_bf16() {
                let module = self.get_module(
                    attention_cuda::KV_ATTENTION_BF16_CUDA,
                    &["kv_attention_bf16"],
                );
                let func = self.device.get_func(&module, "kv_attention_bf16").unwrap();
                let output = self.alloc(total_dim);

                unsafe {
                    func.launch(
                        cfg,
                        (
                            q.bf16_data(), k_cache.bf16_data(), v_cache.bf16_data(), output.bf16_data(),
                            total_len as u32, n_heads as u32, n_kv_heads as u32, head_dim as u32,
                        ),
                    )
                    .unwrap();
                }

                output
            } else {
                let module = self.get_module(
                    attention_cuda::KV_ATTENTION_CUDA,
                    &["kv_attention"],
                );
                let func = self.device.get_func(&module, "kv_attention").unwrap();
                let output = self.alloc(total_dim);

                unsafe {
                    func.launch(
                        cfg,
                        (
                            q.f32_data(), k_cache.f32_data(), v_cache.f32_data(), output.f32_data(),
                            total_len as u32, n_heads as u32, n_kv_heads as u32, head_dim as u32,
                        ),
                    )
                    .unwrap();
                }

                output
            }
        } else {
            // Batched prefill
            let max_attend = cache_start + q_len;
            let tg_size = std::cmp::min(max_attend, 256).next_power_of_two();
            let cfg = LaunchConfig {
                block_dim: (tg_size as u32, 1, 1),
                grid_dim: (q_len as u32, n_heads as u32, 1),
                shared_mem_bytes: 0,
            };

            if q.is_bf16() {
                let module = self.get_module(
                    attention_cuda::KV_ATTENTION_PREFILL_BF16_CUDA,
                    &["kv_attention_prefill_bf16"],
                );
                let func = self.device.get_func(&module, "kv_attention_prefill_bf16").unwrap();
                let output = self.alloc(q_len * total_dim);

                unsafe {
                    func.launch(
                        cfg,
                        (
                            q.bf16_data(), k_cache.bf16_data(), v_cache.bf16_data(), output.bf16_data(),
                            cache_start as u32, q_len as u32, n_heads as u32, n_kv_heads as u32, head_dim as u32,
                        ),
                    )
                    .unwrap();
                }

                output
            } else {
                let module = self.get_module(
                    attention_cuda::KV_ATTENTION_PREFILL_CUDA,
                    &["kv_attention_prefill"],
                );
                let func = self.device.get_func(&module, "kv_attention_prefill").unwrap();
                let output = self.alloc(q_len * total_dim);

                unsafe {
                    func.launch(
                        cfg,
                        (
                            q.f32_data(), k_cache.f32_data(), v_cache.f32_data(), output.f32_data(),
                            cache_start as u32, q_len as u32, n_heads as u32, n_kv_heads as u32, head_dim as u32,
                        ),
                    )
                    .unwrap();
                }

                output
            }
        }
    }

    fn transpose_2d(&self, buf: &CudaBuffer, rows: usize, cols: usize) -> CudaBuffer {
        let cfg = LaunchConfig {
            block_dim: (16, 16, 1),
            grid_dim: (((cols as u32) + 15) / 16, ((rows as u32) + 15) / 16, 1),
            shared_mem_bytes: 0,
        };

        if buf.is_bf16() {
            let module = self.get_module(backward_cuda::TRANSPOSE_2D_BF16_CUDA, &["transpose_2d_bf16"]);
            let func = self.device.get_func(&module, "transpose_2d_bf16").unwrap();
            let output = self.alloc(rows * cols);

            unsafe {
                func.launch(cfg, (buf.bf16_data(), output.bf16_data(), rows as u32, cols as u32))
                    .unwrap();
            }

            output
        } else {
            let module = self.get_module(backward_cuda::TRANSPOSE_2D_CUDA, &["transpose_2d"]);
            let func = self.device.get_func(&module, "transpose_2d").unwrap();
            let output = self.alloc(rows * cols);

            unsafe {
                func.launch(cfg, (buf.f32_data(), output.f32_data(), rows as u32, cols as u32))
                    .unwrap();
            }

            output
        }
    }

    fn softmax_backward(
        &self,
        softmax_out: &CudaBuffer,
        grad_output: &CudaBuffer,
        n_rows: usize,
        row_len: usize,
    ) -> CudaBuffer {
        let tg_size = std::cmp::min(row_len, 256).next_power_of_two();
        let cfg = LaunchConfig {
            block_dim: (tg_size as u32, 1, 1),
            grid_dim: (n_rows as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        if softmax_out.is_bf16() {
            let module = self.get_module(backward_cuda::SOFTMAX_BACKWARD_BF16_CUDA, &["softmax_backward_bf16"]);
            let func = self.device.get_func(&module, "softmax_backward_bf16").unwrap();
            let output = self.alloc(n_rows * row_len);

            unsafe {
                func.launch(
                    cfg,
                    (softmax_out.bf16_data(), grad_output.bf16_data(), output.bf16_data(), n_rows as u32, row_len as u32),
                )
                .unwrap();
            }

            output
        } else {
            let module = self.get_module(backward_cuda::SOFTMAX_BACKWARD_CUDA, &["softmax_backward"]);
            let func = self.device.get_func(&module, "softmax_backward").unwrap();
            let output = self.alloc(n_rows * row_len);

            unsafe {
                func.launch(
                    cfg,
                    (softmax_out.f32_data(), grad_output.f32_data(), output.f32_data(), n_rows as u32, row_len as u32),
                )
                .unwrap();
            }

            output
        }
    }

    fn rms_norm_backward(
        &self,
        input: &CudaBuffer,
        weight: &CudaBuffer,
        grad_output: &CudaBuffer,
        n_groups: usize,
        dim: usize,
        eps: f32,
    ) -> (CudaBuffer, CudaBuffer) {
        let tg_size = std::cmp::min(dim, 256).next_power_of_two();
        let cfg = LaunchConfig {
            block_dim: (tg_size as u32, 1, 1),
            grid_dim: (n_groups as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        if input.is_bf16() {
            let module = self.get_module(backward_cuda::RMS_NORM_BACKWARD_BF16_CUDA, &["rms_norm_backward_bf16"]);
            let func = self.device.get_func(&module, "rms_norm_backward_bf16").unwrap();
            let grad_input = self.alloc(n_groups * dim); // bf16
            let grad_weight = self.alloc_f32(dim); // f32 for atomic accumulation

            unsafe {
                func.launch(
                    cfg,
                    (
                        input.bf16_data(), weight.bf16_data(), grad_output.bf16_data(),
                        grad_input.bf16_data(), grad_weight.f32_data(),
                        n_groups as u32, dim as u32, eps,
                    ),
                )
                .unwrap();
            }

            (grad_input, grad_weight)
        } else {
            let module = self.get_module(backward_cuda::RMS_NORM_BACKWARD_CUDA, &["rms_norm_backward"]);
            let func = self.device.get_func(&module, "rms_norm_backward").unwrap();
            let grad_input = self.alloc(n_groups * dim);
            let grad_weight = self.alloc_f32(dim);

            unsafe {
                func.launch(
                    cfg,
                    (
                        input.f32_data(), weight.f32_data(), grad_output.f32_data(),
                        grad_input.f32_data(), grad_weight.f32_data(),
                        n_groups as u32, dim as u32, eps,
                    ),
                )
                .unwrap();
            }

            (grad_input, grad_weight)
        }
    }

    fn embedding_backward(
        &self,
        grad_output: &CudaBuffer,
        ids: &CudaBuffer,
        vocab_size: usize,
        seq_len: usize,
        dim: usize,
    ) -> CudaBuffer {
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (((seq_len as u32) + 255) / 256, 1, 1),
            shared_mem_bytes: 0,
        };

        // grad_weight is always f32 (atomicAdd accumulator)
        let grad_weight = self.alloc_f32(vocab_size * dim);

        if grad_output.is_bf16() {
            let module = self.get_module(backward_cuda::EMBEDDING_BACKWARD_BF16_CUDA, &["embedding_backward_bf16"]);
            let func = self.device.get_func(&module, "embedding_backward_bf16").unwrap();

            unsafe {
                func.launch(
                    cfg,
                    (
                        grad_output.bf16_data(), ids.f32_data(), grad_weight.f32_data(),
                        vocab_size as u32, seq_len as u32, dim as u32,
                    ),
                )
                .unwrap();
            }
        } else {
            let module = self.get_module(backward_cuda::EMBEDDING_BACKWARD_CUDA, &["embedding_backward"]);
            let func = self.device.get_func(&module, "embedding_backward").unwrap();

            unsafe {
                func.launch(
                    cfg,
                    (
                        grad_output.f32_data(), ids.f32_data(), grad_weight.f32_data(),
                        vocab_size as u32, seq_len as u32, dim as u32,
                    ),
                )
                .unwrap();
            }
        }

        grad_weight
    }

    fn causal_attention_backward(
        &self,
        grad_output: &CudaBuffer,
        q: &CudaBuffer,
        k: &CudaBuffer,
        v: &CudaBuffer,
        seq_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> (CudaBuffer, CudaBuffer, CudaBuffer) {
        let total_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        let tg_size = std::cmp::min(head_dim, 256).next_power_of_two();
        let cfg = LaunchConfig {
            block_dim: (tg_size as u32, 1, 1),
            grid_dim: (seq_len as u32, n_heads as u32, 1),
            shared_mem_bytes: 0,
        };

        if q.is_bf16() {
            let module = self.get_module(
                backward_cuda::CAUSAL_ATTENTION_BACKWARD_BF16_CUDA,
                &["causal_attention_backward_bf16"],
            );
            let func = self.device.get_func(&module, "causal_attention_backward_bf16").unwrap();

            let grad_q = self.alloc(seq_len * total_dim); // bf16
            // grad_K/V are f32 for atomic accumulation precision
            let grad_k = self.alloc_f32(seq_len * kv_dim);
            let grad_v = self.alloc_f32(seq_len * kv_dim);

            unsafe {
                func.launch(
                    cfg,
                    (
                        grad_output.bf16_data(), q.bf16_data(), k.bf16_data(), v.bf16_data(),
                        grad_q.bf16_data(), grad_k.f32_data(), grad_v.f32_data(),
                        seq_len as u32, n_heads as u32, n_kv_heads as u32, head_dim as u32,
                    ),
                )
                .unwrap();
            }

            (grad_q, grad_k, grad_v)
        } else {
            let module = self.get_module(
                backward_cuda::CAUSAL_ATTENTION_BACKWARD_CUDA,
                &["causal_attention_backward"],
            );
            let func = self.device.get_func(&module, "causal_attention_backward").unwrap();

            let grad_q = self.alloc(seq_len * total_dim);
            let grad_k = self.alloc_f32(seq_len * kv_dim);
            let grad_v = self.alloc_f32(seq_len * kv_dim);

            unsafe {
                func.launch(
                    cfg,
                    (
                        grad_output.f32_data(), q.f32_data(), k.f32_data(), v.f32_data(),
                        grad_q.f32_data(), grad_k.f32_data(), grad_v.f32_data(),
                        seq_len as u32, n_heads as u32, n_kv_heads as u32, head_dim as u32,
                    ),
                )
                .unwrap();
            }

            (grad_q, grad_k, grad_v)
        }
    }

    fn cross_entropy_forward_backward(
        &self,
        logits: &CudaBuffer,
        targets: &CudaBuffer,
        n_positions: usize,
        vocab_size: usize,
        pad_id: u32,
    ) -> (f32, CudaBuffer) {
        // Pre-count non-padded positions on CPU (tiny download, same as Metal path)
        let target_data = self.download(targets);
        let count = target_data
            .iter()
            .take(n_positions)
            .filter(|t| t.to_bits() != pad_id)
            .count() as u32;

        let tg_size = std::cmp::min(vocab_size, 256).next_power_of_two();
        let cfg = LaunchConfig {
            block_dim: (tg_size as u32, 1, 1),
            grid_dim: (n_positions as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // loss_out is always f32
        let loss_buf = self.alloc_f32(1);

        if logits.is_bf16() {
            let module = self.get_module(backward_cuda::CROSS_ENTROPY_BF16_CUDA, &["cross_entropy_fwd_bwd_bf16"]);
            let func = self.device.get_func(&module, "cross_entropy_fwd_bwd_bf16").unwrap();
            let grad = self.alloc(n_positions * vocab_size); // bf16

            unsafe {
                func.launch(
                    cfg,
                    (
                        logits.bf16_data(), targets.f32_data(), grad.bf16_data(), loss_buf.f32_data(),
                        n_positions as u32, vocab_size as u32, pad_id, count,
                    ),
                )
                .unwrap();
            }

            let loss_vec = self.download(&loss_buf);
            (loss_vec[0], grad)
        } else {
            let module = self.get_module(backward_cuda::CROSS_ENTROPY_CUDA, &["cross_entropy_fwd_bwd"]);
            let func = self.device.get_func(&module, "cross_entropy_fwd_bwd").unwrap();
            let grad = self.alloc(n_positions * vocab_size);

            unsafe {
                func.launch(
                    cfg,
                    (
                        logits.f32_data(), targets.f32_data(), grad.f32_data(), loss_buf.f32_data(),
                        n_positions as u32, vocab_size as u32, pad_id, count,
                    ),
                )
                .unwrap();
            }

            let loss_vec = self.download(&loss_buf);
            (loss_vec[0], grad)
        }
    }

    fn sync(&self) {
        self.device.synchronize().unwrap();
    }
}
