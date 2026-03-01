//! CUDA compute backend via cudarc.

use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use cudarc::driver::{CudaDevice as CudaDeviceInner, CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig};
use cudarc::nvrtc;

use crate::device::{ComputeBuffer, ComputeDevice};
use crate::kernels::{attention_cuda, matmul_cuda, reduce_cuda};
use tang_expr::codegen::Dialect;
use tang_expr::node::ExprId;
use tang_expr::trace;

/// CUDA buffer wrapping a CudaSlice<f32>.
pub struct CudaBuffer {
    data: CudaSlice<f32>,
    len: usize,
}

impl ComputeBuffer for CudaBuffer {
    fn len(&self) -> usize {
        self.len
    }

    fn to_vec(&self) -> Vec<f32> {
        self.device().dtoh_sync_copy(&self.data).unwrap()
    }
}

impl CudaBuffer {
    fn device(&self) -> &Arc<CudaDeviceInner> {
        self.data.device()
    }
}

/// CUDA compute device.
pub struct CudaComputeDevice {
    device: Arc<CudaDeviceInner>,
    module_cache: RefCell<HashMap<u64, String>>, // hash → module name
}

impl CudaComputeDevice {
    /// Create a new CUDA device (ordinal 0).
    pub fn new() -> Result<Self, cudarc::driver::DriverError> {
        let device = CudaDeviceInner::new(0)?;
        Ok(CudaComputeDevice {
            device,
            module_cache: RefCell::new(HashMap::new()),
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
}

impl ComputeDevice for CudaComputeDevice {
    type Buffer = CudaBuffer;

    fn dialect(&self) -> Dialect {
        Dialect::Cuda
    }

    fn upload(&self, data: &[f32]) -> CudaBuffer {
        let slice = self.device.htod_sync_copy(data).unwrap();
        CudaBuffer {
            len: data.len(),
            data: slice,
        }
    }

    fn upload_u32(&self, data: &[u32]) -> CudaBuffer {
        // Reinterpret u32 as f32 bits
        let f32_data: Vec<f32> = data.iter().map(|&x| f32::from_bits(x)).collect();
        self.upload(&f32_data)
    }

    fn alloc(&self, len: usize) -> CudaBuffer {
        let slice = self.device.alloc_zeros::<f32>(len).unwrap();
        CudaBuffer { data: slice, len }
    }

    fn download(&self, buf: &CudaBuffer) -> Vec<f32> {
        self.device.dtoh_sync_copy(&buf.data).unwrap()
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

        // Interleave inputs
        let mut interleaved = vec![0.0f32; numel * n_inputs];
        for i in 0..numel {
            for (j, inp) in inputs.iter().enumerate() {
                let inp_data = self.download(inp);
                interleaved[i * n_inputs + j] = inp_data[i];
            }
        }

        let input_buf = self.upload(&interleaved);
        let output_buf = self.alloc(numel);
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
            func.launch(cfg, (&input_buf.data, &output_buf.data, count))
                .unwrap();
        }

        output_buf
    }

    fn matmul(&self, a: &CudaBuffer, b: &CudaBuffer, m: usize, k: usize, n: usize) -> CudaBuffer {
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
                (&a.data, &b.data, &output.data, m as u32, k as u32, n as u32),
            )
            .unwrap();
        }

        output
    }

    fn softmax(&self, data: &CudaBuffer, n_rows: usize, row_len: usize) -> CudaBuffer {
        let module = self.get_module(reduce_cuda::SOFTMAX_CUDA, &["softmax"]);
        let func = self.device.get_func(&module, "softmax").unwrap();
        let output = self.alloc(data.len);

        let tg_size = std::cmp::min(row_len, 256).next_power_of_two();
        let cfg = LaunchConfig {
            block_dim: (tg_size as u32, 1, 1),
            grid_dim: (n_rows as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(
                cfg,
                (&data.data, &output.data, n_rows as u32, row_len as u32),
            )
            .unwrap();
        }

        output
    }

    fn rms_norm(
        &self,
        data: &CudaBuffer,
        weight: &CudaBuffer,
        n_groups: usize,
        dim: usize,
        eps: f32,
    ) -> CudaBuffer {
        let module = self.get_module(reduce_cuda::RMS_NORM_CUDA, &["rms_norm"]);
        let func = self.device.get_func(&module, "rms_norm").unwrap();
        let output = self.alloc(data.len);

        let tg_size = std::cmp::min(dim, 256).next_power_of_two();
        let cfg = LaunchConfig {
            block_dim: (tg_size as u32, 1, 1),
            grid_dim: (n_groups as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(
                cfg,
                (
                    &data.data,
                    &weight.data,
                    &output.data,
                    n_groups as u32,
                    dim as u32,
                    eps,
                ),
            )
            .unwrap();
        }

        output
    }

    fn embedding(
        &self,
        weight: &CudaBuffer,
        ids: &CudaBuffer,
        seq_len: usize,
        dim: usize,
    ) -> CudaBuffer {
        // CPU fallback for embedding — simple operation
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
        // CPU fallback for reduce_sum
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
        head_dim: usize,
    ) -> CudaBuffer {
        let module = self.get_module(
            attention_cuda::CAUSAL_ATTENTION_CUDA,
            &["causal_attention"],
        );
        let func = self.device.get_func(&module, "causal_attention").unwrap();
        let total_dim = n_heads * head_dim;
        let output = self.alloc(seq_len * total_dim);

        let tg_size = std::cmp::min(seq_len, 256).next_power_of_two();
        let cfg = LaunchConfig {
            block_dim: (tg_size as u32, 1, 1),
            grid_dim: (seq_len as u32, n_heads as u32, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            func.launch(
                cfg,
                (
                    &q.data,
                    &k.data,
                    &v.data,
                    &output.data,
                    seq_len as u32,
                    n_heads as u32,
                    head_dim as u32,
                ),
            )
            .unwrap();
        }

        output
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
            let module = self.get_module(
                attention_cuda::KV_ATTENTION_CUDA,
                &["kv_attention"],
            );
            let func = self.device.get_func(&module, "kv_attention").unwrap();
            let total_len = cache_start + 1;
            let output = self.alloc(total_dim);

            let tg_size = std::cmp::min(total_len, 256).next_power_of_two();
            let cfg = LaunchConfig {
                block_dim: (tg_size as u32, 1, 1),
                grid_dim: (n_heads as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                func.launch(
                    cfg,
                    (
                        &q.data,
                        &k_cache.data,
                        &v_cache.data,
                        &output.data,
                        total_len as u32,
                        n_heads as u32,
                        n_kv_heads as u32,
                        head_dim as u32,
                    ),
                )
                .unwrap();
            }

            output
        } else {
            // Batched prefill
            let module = self.get_module(
                attention_cuda::KV_ATTENTION_PREFILL_CUDA,
                &["kv_attention_prefill"],
            );
            let func = self.device.get_func(&module, "kv_attention_prefill").unwrap();
            let output = self.alloc(q_len * total_dim);

            let max_attend = cache_start + q_len;
            let tg_size = std::cmp::min(max_attend, 256).next_power_of_two();
            let cfg = LaunchConfig {
                block_dim: (tg_size as u32, 1, 1),
                grid_dim: (q_len as u32, n_heads as u32, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                func.launch(
                    cfg,
                    (
                        &q.data,
                        &k_cache.data,
                        &v_cache.data,
                        &output.data,
                        cache_start as u32,
                        q_len as u32,
                        n_heads as u32,
                        n_kv_heads as u32,
                        head_dim as u32,
                    ),
                )
                .unwrap();
            }

            output
        }
    }

    fn sync(&self) {
        self.device.synchronize().unwrap();
    }
}
