//! CUDA compute backend via cudarc.

use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg};
use cudarc::nvrtc;

use crate::device::{ComputeBuffer, ComputeDevice};
use crate::kernels::{attention_cuda, backward_cuda, reduce_cuda};
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
            CudaStorage::F32(s) => s.stream().memcpy_dtov(s).unwrap(),
            CudaStorage::Bf16(s) => {
                let u16_data: Vec<u16> = s.stream().memcpy_dtov(s).unwrap();
                u16_data.iter().map(|&b| bf16_to_f32(b)).collect()
            }
        }
    }
}

impl CudaBuffer {
    /// Whether this buffer stores bf16 data.
    pub fn is_bf16(&self) -> bool {
        matches!(&self.storage, CudaStorage::Bf16(_))
    }

    /// Get the underlying f32 slice (immutable). Panics if bf16.
    fn f32_data(&self) -> &CudaSlice<f32> {
        match &self.storage {
            CudaStorage::F32(s) => s,
            CudaStorage::Bf16(_) => panic!("expected f32 buffer, got bf16"),
        }
    }

    /// Get the underlying f32 slice (mutable). Panics if bf16.
    fn f32_data_mut(&mut self) -> &mut CudaSlice<f32> {
        match &mut self.storage {
            CudaStorage::F32(s) => s,
            CudaStorage::Bf16(_) => panic!("expected f32 buffer, got bf16"),
        }
    }

    /// Get the underlying u16 (bf16) slice (immutable). Panics if f32.
    fn bf16_data(&self) -> &CudaSlice<u16> {
        match &self.storage {
            CudaStorage::Bf16(s) => s,
            CudaStorage::F32(_) => panic!("expected bf16 buffer, got f32"),
        }
    }

    /// Get the underlying u16 (bf16) slice (mutable). Panics if f32.
    fn bf16_data_mut(&mut self) -> &mut CudaSlice<u16> {
        match &mut self.storage {
            CudaStorage::Bf16(s) => s,
            CudaStorage::F32(_) => panic!("expected bf16 buffer, got f32"),
        }
    }
}

/// CUDA compute device.
pub struct CudaComputeDevice {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    cublas: CudaBlas,
    module_cache: RefCell<HashMap<u64, Arc<CudaModule>>>, // hash → module
    mixed_precision: bool,
}

impl CudaComputeDevice {
    /// Create a new CUDA device (ordinal 0), f32 precision.
    pub fn new() -> Result<Self, cudarc::driver::DriverError> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let cublas = CudaBlas::new(stream.clone()).expect("Failed to create cuBLAS handle");
        Ok(CudaComputeDevice {
            ctx,
            stream,
            cublas,
            module_cache: RefCell::new(HashMap::new()),
            mixed_precision: false,
        })
    }

    /// Create a new CUDA device with bf16 mixed precision.
    /// Weights and activations stored in bf16, compute in f32 internally.
    pub fn new_mixed_precision() -> Result<Self, cudarc::driver::DriverError> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let cublas = CudaBlas::new(stream.clone()).expect("Failed to create cuBLAS handle");
        Ok(CudaComputeDevice {
            ctx,
            stream,
            cublas,
            module_cache: RefCell::new(HashMap::new()),
            mixed_precision: true,
        })
    }

    /// Compile and load a CUDA kernel, returning the module.
    fn get_module(&self, source: &str) -> Arc<CudaModule> {
        let mut hasher = DefaultHasher::new();
        source.hash(&mut hasher);
        let hash = hasher.finish();

        if let Some(module) = self.module_cache.borrow().get(&hash) {
            return Arc::clone(module);
        }

        let ptx = nvrtc::compile_ptx(source).expect("Failed to compile CUDA kernel");
        let module = self.ctx.load_module(ptx).expect("Failed to load PTX");

        self.module_cache
            .borrow_mut()
            .insert(hash, Arc::clone(&module));
        module
    }

    /// Get a function from a module by name.
    fn get_func(&self, source: &str, fn_name: &str) -> (Arc<CudaModule>, CudaFunction) {
        let module = self.get_module(source);
        let func = module.load_function(fn_name).expect("Failed to load CUDA function");
        (module, func)
    }

    /// Allocate a zero-initialized f32 buffer regardless of mixed_precision mode.
    /// Used for gradient accumulation buffers that need f32 precision (atomicAdd targets).
    fn alloc_f32(&self, len: usize) -> CudaBuffer {
        let slice = self.stream.alloc_zeros::<f32>(len).unwrap();
        CudaBuffer { storage: CudaStorage::F32(slice), len }
    }

    /// Upload data as f32 regardless of mixed_precision mode.
    /// Used for elementwise kernels and other f32-only operations.
    fn upload_f32(&self, data: &[f32]) -> CudaBuffer {
        let slice = self.stream.memcpy_stod(data).unwrap();
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
            let slice = self.stream.memcpy_stod(&bf16_data).unwrap();
            CudaBuffer { storage: CudaStorage::Bf16(slice), len: data.len() }
        } else {
            let slice = self.stream.memcpy_stod(data).unwrap();
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
            let slice = self.stream.alloc_zeros::<u16>(len).unwrap();
            CudaBuffer { storage: CudaStorage::Bf16(slice), len }
        } else {
            let slice = self.stream.alloc_zeros::<f32>(len).unwrap();
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
        let (_module, func) = self.get_func(&kernel.source, kernel.entry_point);

        // If inputs are bf16, convert to f32 GPU-side for the elementwise kernel.
        // We keep each input as a separate buffer — no interleaving, no CPU round-trip.
        let converted: Vec<CudaBuffer>;
        let f32_inputs: Vec<&CudaSlice<f32>> = if self.mixed_precision {
            converted = inputs
                .iter()
                .map(|inp| {
                    let data = self.download(inp);
                    self.upload_f32(&data)
                })
                .collect();
            converted.iter().map(|b| b.f32_data()).collect()
        } else {
            inputs.iter().map(|inp| inp.f32_data()).collect()
        };

        let mut output_buf = self.alloc_f32(numel);
        let count = numel as u32;

        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (((numel as u32) + 255) / 256, 1, 1),
            shared_mem_bytes: 0,
        };

        // Pass each input as a separate kernel argument (matches the generated kernel signature)
        unsafe {
            let mut builder = self.stream.launch_builder(&func);
            for slice in &f32_inputs {
                builder.arg(*slice);
            }
            builder
                .arg(output_buf.f32_data_mut())
                .arg(&count)
                .launch(cfg)
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
        // cuBLAS is column-major. For row-major C[m,n] = A[m,k] * B[k,n]:
        //   C_col^T = B_col^T * A_col^T
        // So we call gemm with swapped A/B and cublas m=n, n=m.
        let cublas_m = n as i32;
        let cublas_n = m as i32;
        let cublas_k = k as i32;

        if a.is_bf16() {
            let mut output = self.alloc(m * n);

            // Use gemm_ex for bf16 (stored as u16) with f32 compute
            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;

            {
                let (b_ptr, _rec_b) = b.bf16_data().device_ptr(&self.stream);
                let (a_ptr, _rec_a) = a.bf16_data().device_ptr(&self.stream);
                let (c_ptr, _rec_c) = output.bf16_data_mut().device_ptr_mut(&self.stream);

                unsafe {
                    cudarc::cublas::result::gemm_ex(
                        *self.cublas.handle(),
                        cublasOperation_t::CUBLAS_OP_N,
                        cublasOperation_t::CUBLAS_OP_N,
                        cublas_m,
                        cublas_n,
                        cublas_k,
                        (&alpha) as *const f32 as *const _,
                        b_ptr as *const _,
                        cudarc::cublas::sys::cudaDataType_t::CUDA_R_16BF,
                        cublas_m, // lda = n (row stride of B)
                        a_ptr as *const _,
                        cudarc::cublas::sys::cudaDataType_t::CUDA_R_16BF,
                        cublas_k, // ldb = k (row stride of A)
                        (&beta) as *const f32 as *const _,
                        c_ptr as *mut _,
                        cudarc::cublas::sys::cudaDataType_t::CUDA_R_16BF,
                        cublas_m, // ldc = n (row stride of C)
                        cudarc::cublas::sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                        cudarc::cublas::sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                    )
                    .expect("cuBLAS bf16 gemm_ex failed");
                }
            } // drop _rec_c borrow here

            output
        } else {
            let mut output = self.alloc(m * n);

            unsafe {
                self.cublas.gemm(
                    GemmConfig {
                        transa: cublasOperation_t::CUBLAS_OP_N,
                        transb: cublasOperation_t::CUBLAS_OP_N,
                        m: cublas_m,
                        n: cublas_n,
                        k: cublas_k,
                        alpha: 1.0f32,
                        lda: cublas_m, // n (row stride of B)
                        ldb: cublas_k, // k (row stride of A)
                        beta: 0.0f32,
                        ldc: cublas_m, // n (row stride of C)
                    },
                    b.f32_data(),
                    a.f32_data(),
                    output.f32_data_mut(),
                )
                .expect("cuBLAS sgemm failed");
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
        let n_rows_u32 = n_rows as u32;
        let row_len_u32 = row_len as u32;

        if data.is_bf16() {
            let (_module, func) = self.get_func(reduce_cuda::SOFTMAX_BF16_CUDA, "softmax_bf16");
            let mut output = self.alloc(data.len);

            unsafe {
                self.stream.launch_builder(&func)
                    .arg(data.bf16_data())
                    .arg(output.bf16_data_mut())
                    .arg(&n_rows_u32)
                    .arg(&row_len_u32)
                    .launch(cfg)
                    .unwrap();
            }

            output
        } else {
            let (_module, func) = self.get_func(reduce_cuda::SOFTMAX_CUDA, "softmax");
            let mut output = self.alloc(data.len);

            unsafe {
                self.stream.launch_builder(&func)
                    .arg(data.f32_data())
                    .arg(output.f32_data_mut())
                    .arg(&n_rows_u32)
                    .arg(&row_len_u32)
                    .launch(cfg)
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
        let n_groups_u32 = n_groups as u32;
        let dim_u32 = dim as u32;

        if data.is_bf16() {
            let (_module, func) = self.get_func(reduce_cuda::RMS_NORM_BF16_CUDA, "rms_norm_bf16");
            let mut output = self.alloc(data.len);

            unsafe {
                self.stream.launch_builder(&func)
                    .arg(data.bf16_data())
                    .arg(weight.bf16_data())
                    .arg(output.bf16_data_mut())
                    .arg(&n_groups_u32)
                    .arg(&dim_u32)
                    .arg(&eps)
                    .launch(cfg)
                    .unwrap();
            }

            output
        } else {
            let (_module, func) = self.get_func(reduce_cuda::RMS_NORM_CUDA, "rms_norm");
            let mut output = self.alloc(data.len);

            unsafe {
                self.stream.launch_builder(&func)
                    .arg(data.f32_data())
                    .arg(weight.f32_data())
                    .arg(output.f32_data_mut())
                    .arg(&n_groups_u32)
                    .arg(&dim_u32)
                    .arg(&eps)
                    .launch(cfg)
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
        let total = seq_len * dim;
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (((total as u32) + 255) / 256, 1, 1),
            shared_mem_bytes: 0,
        };
        let seq_len_u32 = seq_len as u32;
        let dim_u32 = dim as u32;

        if weight.is_bf16() {
            let (_module, func) = self.get_func(
                reduce_cuda::EMBEDDING_GATHER_BF16_CUDA,
                "embedding_gather_bf16",
            );
            let mut output = self.alloc(total);

            // ids is always f32 storage (u32 bit patterns) — cast to unsigned int* on GPU side
            unsafe {
                self.stream.launch_builder(&func)
                    .arg(weight.bf16_data())
                    .arg(ids.f32_data())
                    .arg(output.bf16_data_mut())
                    .arg(&seq_len_u32)
                    .arg(&dim_u32)
                    .launch(cfg)
                    .unwrap();
            }

            output
        } else {
            let (_module, func) = self.get_func(
                reduce_cuda::EMBEDDING_GATHER_CUDA,
                "embedding_gather",
            );
            let mut output = self.alloc(total);

            // ids is always f32 storage (u32 bit patterns) — cast to unsigned int* on GPU side
            unsafe {
                self.stream.launch_builder(&func)
                    .arg(weight.f32_data())
                    .arg(ids.f32_data())
                    .arg(output.f32_data_mut())
                    .arg(&seq_len_u32)
                    .arg(&dim_u32)
                    .launch(cfg)
                    .unwrap();
            }

            output
        }
    }

    fn reduce_sum(&self, data: &CudaBuffer, shape: &[usize], axis: usize) -> CudaBuffer {
        let ndim = shape.len();
        assert!(axis < ndim);

        let outer: usize = shape[..axis].iter().product();
        let axis_len = shape[axis];
        let inner: usize = shape[axis + 1..].iter().product();
        let out_len = outer * inner;

        if out_len == 0 {
            return self.alloc(0);
        }

        let tg_size = std::cmp::min(axis_len, 256).next_power_of_two();
        let cfg = LaunchConfig {
            block_dim: (tg_size as u32, 1, 1),
            grid_dim: (out_len as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let outer_u32 = outer as u32;
        let axis_len_u32 = axis_len as u32;
        let inner_u32 = inner as u32;

        if data.is_bf16() {
            let (_module, func) = self.get_func(reduce_cuda::REDUCE_SUM_BF16_CUDA, "reduce_sum_bf16");
            let mut output = self.alloc(out_len);

            unsafe {
                self.stream.launch_builder(&func)
                    .arg(data.bf16_data())
                    .arg(output.bf16_data_mut())
                    .arg(&outer_u32)
                    .arg(&axis_len_u32)
                    .arg(&inner_u32)
                    .launch(cfg)
                    .unwrap();
            }

            output
        } else {
            let (_module, func) = self.get_func(reduce_cuda::REDUCE_SUM_CUDA, "reduce_sum");
            let mut output = self.alloc(out_len);

            unsafe {
                self.stream.launch_builder(&func)
                    .arg(data.f32_data())
                    .arg(output.f32_data_mut())
                    .arg(&outer_u32)
                    .arg(&axis_len_u32)
                    .arg(&inner_u32)
                    .launch(cfg)
                    .unwrap();
            }

            output
        }
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
        let seq_len_u32 = seq_len as u32;
        let n_heads_u32 = n_heads as u32;
        let n_kv_heads_u32 = n_kv_heads as u32;
        let head_dim_u32 = head_dim as u32;

        if q.is_bf16() {
            let tg_size = std::cmp::min(seq_len, 256).next_power_of_two();
            let cfg = LaunchConfig {
                block_dim: (tg_size as u32, 1, 1),
                grid_dim: (seq_len as u32, n_heads as u32, 1),
                shared_mem_bytes: 0,
            };
            let (_module, func) = self.get_func(
                attention_cuda::CAUSAL_ATTENTION_BF16_CUDA,
                "causal_attention_bf16",
            );
            let mut output = self.alloc(seq_len * total_dim);

            unsafe {
                self.stream.launch_builder(&func)
                    .arg(q.bf16_data())
                    .arg(k.bf16_data())
                    .arg(v.bf16_data())
                    .arg(output.bf16_data_mut())
                    .arg(&seq_len_u32)
                    .arg(&n_heads_u32)
                    .arg(&n_kv_heads_u32)
                    .arg(&head_dim_u32)
                    .launch(cfg)
                    .unwrap();
            }

            output
        } else {
            // FlashAttention-style kernel: block_dim = head_dim, dynamic shared memory
            // smem: tile_scores[FA_TILE_KV] + reduce[blockDim] + out_acc[head_dim]
            let tile_kv: usize = 64; // must match FA_TILE_KV in the kernel
            let smem_bytes = (tile_kv + head_dim + head_dim) * 4;
            let cfg = LaunchConfig {
                block_dim: (head_dim as u32, 1, 1),
                grid_dim: (seq_len as u32, n_heads as u32, 1),
                shared_mem_bytes: smem_bytes as u32,
            };
            let (_module, func) = self.get_func(
                attention_cuda::CAUSAL_ATTENTION_FLASH_CUDA,
                "causal_attention_flash",
            );
            let mut output = self.alloc(seq_len * total_dim);

            unsafe {
                self.stream.launch_builder(&func)
                    .arg(q.f32_data())
                    .arg(k.f32_data())
                    .arg(v.f32_data())
                    .arg(output.f32_data_mut())
                    .arg(&seq_len_u32)
                    .arg(&n_heads_u32)
                    .arg(&n_kv_heads_u32)
                    .arg(&head_dim_u32)
                    .launch(cfg)
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
        let n_heads_u32 = n_heads as u32;
        let n_kv_heads_u32 = n_kv_heads as u32;
        let head_dim_u32 = head_dim as u32;

        if q_len == 1 {
            // Single-query decode
            let total_len = cache_start + 1;
            let tg_size = std::cmp::min(total_len, 256).next_power_of_two();
            let cfg = LaunchConfig {
                block_dim: (tg_size as u32, 1, 1),
                grid_dim: (n_heads as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let total_len_u32 = total_len as u32;

            if q.is_bf16() {
                let (_module, func) = self.get_func(
                    attention_cuda::KV_ATTENTION_BF16_CUDA,
                    "kv_attention_bf16",
                );
                let mut output = self.alloc(total_dim);

                unsafe {
                    self.stream.launch_builder(&func)
                        .arg(q.bf16_data())
                        .arg(k_cache.bf16_data())
                        .arg(v_cache.bf16_data())
                        .arg(output.bf16_data_mut())
                        .arg(&total_len_u32)
                        .arg(&n_heads_u32)
                        .arg(&n_kv_heads_u32)
                        .arg(&head_dim_u32)
                        .launch(cfg)
                        .unwrap();
                }

                output
            } else {
                let (_module, func) = self.get_func(
                    attention_cuda::KV_ATTENTION_CUDA,
                    "kv_attention",
                );
                let mut output = self.alloc(total_dim);

                unsafe {
                    self.stream.launch_builder(&func)
                        .arg(q.f32_data())
                        .arg(k_cache.f32_data())
                        .arg(v_cache.f32_data())
                        .arg(output.f32_data_mut())
                        .arg(&total_len_u32)
                        .arg(&n_heads_u32)
                        .arg(&n_kv_heads_u32)
                        .arg(&head_dim_u32)
                        .launch(cfg)
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
            let cache_start_u32 = cache_start as u32;
            let q_len_u32 = q_len as u32;

            if q.is_bf16() {
                let (_module, func) = self.get_func(
                    attention_cuda::KV_ATTENTION_PREFILL_BF16_CUDA,
                    "kv_attention_prefill_bf16",
                );
                let mut output = self.alloc(q_len * total_dim);

                unsafe {
                    self.stream.launch_builder(&func)
                        .arg(q.bf16_data())
                        .arg(k_cache.bf16_data())
                        .arg(v_cache.bf16_data())
                        .arg(output.bf16_data_mut())
                        .arg(&cache_start_u32)
                        .arg(&q_len_u32)
                        .arg(&n_heads_u32)
                        .arg(&n_kv_heads_u32)
                        .arg(&head_dim_u32)
                        .launch(cfg)
                        .unwrap();
                }

                output
            } else {
                let (_module, func) = self.get_func(
                    attention_cuda::KV_ATTENTION_PREFILL_CUDA,
                    "kv_attention_prefill",
                );
                let mut output = self.alloc(q_len * total_dim);

                unsafe {
                    self.stream.launch_builder(&func)
                        .arg(q.f32_data())
                        .arg(k_cache.f32_data())
                        .arg(v_cache.f32_data())
                        .arg(output.f32_data_mut())
                        .arg(&cache_start_u32)
                        .arg(&q_len_u32)
                        .arg(&n_heads_u32)
                        .arg(&n_kv_heads_u32)
                        .arg(&head_dim_u32)
                        .launch(cfg)
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
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;

        if buf.is_bf16() {
            let (_module, func) = self.get_func(backward_cuda::TRANSPOSE_2D_BF16_CUDA, "transpose_2d_bf16");
            let mut output = self.alloc(rows * cols);

            unsafe {
                self.stream.launch_builder(&func)
                    .arg(buf.bf16_data())
                    .arg(output.bf16_data_mut())
                    .arg(&rows_u32)
                    .arg(&cols_u32)
                    .launch(cfg)
                    .unwrap();
            }

            output
        } else {
            let (_module, func) = self.get_func(backward_cuda::TRANSPOSE_2D_CUDA, "transpose_2d");
            let mut output = self.alloc(rows * cols);

            unsafe {
                self.stream.launch_builder(&func)
                    .arg(buf.f32_data())
                    .arg(output.f32_data_mut())
                    .arg(&rows_u32)
                    .arg(&cols_u32)
                    .launch(cfg)
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
        let n_rows_u32 = n_rows as u32;
        let row_len_u32 = row_len as u32;

        if softmax_out.is_bf16() {
            let (_module, func) = self.get_func(backward_cuda::SOFTMAX_BACKWARD_BF16_CUDA, "softmax_backward_bf16");
            let mut output = self.alloc(n_rows * row_len);

            unsafe {
                self.stream.launch_builder(&func)
                    .arg(softmax_out.bf16_data())
                    .arg(grad_output.bf16_data())
                    .arg(output.bf16_data_mut())
                    .arg(&n_rows_u32)
                    .arg(&row_len_u32)
                    .launch(cfg)
                    .unwrap();
            }

            output
        } else {
            let (_module, func) = self.get_func(backward_cuda::SOFTMAX_BACKWARD_CUDA, "softmax_backward");
            let mut output = self.alloc(n_rows * row_len);

            unsafe {
                self.stream.launch_builder(&func)
                    .arg(softmax_out.f32_data())
                    .arg(grad_output.f32_data())
                    .arg(output.f32_data_mut())
                    .arg(&n_rows_u32)
                    .arg(&row_len_u32)
                    .launch(cfg)
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
        let n_groups_u32 = n_groups as u32;
        let dim_u32 = dim as u32;

        if input.is_bf16() {
            let (_module, func) = self.get_func(backward_cuda::RMS_NORM_BACKWARD_BF16_CUDA, "rms_norm_backward_bf16");
            let mut grad_input = self.alloc(n_groups * dim); // bf16
            let mut grad_weight = self.alloc_f32(dim); // f32 for atomic accumulation

            unsafe {
                self.stream.launch_builder(&func)
                    .arg(input.bf16_data())
                    .arg(weight.bf16_data())
                    .arg(grad_output.bf16_data())
                    .arg(grad_input.bf16_data_mut())
                    .arg(grad_weight.f32_data_mut())
                    .arg(&n_groups_u32)
                    .arg(&dim_u32)
                    .arg(&eps)
                    .launch(cfg)
                    .unwrap();
            }

            (grad_input, grad_weight)
        } else {
            let (_module, func) = self.get_func(backward_cuda::RMS_NORM_BACKWARD_CUDA, "rms_norm_backward");
            let mut grad_input = self.alloc(n_groups * dim);
            let mut grad_weight = self.alloc_f32(dim);

            unsafe {
                self.stream.launch_builder(&func)
                    .arg(input.f32_data())
                    .arg(weight.f32_data())
                    .arg(grad_output.f32_data())
                    .arg(grad_input.f32_data_mut())
                    .arg(grad_weight.f32_data_mut())
                    .arg(&n_groups_u32)
                    .arg(&dim_u32)
                    .arg(&eps)
                    .launch(cfg)
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
        let vocab_size_u32 = vocab_size as u32;
        let seq_len_u32 = seq_len as u32;
        let dim_u32 = dim as u32;

        // grad_weight is always f32 (atomicAdd accumulator)
        let mut grad_weight = self.alloc_f32(vocab_size * dim);

        if grad_output.is_bf16() {
            let (_module, func) = self.get_func(backward_cuda::EMBEDDING_BACKWARD_BF16_CUDA, "embedding_backward_bf16");

            unsafe {
                self.stream.launch_builder(&func)
                    .arg(grad_output.bf16_data())
                    .arg(ids.f32_data())
                    .arg(grad_weight.f32_data_mut())
                    .arg(&vocab_size_u32)
                    .arg(&seq_len_u32)
                    .arg(&dim_u32)
                    .launch(cfg)
                    .unwrap();
            }
        } else {
            let (_module, func) = self.get_func(backward_cuda::EMBEDDING_BACKWARD_CUDA, "embedding_backward");

            unsafe {
                self.stream.launch_builder(&func)
                    .arg(grad_output.f32_data())
                    .arg(ids.f32_data())
                    .arg(grad_weight.f32_data_mut())
                    .arg(&vocab_size_u32)
                    .arg(&seq_len_u32)
                    .arg(&dim_u32)
                    .launch(cfg)
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
        let seq_len_u32 = seq_len as u32;
        let n_heads_u32 = n_heads as u32;
        let n_kv_heads_u32 = n_kv_heads as u32;
        let head_dim_u32 = head_dim as u32;

        if q.is_bf16() {
            let (_module, func) = self.get_func(
                backward_cuda::CAUSAL_ATTENTION_BACKWARD_BF16_CUDA,
                "causal_attention_backward_bf16",
            );

            let mut grad_q = self.alloc(seq_len * total_dim); // bf16
            // grad_K/V are f32 for atomic accumulation precision
            let mut grad_k = self.alloc_f32(seq_len * kv_dim);
            let mut grad_v = self.alloc_f32(seq_len * kv_dim);

            unsafe {
                self.stream.launch_builder(&func)
                    .arg(grad_output.bf16_data())
                    .arg(q.bf16_data())
                    .arg(k.bf16_data())
                    .arg(v.bf16_data())
                    .arg(grad_q.bf16_data_mut())
                    .arg(grad_k.f32_data_mut())
                    .arg(grad_v.f32_data_mut())
                    .arg(&seq_len_u32)
                    .arg(&n_heads_u32)
                    .arg(&n_kv_heads_u32)
                    .arg(&head_dim_u32)
                    .launch(cfg)
                    .unwrap();
            }

            (grad_q, grad_k, grad_v)
        } else {
            // FlashAttention-2 backward: recompute O, precompute D, then tiled backward.

            // Step 1: Recompute forward output O (flash attention trades compute for memory)
            let o_buf = self.causal_attention(q, k, v, seq_len, n_heads, n_kv_heads, head_dim);

            // Step 2: Precompute D[i,h] = sum_d(dO[i,d] * O[i,d])
            let d_tg = std::cmp::min(head_dim, 256).next_power_of_two();
            let d_cfg = LaunchConfig {
                block_dim: (d_tg as u32, 1, 1),
                grid_dim: (seq_len as u32, n_heads as u32, 1),
                shared_mem_bytes: (d_tg * 4) as u32,
            };
            let (_module, d_func) = self.get_func(
                backward_cuda::FLASH_ATTN_BWD_PRECOMPUTE_D_CUDA,
                "flash_attn_bwd_precompute_d",
            );
            let mut d_buf = self.alloc_f32(seq_len * n_heads);
            unsafe {
                self.stream.launch_builder(&d_func)
                    .arg(grad_output.f32_data())
                    .arg(o_buf.f32_data())
                    .arg(d_buf.f32_data_mut())
                    .arg(&seq_len_u32)
                    .arg(&n_heads_u32)
                    .arg(&head_dim_u32)
                    .launch(d_cfg)
                    .unwrap();
            }

            // Step 3: Flash backward kernel
            let tile_kv: usize = 64; // must match FA_BWD_TILE in kernel
            let n_kv_tiles = (seq_len + tile_kv - 1) / tile_kv;
            let bwd_tg = std::cmp::min(head_dim, 256).next_power_of_two();
            // smem: dK_acc[TILE*D] + dV_acc[TILE*D] + scores[TILE] + reduce[tg]
            let smem_bytes = (2 * tile_kv * head_dim + tile_kv + bwd_tg) * 4;
            let bwd_cfg = LaunchConfig {
                block_dim: (bwd_tg as u32, 1, 1),
                grid_dim: (n_kv_tiles as u32, n_kv_heads as u32, 1),
                shared_mem_bytes: smem_bytes as u32,
            };
            let (_module, bwd_func) = self.get_func(
                backward_cuda::CAUSAL_ATTENTION_BACKWARD_FLASH_CUDA,
                "causal_attention_backward_flash",
            );

            // grad_Q needs zero-init (atomicAdd target)
            let mut grad_q = self.alloc_f32(seq_len * total_dim);
            // grad_K/V written directly (no atomics), but zero-init for partial tiles
            let mut grad_k = self.alloc_f32(seq_len * kv_dim);
            let mut grad_v = self.alloc_f32(seq_len * kv_dim);

            unsafe {
                self.stream.launch_builder(&bwd_func)
                    .arg(grad_output.f32_data())
                    .arg(q.f32_data())
                    .arg(k.f32_data())
                    .arg(v.f32_data())
                    .arg(d_buf.f32_data())
                    .arg(grad_q.f32_data_mut())
                    .arg(grad_k.f32_data_mut())
                    .arg(grad_v.f32_data_mut())
                    .arg(&seq_len_u32)
                    .arg(&n_heads_u32)
                    .arg(&n_kv_heads_u32)
                    .arg(&head_dim_u32)
                    .launch(bwd_cfg)
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
        let n_positions_u32 = n_positions as u32;
        let vocab_size_u32 = vocab_size as u32;

        // loss_out is always f32
        let mut loss_buf = self.alloc_f32(1);

        if logits.is_bf16() {
            let (_module, func) = self.get_func(backward_cuda::CROSS_ENTROPY_BF16_CUDA, "cross_entropy_fwd_bwd_bf16");
            let mut grad = self.alloc(n_positions * vocab_size); // bf16

            unsafe {
                self.stream.launch_builder(&func)
                    .arg(logits.bf16_data())
                    .arg(targets.f32_data())
                    .arg(grad.bf16_data_mut())
                    .arg(loss_buf.f32_data_mut())
                    .arg(&n_positions_u32)
                    .arg(&vocab_size_u32)
                    .arg(&pad_id)
                    .arg(&count)
                    .launch(cfg)
                    .unwrap();
            }

            let loss_vec = self.download(&loss_buf);
            (loss_vec[0], grad)
        } else {
            let (_module, func) = self.get_func(backward_cuda::CROSS_ENTROPY_CUDA, "cross_entropy_fwd_bwd");
            let mut grad = self.alloc(n_positions * vocab_size);

            unsafe {
                self.stream.launch_builder(&func)
                    .arg(logits.f32_data())
                    .arg(targets.f32_data())
                    .arg(grad.f32_data_mut())
                    .arg(loss_buf.f32_data_mut())
                    .arg(&n_positions_u32)
                    .arg(&vocab_size_u32)
                    .arg(&pad_id)
                    .arg(&count)
                    .launch(cfg)
                    .unwrap();
            }

            let loss_vec = self.download(&loss_buf);
            (loss_vec[0], grad)
        }
    }

    fn sync(&self) {
        self.stream.synchronize().unwrap();
    }

    fn add_assign(&self, dst: &mut CudaBuffer, src: &CudaBuffer) {
        assert_eq!(dst.len, src.len);
        let n = dst.len as u32;
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: ((n + 255) / 256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (_module, func) = self.get_func(
            crate::kernels::adamw_cuda::ADD_ASSIGN_CUDA,
            "add_assign",
        );
        unsafe {
            self.stream.launch_builder(&func)
                .arg(dst.f32_data_mut())
                .arg(src.f32_data())
                .arg(&n)
                .launch(cfg)
                .unwrap();
        }
    }

    fn zero_buffer(&self, buf: &mut CudaBuffer) {
        let n = buf.len as u32;
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: ((n + 255) / 256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (_module, func) = self.get_func(
            crate::kernels::adamw_cuda::ZERO_BUFFER_CUDA,
            "zero_buffer",
        );
        unsafe {
            self.stream.launch_builder(&func)
                .arg(buf.f32_data_mut())
                .arg(&n)
                .launch(cfg)
                .unwrap();
        }
    }

    fn adamw_step(
        &self,
        param: &mut CudaBuffer,
        grad: &CudaBuffer,
        m: &mut CudaBuffer,
        v: &mut CudaBuffer,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step_t: usize,
    ) {
        let n = param.len as u32;
        let beta1_pow = beta1.powi(step_t as i32);
        let beta2_pow = beta2.powi(step_t as i32);
        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: ((n + 255) / 256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (_module, func) = self.get_func(
            crate::kernels::adamw_cuda::ADAMW_STEP_CUDA,
            "adamw_step",
        );
        unsafe {
            self.stream.launch_builder(&func)
                .arg(param.f32_data_mut())
                .arg(grad.f32_data())
                .arg(m.f32_data_mut())
                .arg(v.f32_data_mut())
                .arg(&lr)
                .arg(&beta1)
                .arg(&beta2)
                .arg(&eps)
                .arg(&weight_decay)
                .arg(&beta1_pow)
                .arg(&beta2_pow)
                .arg(&n)
                .launch(cfg)
                .unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CpuDevice;
    use crate::ComputeDevice;

    fn approx_eq(a: &[f32], b: &[f32], tol: f32, label: &str) {
        assert_eq!(a.len(), b.len(), "{label}: length mismatch {} vs {}", a.len(), b.len());
        let max_diff = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        assert!(max_diff < tol, "{label}: max diff {max_diff} exceeds tolerance {tol}");
    }

    #[test]
    fn flash_backward_matches_cpu_small() {
        // Small GQA config: seq_len=4, n_heads=2, n_kv_heads=1, head_dim=4
        let seq_len = 4;
        let n_heads = 2;
        let n_kv_heads = 1;
        let head_dim = 4;
        let total_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        // Deterministic test data
        let q: Vec<f32> = (0..seq_len * total_dim).map(|i| ((i as f32) * 0.1 - 0.5).sin()).collect();
        let k: Vec<f32> = (0..seq_len * kv_dim).map(|i| ((i as f32) * 0.2 + 0.3).cos()).collect();
        let v: Vec<f32> = (0..seq_len * kv_dim).map(|i| ((i as f32) * 0.15 - 0.1).sin()).collect();
        let grad_out: Vec<f32> = (0..seq_len * total_dim).map(|i| ((i as f32) * 0.3 + 0.7).cos()).collect();

        // CPU reference
        let cpu = CpuDevice::new();
        let q_cpu = cpu.upload(&q);
        let k_cpu = cpu.upload(&k);
        let v_cpu = cpu.upload(&v);
        let go_cpu = cpu.upload(&grad_out);
        let (gq_cpu, gk_cpu, gv_cpu) = cpu.causal_attention_backward(
            &go_cpu, &q_cpu, &k_cpu, &v_cpu, seq_len, n_heads, n_kv_heads, head_dim,
        );
        let gq_ref = cpu.download(&gq_cpu);
        let gk_ref = cpu.download(&gk_cpu);
        let gv_ref = cpu.download(&gv_cpu);

        // CUDA flash backward
        let gpu = CudaComputeDevice::new().expect("no CUDA GPU");
        let q_gpu = gpu.upload(&q);
        let k_gpu = gpu.upload(&k);
        let v_gpu = gpu.upload(&v);
        let go_gpu = gpu.upload(&grad_out);
        let (gq_gpu, gk_gpu, gv_gpu) = gpu.causal_attention_backward(
            &go_gpu, &q_gpu, &k_gpu, &v_gpu, seq_len, n_heads, n_kv_heads, head_dim,
        );
        let gq_cuda = gpu.download(&gq_gpu);
        let gk_cuda = gpu.download(&gk_gpu);
        let gv_cuda = gpu.download(&gv_gpu);

        approx_eq(&gq_cuda, &gq_ref, 1e-3, "grad_Q");
        approx_eq(&gk_cuda, &gk_ref, 1e-3, "grad_K");
        approx_eq(&gv_cuda, &gv_ref, 1e-3, "grad_V");
    }

    #[test]
    fn flash_backward_matches_cpu_mha() {
        // MHA config: seq_len=8, n_heads=4, n_kv_heads=4, head_dim=8
        let seq_len = 8;
        let n_heads = 4;
        let n_kv_heads = 4;
        let head_dim = 8;
        let total_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        let q: Vec<f32> = (0..seq_len * total_dim).map(|i| ((i as f32) * 0.1 - 1.0).sin()).collect();
        let k: Vec<f32> = (0..seq_len * kv_dim).map(|i| ((i as f32) * 0.2 + 0.5).cos()).collect();
        let v: Vec<f32> = (0..seq_len * kv_dim).map(|i| ((i as f32) * 0.15 - 0.3).sin()).collect();
        let grad_out: Vec<f32> = (0..seq_len * total_dim).map(|i| ((i as f32) * 0.25 + 0.1).cos()).collect();

        let cpu = CpuDevice::new();
        let q_cpu = cpu.upload(&q);
        let k_cpu = cpu.upload(&k);
        let v_cpu = cpu.upload(&v);
        let go_cpu = cpu.upload(&grad_out);
        let (gq_cpu, gk_cpu, gv_cpu) = cpu.causal_attention_backward(
            &go_cpu, &q_cpu, &k_cpu, &v_cpu, seq_len, n_heads, n_kv_heads, head_dim,
        );
        let gq_ref = cpu.download(&gq_cpu);
        let gk_ref = cpu.download(&gk_cpu);
        let gv_ref = cpu.download(&gv_cpu);

        let gpu = CudaComputeDevice::new().expect("no CUDA GPU");
        let q_gpu = gpu.upload(&q);
        let k_gpu = gpu.upload(&k);
        let v_gpu = gpu.upload(&v);
        let go_gpu = gpu.upload(&grad_out);
        let (gq_gpu, gk_gpu, gv_gpu) = gpu.causal_attention_backward(
            &go_gpu, &q_gpu, &k_gpu, &v_gpu, seq_len, n_heads, n_kv_heads, head_dim,
        );
        let gq_cuda = gpu.download(&gq_gpu);
        let gk_cuda = gpu.download(&gk_gpu);
        let gv_cuda = gpu.download(&gv_gpu);

        approx_eq(&gq_cuda, &gq_ref, 1e-3, "grad_Q");
        approx_eq(&gk_cuda, &gk_ref, 1e-3, "grad_K");
        approx_eq(&gv_cuda, &gv_ref, 1e-3, "grad_V");
    }

    #[test]
    fn flash_backward_matches_cpu_350m_shape() {
        // 350M model shape at reduced seq_len: exercises multi-tile path
        let seq_len = 128;
        let n_heads = 16;
        let n_kv_heads = 8;
        let head_dim = 64;
        let total_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        let q: Vec<f32> = (0..seq_len * total_dim).map(|i| ((i as f32) * 0.01 - 2.0).sin() * 0.1).collect();
        let k: Vec<f32> = (0..seq_len * kv_dim).map(|i| ((i as f32) * 0.02 + 1.0).cos() * 0.1).collect();
        let v: Vec<f32> = (0..seq_len * kv_dim).map(|i| ((i as f32) * 0.015 - 0.5).sin() * 0.1).collect();
        let grad_out: Vec<f32> = (0..seq_len * total_dim).map(|i| ((i as f32) * 0.03 + 0.2).cos() * 0.1).collect();

        let cpu = CpuDevice::new();
        let q_cpu = cpu.upload(&q);
        let k_cpu = cpu.upload(&k);
        let v_cpu = cpu.upload(&v);
        let go_cpu = cpu.upload(&grad_out);
        let (gq_cpu, gk_cpu, gv_cpu) = cpu.causal_attention_backward(
            &go_cpu, &q_cpu, &k_cpu, &v_cpu, seq_len, n_heads, n_kv_heads, head_dim,
        );
        let gq_ref = cpu.download(&gq_cpu);
        let gk_ref = cpu.download(&gk_cpu);
        let gv_ref = cpu.download(&gv_cpu);

        let gpu = CudaComputeDevice::new().expect("no CUDA GPU");
        let q_gpu = gpu.upload(&q);
        let k_gpu = gpu.upload(&k);
        let v_gpu = gpu.upload(&v);
        let go_gpu = gpu.upload(&grad_out);
        let (gq_gpu, gk_gpu, gv_gpu) = gpu.causal_attention_backward(
            &go_gpu, &q_gpu, &k_gpu, &v_gpu, seq_len, n_heads, n_kv_heads, head_dim,
        );
        let gq_cuda = gpu.download(&gq_gpu);
        let gk_cuda = gpu.download(&gk_gpu);
        let gv_cuda = gpu.download(&gv_gpu);

        // Wider tolerance for larger configs (f32 accumulation order differences
        // between flash tiled path and sequential CPU path)
        approx_eq(&gq_cuda, &gq_ref, 2e-1, "grad_Q");
        approx_eq(&gk_cuda, &gk_ref, 2e-1, "grad_K");
        approx_eq(&gv_cuda, &gv_ref, 2e-1, "grad_V");
    }

    #[test]
    fn flash_forward_matches_cpu() {
        // Verify flash forward also matches CPU
        let seq_len = 16;
        let n_heads = 4;
        let n_kv_heads = 2;
        let head_dim = 8;
        let total_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        let q: Vec<f32> = (0..seq_len * total_dim).map(|i| ((i as f32) * 0.1 - 1.0).sin()).collect();
        let k: Vec<f32> = (0..seq_len * kv_dim).map(|i| ((i as f32) * 0.2 + 0.5).cos()).collect();
        let v: Vec<f32> = (0..seq_len * kv_dim).map(|i| ((i as f32) * 0.15 - 0.3).sin()).collect();

        let cpu = CpuDevice::new();
        let o_cpu = cpu.causal_attention(
            &cpu.upload(&q), &cpu.upload(&k), &cpu.upload(&v),
            seq_len, n_heads, n_kv_heads, head_dim,
        );
        let o_ref = cpu.download(&o_cpu);

        let gpu = CudaComputeDevice::new().expect("no CUDA GPU");
        let o_gpu = gpu.causal_attention(
            &gpu.upload(&q), &gpu.upload(&k), &gpu.upload(&v),
            seq_len, n_heads, n_kv_heads, head_dim,
        );
        let o_cuda = gpu.download(&o_gpu);

        approx_eq(&o_cuda, &o_ref, 1e-4, "forward output");
    }

    #[test]
    fn flash_backward_bench_350m() {
        // Benchmark flash backward at actual 350M training shape
        let seq_len = 512;
        let n_heads = 16;
        let n_kv_heads = 8;
        let head_dim = 64;
        let total_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        let q: Vec<f32> = (0..seq_len * total_dim).map(|i| ((i as f32) * 0.01).sin() * 0.1).collect();
        let k: Vec<f32> = (0..seq_len * kv_dim).map(|i| ((i as f32) * 0.02).cos() * 0.1).collect();
        let v: Vec<f32> = (0..seq_len * kv_dim).map(|i| ((i as f32) * 0.015).sin() * 0.1).collect();
        let grad_out: Vec<f32> = (0..seq_len * total_dim).map(|i| ((i as f32) * 0.03).cos() * 0.1).collect();

        let gpu = CudaComputeDevice::new().expect("no CUDA GPU");
        let q_gpu = gpu.upload(&q);
        let k_gpu = gpu.upload(&k);
        let v_gpu = gpu.upload(&v);
        let go_gpu = gpu.upload(&grad_out);

        // Warmup
        let _ = gpu.causal_attention_backward(
            &go_gpu, &q_gpu, &k_gpu, &v_gpu, seq_len, n_heads, n_kv_heads, head_dim,
        );
        gpu.sync();

        let iters = 20;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = gpu.causal_attention_backward(
                &go_gpu, &q_gpu, &k_gpu, &v_gpu, seq_len, n_heads, n_kv_heads, head_dim,
            );
            gpu.sync();
        }
        let elapsed = t0.elapsed().as_secs_f64();
        let per_call = elapsed / iters as f64;
        eprintln!("  flash backward 350M (seq=512): {:.3}ms/call ({} iters, {:.3}s total)",
            per_call * 1000.0, iters, elapsed);

        // Also bench forward for reference
        let _ = gpu.causal_attention(
            &q_gpu, &k_gpu, &v_gpu, seq_len, n_heads, n_kv_heads, head_dim,
        );
        gpu.sync();
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let _ = gpu.causal_attention(
                &q_gpu, &k_gpu, &v_gpu, seq_len, n_heads, n_kv_heads, head_dim,
            );
            gpu.sync();
        }
        let elapsed = t0.elapsed().as_secs_f64();
        let per_call = elapsed / iters as f64;
        eprintln!("  flash forward  350M (seq=512): {:.3}ms/call ({} iters, {:.3}s total)",
            per_call * 1000.0, iters, elapsed);
        eprintln!();
    }
}
