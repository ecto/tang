//! Metal compute backend for Apple Silicon.

use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use metal::objc::rc::autoreleasepool;
use metal::*;

use crate::device::{ComputeBuffer, ComputeDevice};
use crate::kernels::{attention_msl, backward_msl, matmul_msl, reduce_msl};
use tang_expr::codegen::Dialect;
use tang_expr::node::ExprId;
use tang_expr::trace;

/// Metal buffer wrapping a `metal::Buffer`.
pub struct MetalBuffer {
    buffer: metal::Buffer,
    len: usize,
}

impl ComputeBuffer for MetalBuffer {
    fn len(&self) -> usize {
        self.len
    }

    fn to_vec(&self) -> Vec<f32> {
        let ptr = self.buffer.contents() as *const f32;
        let slice = unsafe { std::slice::from_raw_parts(ptr, self.len) };
        slice.to_vec()
    }
}

/// Metal compute device.
pub struct MetalDevice {
    device: metal::Device,
    queue: CommandQueue,
    pipeline_cache: RefCell<HashMap<u64, ComputePipelineState>>,
    /// Last committed command buffer — waited on in `sync()`.
    last_cb: RefCell<Option<CommandBuffer>>,
}

impl MetalDevice {
    /// Create a new Metal device using the system default GPU.
    pub fn new() -> Option<Self> {
        let device = metal::Device::system_default()?;
        let queue = device.new_command_queue();
        Some(MetalDevice {
            device,
            queue,
            pipeline_cache: RefCell::new(HashMap::new()),
            last_cb: RefCell::new(None),
        })
    }

    /// Get or compile a pipeline from MSL source.
    fn get_pipeline(&self, source: &str, fn_name: &str) -> ComputePipelineState {
        let mut hasher = DefaultHasher::new();
        source.hash(&mut hasher);
        fn_name.hash(&mut hasher);
        let hash = hasher.finish();

        if let Some(pipeline) = self.pipeline_cache.borrow().get(&hash) {
            return pipeline.clone();
        }

        let options = CompileOptions::new();
        let library = self
            .device
            .new_library_with_source(source, &options)
            .expect("Failed to compile MSL");
        let func = library
            .get_function(fn_name, None)
            .expect("Failed to get function");
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .expect("Failed to create pipeline");

        self.pipeline_cache.borrow_mut().insert(hash, pipeline.clone());
        pipeline
    }

    /// Create a Metal buffer from f32 data.
    fn make_buffer(&self, data: &[f32]) -> metal::Buffer {
        let len = data.len() * std::mem::size_of::<f32>();
        self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            len as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Create a Metal buffer from u32 data.
    fn make_buffer_u32(&self, data: &[u32]) -> metal::Buffer {
        let len = data.len() * std::mem::size_of::<u32>();
        self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            len as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Allocate an empty Metal buffer.
    fn make_buffer_empty(&self, byte_len: usize) -> metal::Buffer {
        self.device.new_buffer(
            byte_len as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Dispatch a compute pipeline with given buffers.
    fn dispatch(
        &self,
        pipeline: &ComputePipelineState,
        buffers: &[&metal::Buffer],
        threads: u64,
    ) {
        autoreleasepool(|| {
            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pipeline);

            for (i, buf) in buffers.iter().enumerate() {
                enc.set_buffer(i as u64, Some(buf), 0);
            }

            let tg_size = std::cmp::min(
                pipeline.max_total_threads_per_threadgroup(),
                256,
            );
            let tg_count = (threads + tg_size - 1) / tg_size;

            enc.dispatch_thread_groups(
                MTLSize::new(tg_count, 1, 1),
                MTLSize::new(tg_size, 1, 1),
            );

            enc.end_encoding();
            cmd.commit();
            *self.last_cb.borrow_mut() = Some(cmd.to_owned());
        });
    }

    /// Dispatch with 2D grid.
    fn dispatch_2d(
        &self,
        pipeline: &ComputePipelineState,
        buffers: &[&metal::Buffer],
        grid: (u64, u64),
        tg_size: (u64, u64),
    ) {
        autoreleasepool(|| {
            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pipeline);

            for (i, buf) in buffers.iter().enumerate() {
                enc.set_buffer(i as u64, Some(buf), 0);
            }

            let tg_count_x = (grid.0 + tg_size.0 - 1) / tg_size.0;
            let tg_count_y = (grid.1 + tg_size.1 - 1) / tg_size.1;

            enc.dispatch_thread_groups(
                MTLSize::new(tg_count_x, tg_count_y, 1),
                MTLSize::new(tg_size.0, tg_size.1, 1),
            );

            enc.end_encoding();
            cmd.commit();
            *self.last_cb.borrow_mut() = Some(cmd.to_owned());
        });
    }
}

impl ComputeDevice for MetalDevice {
    type Buffer = MetalBuffer;

    fn dialect(&self) -> Dialect {
        Dialect::Msl
    }

    fn upload(&self, data: &[f32]) -> MetalBuffer {
        MetalBuffer {
            buffer: self.make_buffer(data),
            len: data.len(),
        }
    }

    fn upload_u32(&self, data: &[u32]) -> MetalBuffer {
        MetalBuffer {
            buffer: self.make_buffer_u32(data),
            len: data.len(),
        }
    }

    fn alloc(&self, len: usize) -> MetalBuffer {
        MetalBuffer {
            buffer: self.make_buffer_empty(len * 4),
            len,
        }
    }

    fn download(&self, buf: &MetalBuffer) -> Vec<f32> {
        self.sync();
        buf.to_vec()
    }

    fn elementwise(
        &self,
        inputs: &[&MetalBuffer],
        numel: usize,
        f: &dyn Fn(&[ExprId]) -> ExprId,
    ) -> MetalBuffer {
        let n_inputs = inputs.len();

        // Trace closure → MSL kernel
        let (graph, output) = trace(|| {
            let vars: Vec<ExprId> = (0..n_inputs as u16).map(ExprId::var).collect();
            f(&vars)
        });
        let kernel = graph.to_kernel(&[output], n_inputs, Dialect::Msl);
        let pipeline = self.get_pipeline(&kernel.source, kernel.entry_point);

        // Sync before reading input buffers that may have pending GPU writes
        self.sync();

        // Pre-download all inputs once, then interleave
        let input_vecs: Vec<Vec<f32>> = inputs.iter().map(|inp| inp.to_vec()).collect();
        let mut interleaved = vec![0.0f32; numel * n_inputs];
        for i in 0..numel {
            for (j, vec) in input_vecs.iter().enumerate() {
                interleaved[i * n_inputs + j] = vec[i];
            }
        }
        let input_buf = self.make_buffer(&interleaved);
        let output_buf = self.make_buffer_empty(numel * 4);
        let count_buf = self.make_buffer_u32(&[numel as u32]);

        self.dispatch(
            &pipeline,
            &[&input_buf, &output_buf, &count_buf],
            numel as u64,
        );

        MetalBuffer {
            buffer: output_buf,
            len: numel,
        }
    }

    fn matmul(&self, a: &MetalBuffer, b: &MetalBuffer, m: usize, k: usize, n: usize) -> MetalBuffer {
        // Use naive kernel for now — simdgroup_matrix version requires
        // specific alignment and dispatch patterns
        let pipeline = self.get_pipeline(matmul_msl::MATMUL_NAIVE_MSL, "matmul_naive");
        let output_buf = self.make_buffer_empty(m * n * 4);
        let params = self.make_buffer_u32(&[m as u32, k as u32, n as u32]);

        self.dispatch_2d(
            &pipeline,
            &[&a.buffer, &b.buffer, &output_buf, &params],
            (n as u64, m as u64),
            (16, 16),
        );

        MetalBuffer {
            buffer: output_buf,
            len: m * n,
        }
    }

    fn softmax(&self, data: &MetalBuffer, n_rows: usize, row_len: usize) -> MetalBuffer {
        let pipeline = self.get_pipeline(reduce_msl::SOFTMAX_MSL, "softmax");
        let output_buf = self.make_buffer_empty(data.len * 4);
        let params = self.make_buffer_u32(&[n_rows as u32, row_len as u32]);

        autoreleasepool(|| {
            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&data.buffer), 0);
            enc.set_buffer(1, Some(&output_buf), 0);
            enc.set_buffer(2, Some(&params), 0);

            let tg_size = std::cmp::min(row_len as u64, 256).next_power_of_two();
            enc.dispatch_thread_groups(
                MTLSize::new(n_rows as u64, 1, 1),
                MTLSize::new(tg_size, 1, 1),
            );
            enc.end_encoding();
            cmd.commit();
            *self.last_cb.borrow_mut() = Some(cmd.to_owned());
        });

        MetalBuffer {
            buffer: output_buf,
            len: data.len,
        }
    }

    fn rms_norm(
        &self,
        data: &MetalBuffer,
        weight: &MetalBuffer,
        n_groups: usize,
        dim: usize,
        eps: f32,
    ) -> MetalBuffer {
        let pipeline = self.get_pipeline(reduce_msl::RMS_NORM_MSL, "rms_norm");
        let output_buf = self.make_buffer_empty(data.len * 4);
        let params = self.make_buffer_u32(&[n_groups as u32, dim as u32, eps.to_bits()]);

        autoreleasepool(|| {
            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&data.buffer), 0);
            enc.set_buffer(1, Some(&weight.buffer), 0);
            enc.set_buffer(2, Some(&output_buf), 0);
            enc.set_buffer(3, Some(&params), 0);

            let tg_size = std::cmp::min(dim as u64, 256).next_power_of_two();
            enc.dispatch_thread_groups(
                MTLSize::new(n_groups as u64, 1, 1),
                MTLSize::new(tg_size, 1, 1),
            );
            enc.end_encoding();
            cmd.commit();
            *self.last_cb.borrow_mut() = Some(cmd.to_owned());
        });

        MetalBuffer {
            buffer: output_buf,
            len: data.len,
        }
    }

    fn embedding(
        &self,
        weight: &MetalBuffer,
        ids: &MetalBuffer,
        seq_len: usize,
        dim: usize,
    ) -> MetalBuffer {
        // Simple MSL embedding kernel inline
        let src = r#"
#include <metal_stdlib>
using namespace metal;

kernel void embedding(
    device const float* weight [[buffer(0)]],
    device const uint* ids [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const uint* params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint seq_len = params[0];
    uint dim = params[1];
    if (gid >= seq_len) return;

    uint id = ids[gid];
    for (uint d = 0; d < dim; d++) {
        output[gid * dim + d] = weight[id * dim + d];
    }
}
"#;
        let pipeline = self.get_pipeline(src, "embedding");
        let output_buf = self.make_buffer_empty(seq_len * dim * 4);
        let params = self.make_buffer_u32(&[seq_len as u32, dim as u32]);

        self.dispatch(
            &pipeline,
            &[&weight.buffer, &ids.buffer, &output_buf, &params],
            seq_len as u64,
        );

        MetalBuffer {
            buffer: output_buf,
            len: seq_len * dim,
        }
    }

    fn reduce_sum(&self, data: &MetalBuffer, shape: &[usize], axis: usize) -> MetalBuffer {
        // CPU fallback for reduce_sum — complex axis handling
        self.sync();
        let cpu_data = data.to_vec();
        let cpu_dev = crate::CpuDevice::new();
        let cpu_buf = cpu_dev.upload(&cpu_data);
        let result = cpu_dev.reduce_sum(&cpu_buf, shape, axis);
        let out = cpu_dev.download(&result);
        self.upload(&out)
    }

    fn causal_attention(
        &self,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        seq_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> MetalBuffer {
        let pipeline = self.get_pipeline(
            attention_msl::CAUSAL_ATTENTION_MSL,
            "causal_attention",
        );
        let total_dim = n_heads * head_dim;
        let output_buf = self.make_buffer_empty(seq_len * total_dim * 4);
        let params = self.make_buffer_u32(&[seq_len as u32, n_heads as u32, n_kv_heads as u32, head_dim as u32]);

        // Flash attention: threads parallelize over head_dim
        let tg_size = std::cmp::min(head_dim as u64, 256).next_power_of_two();

        autoreleasepool(|| {
            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&q.buffer), 0);
            enc.set_buffer(1, Some(&k.buffer), 0);
            enc.set_buffer(2, Some(&v.buffer), 0);
            enc.set_buffer(3, Some(&output_buf), 0);
            enc.set_buffer(4, Some(&params), 0);

            enc.dispatch_thread_groups(
                MTLSize::new(seq_len as u64, n_heads as u64, 1),
                MTLSize::new(tg_size, 1, 1),
            );
            enc.end_encoding();
            cmd.commit();
            *self.last_cb.borrow_mut() = Some(cmd.to_owned());
        });

        MetalBuffer {
            buffer: output_buf,
            len: seq_len * total_dim,
        }
    }

    fn kv_attention(
        &self,
        q: &MetalBuffer,
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        cache_start: usize,
        q_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> MetalBuffer {
        let total_dim = n_heads * head_dim;
        // Flash attention: threads parallelize over head_dim
        let tg_size = std::cmp::min(head_dim as u64, 256).next_power_of_two();

        if q_len == 1 {
            // Single-query decode
            let pipeline = self.get_pipeline(
                attention_msl::KV_ATTENTION_MSL,
                "kv_attention",
            );
            let total_len = cache_start + 1;
            let output_buf = self.make_buffer_empty(total_dim * 4);
            let params = self.make_buffer_u32(&[
                total_len as u32,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
            ]);

            autoreleasepool(|| {
                let cmd = self.queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&pipeline);
                enc.set_buffer(0, Some(&q.buffer), 0);
                enc.set_buffer(1, Some(&k_cache.buffer), 0);
                enc.set_buffer(2, Some(&v_cache.buffer), 0);
                enc.set_buffer(3, Some(&output_buf), 0);
                enc.set_buffer(4, Some(&params), 0);

                enc.dispatch_thread_groups(
                    MTLSize::new(n_heads as u64, 1, 1),
                    MTLSize::new(tg_size, 1, 1),
                );
                enc.end_encoding();
                cmd.commit();
                *self.last_cb.borrow_mut() = Some(cmd.to_owned());
            });

            MetalBuffer {
                buffer: output_buf,
                len: total_dim,
            }
        } else {
            // Batched prefill
            let pipeline = self.get_pipeline(
                attention_msl::KV_ATTENTION_PREFILL_MSL,
                "kv_attention_prefill",
            );
            let output_buf = self.make_buffer_empty(q_len * total_dim * 4);
            let params = self.make_buffer_u32(&[
                cache_start as u32,
                q_len as u32,
                n_heads as u32,
                n_kv_heads as u32,
                head_dim as u32,
            ]);

            autoreleasepool(|| {
                let cmd = self.queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&pipeline);
                enc.set_buffer(0, Some(&q.buffer), 0);
                enc.set_buffer(1, Some(&k_cache.buffer), 0);
                enc.set_buffer(2, Some(&v_cache.buffer), 0);
                enc.set_buffer(3, Some(&output_buf), 0);
                enc.set_buffer(4, Some(&params), 0);

                enc.dispatch_thread_groups(
                    MTLSize::new(q_len as u64, n_heads as u64, 1),
                    MTLSize::new(tg_size, 1, 1),
                );
                enc.end_encoding();
                cmd.commit();
                *self.last_cb.borrow_mut() = Some(cmd.to_owned());
            });

            MetalBuffer {
                buffer: output_buf,
                len: q_len * total_dim,
            }
        }
    }

    fn transpose_2d(&self, buf: &MetalBuffer, rows: usize, cols: usize) -> MetalBuffer {
        let pipeline = self.get_pipeline(backward_msl::TRANSPOSE_2D_MSL, "transpose_2d");
        let output_buf = self.make_buffer_empty(rows * cols * 4);
        let params = self.make_buffer_u32(&[rows as u32, cols as u32]);

        self.dispatch_2d(
            &pipeline,
            &[&buf.buffer, &output_buf, &params],
            (cols as u64, rows as u64),
            (16, 16),
        );

        MetalBuffer { buffer: output_buf, len: rows * cols }
    }

    fn softmax_backward(
        &self,
        softmax_out: &MetalBuffer,
        grad_output: &MetalBuffer,
        n_rows: usize,
        row_len: usize,
    ) -> MetalBuffer {
        let pipeline = self.get_pipeline(backward_msl::SOFTMAX_BACKWARD_MSL, "softmax_backward");
        let output_buf = self.make_buffer_empty(n_rows * row_len * 4);
        let params = self.make_buffer_u32(&[n_rows as u32, row_len as u32]);

        let tg_size = std::cmp::min(row_len as u64, 256).next_power_of_two();
        autoreleasepool(|| {
            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&softmax_out.buffer), 0);
            enc.set_buffer(1, Some(&grad_output.buffer), 0);
            enc.set_buffer(2, Some(&output_buf), 0);
            enc.set_buffer(3, Some(&params), 0);
            enc.dispatch_thread_groups(
                MTLSize::new(n_rows as u64, 1, 1),
                MTLSize::new(tg_size, 1, 1),
            );
            enc.end_encoding();
            cmd.commit();
            *self.last_cb.borrow_mut() = Some(cmd.to_owned());
        });

        MetalBuffer { buffer: output_buf, len: n_rows * row_len }
    }

    fn rms_norm_backward(
        &self,
        input: &MetalBuffer,
        weight: &MetalBuffer,
        grad_output: &MetalBuffer,
        n_groups: usize,
        dim: usize,
        eps: f32,
    ) -> (MetalBuffer, MetalBuffer) {
        let pipeline = self.get_pipeline(backward_msl::RMS_NORM_BACKWARD_MSL, "rms_norm_backward");
        let grad_input_raw = self.make_buffer_empty(n_groups * dim * 4);
        // Zero-init grad_weight for atomic accumulation
        let grad_weight_raw = self.make_buffer(&vec![0.0f32; dim]);
        let params = self.make_buffer_u32(&[n_groups as u32, dim as u32, eps.to_bits()]);

        let tg_size = std::cmp::min(dim as u64, 256).next_power_of_two();
        autoreleasepool(|| {
            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&input.buffer), 0);
            enc.set_buffer(1, Some(&weight.buffer), 0);
            enc.set_buffer(2, Some(&grad_output.buffer), 0);
            enc.set_buffer(3, Some(&grad_input_raw), 0);
            enc.set_buffer(4, Some(&grad_weight_raw), 0);
            enc.set_buffer(5, Some(&params), 0);
            enc.dispatch_thread_groups(
                MTLSize::new(n_groups as u64, 1, 1),
                MTLSize::new(tg_size, 1, 1),
            );
            enc.end_encoding();
            cmd.commit();
            *self.last_cb.borrow_mut() = Some(cmd.to_owned());
        });

        (
            MetalBuffer { buffer: grad_input_raw, len: n_groups * dim },
            MetalBuffer { buffer: grad_weight_raw, len: dim },
        )
    }

    fn embedding_backward(
        &self,
        grad_output: &MetalBuffer,
        ids: &MetalBuffer,
        vocab_size: usize,
        seq_len: usize,
        dim: usize,
    ) -> MetalBuffer {
        let pipeline = self.get_pipeline(backward_msl::EMBEDDING_BACKWARD_MSL, "embedding_backward");
        // Zero-init for atomic accumulation
        let grad_weight_raw = self.make_buffer(&vec![0.0f32; vocab_size * dim]);
        let params = self.make_buffer_u32(&[vocab_size as u32, seq_len as u32, dim as u32]);

        self.dispatch(
            &pipeline,
            &[&grad_output.buffer, &ids.buffer, &grad_weight_raw, &params],
            seq_len as u64,
        );

        MetalBuffer { buffer: grad_weight_raw, len: vocab_size * dim }
    }

    fn causal_attention_backward(
        &self,
        grad_output: &MetalBuffer,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        seq_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> (MetalBuffer, MetalBuffer, MetalBuffer) {
        assert!(seq_len <= 2048, "causal_attention_backward: seq_len {} exceeds MAX_SEQ 2048", seq_len);
        let pipeline = self.get_pipeline(
            backward_msl::CAUSAL_ATTENTION_BACKWARD_MSL,
            "causal_attention_backward",
        );
        let total_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        let grad_q_raw = self.make_buffer_empty(seq_len * total_dim * 4);
        // Zero-init for atomic accumulation
        let grad_k_raw = self.make_buffer(&vec![0.0f32; seq_len * kv_dim]);
        let grad_v_raw = self.make_buffer(&vec![0.0f32; seq_len * kv_dim]);
        let params = self.make_buffer_u32(&[
            seq_len as u32, n_heads as u32, n_kv_heads as u32, head_dim as u32,
        ]);

        let tg_size = std::cmp::min(head_dim as u64, 256).next_power_of_two();

        autoreleasepool(|| {
            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&grad_output.buffer), 0);
            enc.set_buffer(1, Some(&q.buffer), 0);
            enc.set_buffer(2, Some(&k.buffer), 0);
            enc.set_buffer(3, Some(&v.buffer), 0);
            enc.set_buffer(4, Some(&grad_q_raw), 0);
            enc.set_buffer(5, Some(&grad_k_raw), 0);
            enc.set_buffer(6, Some(&grad_v_raw), 0);
            enc.set_buffer(7, Some(&params), 0);

            enc.dispatch_thread_groups(
                MTLSize::new(seq_len as u64, n_heads as u64, 1),
                MTLSize::new(tg_size, 1, 1),
            );
            enc.end_encoding();
            cmd.commit();
            *self.last_cb.borrow_mut() = Some(cmd.to_owned());
        });

        (
            MetalBuffer { buffer: grad_q_raw, len: seq_len * total_dim },
            MetalBuffer { buffer: grad_k_raw, len: seq_len * kv_dim },
            MetalBuffer { buffer: grad_v_raw, len: seq_len * kv_dim },
        )
    }

    fn cross_entropy_forward_backward(
        &self,
        logits: &MetalBuffer,
        targets: &MetalBuffer,
        n_positions: usize,
        vocab_size: usize,
        pad_id: u32,
    ) -> (f32, MetalBuffer) {
        // Pre-count non-padded positions on CPU (targets are small)
        let target_data = targets.to_vec();
        let count = target_data.iter().filter(|t| t.to_bits() != pad_id).count() as u32;

        let pipeline = self.get_pipeline(backward_msl::CROSS_ENTROPY_MSL, "cross_entropy_fwd_bwd");
        let grad_buf = self.make_buffer_empty(n_positions * vocab_size * 4);
        let loss_raw = self.make_buffer(&[0.0f32]);
        let params = self.make_buffer_u32(&[n_positions as u32, vocab_size as u32, pad_id, count]);

        let tg_size = std::cmp::min(vocab_size as u64, 256).next_power_of_two();
        autoreleasepool(|| {
            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&logits.buffer), 0);
            enc.set_buffer(1, Some(&targets.buffer), 0);
            enc.set_buffer(2, Some(&grad_buf), 0);
            enc.set_buffer(3, Some(&loss_raw), 0);
            enc.set_buffer(4, Some(&params), 0);
            enc.dispatch_thread_groups(
                MTLSize::new(n_positions as u64, 1, 1),
                MTLSize::new(tg_size, 1, 1),
            );
            enc.end_encoding();
            cmd.commit();
            *self.last_cb.borrow_mut() = Some(cmd.to_owned());
        });

        self.sync();
        let loss_ptr = loss_raw.contents() as *const f32;
        let loss = unsafe { *loss_ptr };
        (loss, MetalBuffer { buffer: grad_buf, len: n_positions * vocab_size })
    }

    fn sync(&self) {
        if let Some(cb) = self.last_cb.borrow_mut().take() {
            cb.wait_until_completed();
        }
    }

    fn copy_buffer(&self, src: &MetalBuffer) -> MetalBuffer {
        let dst = self.make_buffer_empty(src.len * 4);
        autoreleasepool(|| {
            let cmd = self.queue.new_command_buffer();
            let blit = cmd.new_blit_command_encoder();
            blit.copy_from_buffer(&src.buffer, 0, &dst, 0, (src.len * 4) as u64);
            blit.end_encoding();
            cmd.commit();
            *self.last_cb.borrow_mut() = Some(cmd.to_owned());
        });
        MetalBuffer { buffer: dst, len: src.len }
    }

    fn bias_add(&self, matrix: &MetalBuffer, bias: &MetalBuffer, numel: usize, dim: usize) -> MetalBuffer {
        let src = r#"
#include <metal_stdlib>
using namespace metal;

kernel void bias_add(
    device const float* matrix [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const uint* params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint numel = params[0];
    uint dim = params[1];
    if (gid >= numel) return;
    output[gid] = matrix[gid] + bias[gid % dim];
}
"#;
        let pipeline = self.get_pipeline(src, "bias_add");
        let output_buf = self.make_buffer_empty(numel * 4);
        let params = self.make_buffer_u32(&[numel as u32, dim as u32]);

        self.dispatch(
            &pipeline,
            &[&matrix.buffer, &bias.buffer, &output_buf, &params],
            numel as u64,
        );

        MetalBuffer { buffer: output_buf, len: numel }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::CpuDevice;
    use crate::device::ComputeDevice;

    fn get_metal_device() -> MetalDevice {
        MetalDevice::new().expect("Metal device not available")
    }

    #[test]
    fn metal_upload_download() {
        let dev = get_metal_device();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let buf = dev.upload(&data);
        assert_eq!(dev.download(&buf), data);
    }

    #[test]
    fn metal_elementwise_add() {
        let dev = get_metal_device();
        let a = dev.upload(&[1.0, 2.0, 3.0]);
        let b = dev.upload(&[4.0, 5.0, 6.0]);
        let c = dev.elementwise(&[&a, &b], 3, &|vars| vars[0] + vars[1]);
        let out = dev.download(&c);
        assert!((out[0] - 5.0).abs() < 1e-5);
        assert!((out[1] - 7.0).abs() < 1e-5);
        assert!((out[2] - 9.0).abs() < 1e-5);
    }

    #[test]
    fn metal_matmul_vs_cpu() {
        let metal = get_metal_device();
        let cpu = CpuDevice::new();

        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b_data = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2

        let a_m = metal.upload(&a_data);
        let b_m = metal.upload(&b_data);
        let c_m = metal.matmul(&a_m, &b_m, 2, 3, 2);
        let metal_out = metal.download(&c_m);

        let a_c = cpu.upload(&a_data);
        let b_c = cpu.upload(&b_data);
        let c_c = cpu.matmul(&a_c, &b_c, 2, 3, 2);
        let cpu_out = cpu.download(&c_c);

        for i in 0..4 {
            assert!(
                (metal_out[i] - cpu_out[i]).abs() < 1e-3,
                "matmul mismatch at {i}: metal={} cpu={}",
                metal_out[i],
                cpu_out[i]
            );
        }
    }

    #[test]
    fn metal_softmax_vs_cpu() {
        let metal = get_metal_device();
        let cpu = CpuDevice::new();

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let m_buf = metal.upload(&data);
        let m_out = metal.download(&metal.softmax(&m_buf, 2, 3));

        let c_buf = cpu.upload(&data);
        let c_out = cpu.download(&cpu.softmax(&c_buf, 2, 3));

        for i in 0..6 {
            assert!(
                (m_out[i] - c_out[i]).abs() < 1e-5,
                "softmax mismatch at {i}: metal={} cpu={}",
                m_out[i],
                c_out[i]
            );
        }
    }

    #[test]
    fn metal_rms_norm_vs_cpu() {
        let metal = get_metal_device();
        let cpu = CpuDevice::new();

        let data = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0];

        let m_data = metal.upload(&data);
        let m_weight = metal.upload(&weight);
        let m_out = metal.download(&metal.rms_norm(&m_data, &m_weight, 2, 2, 1e-5));

        let c_data = cpu.upload(&data);
        let c_weight = cpu.upload(&weight);
        let c_out = cpu.download(&cpu.rms_norm(&c_data, &c_weight, 2, 2, 1e-5));

        for i in 0..4 {
            assert!(
                (m_out[i] - c_out[i]).abs() < 1e-4,
                "rms_norm mismatch at {i}: metal={} cpu={}",
                m_out[i],
                c_out[i]
            );
        }
    }

    #[test]
    fn metal_embedding() {
        let metal = get_metal_device();
        let weight = metal.upload(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let ids = metal.upload_u32(&[2, 0, 1]);
        let result = metal.embedding(&weight, &ids, 3, 2);
        let out = metal.download(&result);
        assert!((out[0] - 0.5).abs() < 1e-6);
        assert!((out[1] - 0.6).abs() < 1e-6);
        assert!((out[2] - 0.1).abs() < 1e-6);
        assert!((out[3] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn metal_kv_attention_vs_cpu() {
        let metal = get_metal_device();
        let cpu = CpuDevice::new();

        let q_data = vec![1.0, 0.0];
        let k_data = vec![1.0, 0.0, 0.0, 1.0];
        let v_data = vec![1.0, 2.0, 3.0, 4.0];

        let m_q = metal.upload(&q_data);
        let m_k = metal.upload(&k_data);
        let m_v = metal.upload(&v_data);
        let m_out = metal.download(&metal.kv_attention(&m_q, &m_k, &m_v, 1, 1, 1, 1, 2));

        let c_q = cpu.upload(&q_data);
        let c_k = cpu.upload(&k_data);
        let c_v = cpu.upload(&v_data);
        let c_out = cpu.download(&cpu.kv_attention(&c_q, &c_k, &c_v, 1, 1, 1, 1, 2));

        for i in 0..2 {
            assert!(
                (m_out[i] - c_out[i]).abs() < 1e-3,
                "kv_attention mismatch at {i}: metal={} cpu={}",
                m_out[i],
                c_out[i]
            );
        }
    }

    #[test]
    fn metal_transpose_2d_vs_cpu() {
        let metal = get_metal_device();
        let cpu = CpuDevice::new();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3

        let m_buf = metal.upload(&data);
        let m_out = metal.download(&metal.transpose_2d(&m_buf, 2, 3));

        let c_buf = cpu.upload(&data);
        let c_out = cpu.download(&cpu.transpose_2d(&c_buf, 2, 3));

        assert_eq!(m_out.len(), c_out.len());
        for i in 0..m_out.len() {
            assert!((m_out[i] - c_out[i]).abs() < 1e-5,
                "transpose mismatch at {i}: metal={} cpu={}", m_out[i], c_out[i]);
        }
    }

    #[test]
    fn metal_softmax_backward_vs_cpu() {
        let metal = get_metal_device();
        let cpu = CpuDevice::new();

        let sm_data = vec![0.2, 0.3, 0.5, 0.1, 0.6, 0.3];
        let grad_data = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];

        let m_sm = metal.upload(&sm_data);
        let m_grad = metal.upload(&grad_data);
        let m_out = metal.download(&metal.softmax_backward(&m_sm, &m_grad, 2, 3));

        let c_sm = cpu.upload(&sm_data);
        let c_grad = cpu.upload(&grad_data);
        let c_out = cpu.download(&cpu.softmax_backward(&c_sm, &c_grad, 2, 3));

        for i in 0..m_out.len() {
            assert!((m_out[i] - c_out[i]).abs() < 1e-4,
                "softmax_backward mismatch at {i}: metal={} cpu={}", m_out[i], c_out[i]);
        }
    }

    #[test]
    fn metal_embedding_backward_vs_cpu() {
        let metal = get_metal_device();
        let cpu = CpuDevice::new();

        let grad_data = vec![1.0, 2.0, 3.0, 0.1, 0.2, 0.3];
        let ids: Vec<u32> = vec![1, 3];

        let m_grad = metal.upload(&grad_data);
        let m_ids = metal.upload_u32(&ids);
        let m_out = metal.download(&metal.embedding_backward(&m_grad, &m_ids, 4, 2, 3));

        let c_grad = cpu.upload(&grad_data);
        let c_ids = cpu.upload_u32(&ids);
        let c_out = cpu.download(&cpu.embedding_backward(&c_grad, &c_ids, 4, 2, 3));

        for i in 0..m_out.len() {
            assert!((m_out[i] - c_out[i]).abs() < 1e-4,
                "embedding_backward mismatch at {i}: metal={} cpu={}", m_out[i], c_out[i]);
        }
    }

    #[test]
    fn metal_cross_entropy_vs_cpu() {
        let metal = get_metal_device();
        let cpu = CpuDevice::new();

        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let targets: Vec<u32> = vec![2, 0];

        let m_logits = metal.upload(&logits);
        let m_targets = metal.upload_u32(&targets);
        let (m_loss, m_grad_buf) = metal.cross_entropy_forward_backward(&m_logits, &m_targets, 2, 3, 99);
        let m_grad = metal.download(&m_grad_buf);

        let c_logits = cpu.upload(&logits);
        let c_targets = cpu.upload_u32(&targets);
        let (c_loss, c_grad_buf) = cpu.cross_entropy_forward_backward(&c_logits, &c_targets, 2, 3, 99);
        let c_grad = cpu.download(&c_grad_buf);

        assert!((m_loss - c_loss).abs() < 1e-3,
            "cross_entropy loss mismatch: metal={} cpu={}", m_loss, c_loss);
        for i in 0..m_grad.len() {
            assert!((m_grad[i] - c_grad[i]).abs() < 1e-3,
                "cross_entropy grad mismatch at {i}: metal={} cpu={}", m_grad[i], c_grad[i]);
        }
    }

    #[test]
    fn metal_rms_norm_backward_vs_cpu() {
        let metal = get_metal_device();
        let cpu = CpuDevice::new();

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0];
        let grad_out = vec![1.0, 0.0, 0.0, 1.0];

        let m_input = metal.upload(&input);
        let m_weight = metal.upload(&weight);
        let m_grad = metal.upload(&grad_out);
        let (m_gi, m_gw) = metal.rms_norm_backward(&m_input, &m_weight, &m_grad, 2, 2, 1e-5);
        let m_gi_v = metal.download(&m_gi);
        let m_gw_v = metal.download(&m_gw);

        let c_input = cpu.upload(&input);
        let c_weight = cpu.upload(&weight);
        let c_grad = cpu.upload(&grad_out);
        let (c_gi, c_gw) = cpu.rms_norm_backward(&c_input, &c_weight, &c_grad, 2, 2, 1e-5);
        let c_gi_v = cpu.download(&c_gi);
        let c_gw_v = cpu.download(&c_gw);

        for i in 0..m_gi_v.len() {
            assert!((m_gi_v[i] - c_gi_v[i]).abs() < 1e-3,
                "rms_norm_backward grad_input mismatch at {i}: metal={} cpu={}", m_gi_v[i], c_gi_v[i]);
        }
        for i in 0..m_gw_v.len() {
            assert!((m_gw_v[i] - c_gw_v[i]).abs() < 1e-3,
                "rms_norm_backward grad_weight mismatch at {i}: metal={} cpu={}", m_gw_v[i], c_gw_v[i]);
        }
    }

    #[test]
    fn metal_causal_attention_backward_vs_cpu() {
        let metal = get_metal_device();
        let cpu = CpuDevice::new();

        let seq_len = 4;
        let n_heads = 2;
        let n_kv_heads = 1;
        let head_dim = 4;
        let total_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        // Deterministic pseudo-random data
        let q_data: Vec<f32> = (0..seq_len * total_dim)
            .map(|i| ((i * 7 + 3) % 13) as f32 / 13.0 - 0.5).collect();
        let k_data: Vec<f32> = (0..seq_len * kv_dim)
            .map(|i| ((i * 11 + 5) % 17) as f32 / 17.0 - 0.5).collect();
        let v_data: Vec<f32> = (0..seq_len * kv_dim)
            .map(|i| ((i * 13 + 7) % 19) as f32 / 19.0 - 0.5).collect();
        let go_data: Vec<f32> = (0..seq_len * total_dim)
            .map(|i| ((i * 3 + 11) % 23) as f32 / 23.0 - 0.5).collect();

        // Metal
        let m_go = metal.upload(&go_data);
        let m_q = metal.upload(&q_data);
        let m_k = metal.upload(&k_data);
        let m_v = metal.upload(&v_data);
        let (m_gq, m_gk, m_gv) = metal.causal_attention_backward(
            &m_go, &m_q, &m_k, &m_v, seq_len, n_heads, n_kv_heads, head_dim,
        );
        let m_gq_v = metal.download(&m_gq);
        let m_gk_v = metal.download(&m_gk);
        let m_gv_v = metal.download(&m_gv);

        // CPU
        let c_go = cpu.upload(&go_data);
        let c_q = cpu.upload(&q_data);
        let c_k = cpu.upload(&k_data);
        let c_v = cpu.upload(&v_data);
        let (c_gq, c_gk, c_gv) = cpu.causal_attention_backward(
            &c_go, &c_q, &c_k, &c_v, seq_len, n_heads, n_kv_heads, head_dim,
        );
        let c_gq_v = cpu.download(&c_gq);
        let c_gk_v = cpu.download(&c_gk);
        let c_gv_v = cpu.download(&c_gv);

        for i in 0..m_gq_v.len() {
            assert!((m_gq_v[i] - c_gq_v[i]).abs() < 1e-3,
                "grad_Q mismatch at {i}: metal={} cpu={}", m_gq_v[i], c_gq_v[i]);
        }
        for i in 0..m_gk_v.len() {
            assert!((m_gk_v[i] - c_gk_v[i]).abs() < 1e-3,
                "grad_K mismatch at {i}: metal={} cpu={}", m_gk_v[i], c_gk_v[i]);
        }
        for i in 0..m_gv_v.len() {
            assert!((m_gv_v[i] - c_gv_v[i]).abs() < 1e-3,
                "grad_V mismatch at {i}: metal={} cpu={}", m_gv_v[i], c_gv_v[i]);
        }
    }

    #[test]
    fn metal_causal_attention_backward_mha_vs_cpu() {
        // Test with n_heads == n_kv_heads (standard MHA, no GQA)
        let metal = get_metal_device();
        let cpu = CpuDevice::new();

        let seq_len = 8;
        let n_heads = 4;
        let n_kv_heads = 4;
        let head_dim = 8;
        let total_dim = n_heads * head_dim;

        let q_data: Vec<f32> = (0..seq_len * total_dim)
            .map(|i| ((i * 7 + 3) % 13) as f32 / 13.0 - 0.5).collect();
        let k_data: Vec<f32> = (0..seq_len * total_dim)
            .map(|i| ((i * 11 + 5) % 17) as f32 / 17.0 - 0.5).collect();
        let v_data: Vec<f32> = (0..seq_len * total_dim)
            .map(|i| ((i * 13 + 7) % 19) as f32 / 19.0 - 0.5).collect();
        let go_data: Vec<f32> = (0..seq_len * total_dim)
            .map(|i| ((i * 3 + 11) % 23) as f32 / 23.0 - 0.5).collect();

        let m_go = metal.upload(&go_data);
        let m_q = metal.upload(&q_data);
        let m_k = metal.upload(&k_data);
        let m_v = metal.upload(&v_data);
        let (m_gq, m_gk, m_gv) = metal.causal_attention_backward(
            &m_go, &m_q, &m_k, &m_v, seq_len, n_heads, n_kv_heads, head_dim,
        );
        let m_gq_v = metal.download(&m_gq);
        let m_gk_v = metal.download(&m_gk);
        let m_gv_v = metal.download(&m_gv);

        let c_go = cpu.upload(&go_data);
        let c_q = cpu.upload(&q_data);
        let c_k = cpu.upload(&k_data);
        let c_v = cpu.upload(&v_data);
        let (c_gq, c_gk, c_gv) = cpu.causal_attention_backward(
            &c_go, &c_q, &c_k, &c_v, seq_len, n_heads, n_kv_heads, head_dim,
        );
        let c_gq_v = cpu.download(&c_gq);
        let c_gk_v = cpu.download(&c_gk);
        let c_gv_v = cpu.download(&c_gv);

        for i in 0..m_gq_v.len() {
            assert!((m_gq_v[i] - c_gq_v[i]).abs() < 1e-3,
                "MHA grad_Q mismatch at {i}: metal={} cpu={}", m_gq_v[i], c_gq_v[i]);
        }
        for i in 0..m_gk_v.len() {
            assert!((m_gk_v[i] - c_gk_v[i]).abs() < 1e-3,
                "MHA grad_K mismatch at {i}: metal={} cpu={}", m_gk_v[i], c_gk_v[i]);
        }
        for i in 0..m_gv_v.len() {
            assert!((m_gv_v[i] - c_gv_v[i]).abs() < 1e-3,
                "MHA grad_V mismatch at {i}: metal={} cpu={}", m_gv_v[i], c_gv_v[i]);
        }
    }

    #[test]
    fn metal_kv_attention_batched_vs_cpu() {
        let metal = get_metal_device();
        let cpu = CpuDevice::new();

        let n_heads = 2;
        let n_kv_heads = 1;
        let head_dim = 4;
        let q_len = 4;
        let total_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        let q_data: Vec<f32> = (0..q_len * total_dim).map(|i| ((i * 7 + 3) % 13) as f32 / 13.0).collect();
        let kv_data: Vec<f32> = (0..q_len * kv_dim).map(|i| ((i * 11 + 5) % 17) as f32 / 17.0).collect();
        let v_data: Vec<f32> = (0..q_len * kv_dim).map(|i| ((i * 13 + 7) % 19) as f32 / 19.0).collect();

        let m_q = metal.upload(&q_data);
        let m_k = metal.upload(&kv_data);
        let m_v = metal.upload(&v_data);
        let m_out = metal.download(&metal.kv_attention(&m_q, &m_k, &m_v, 0, q_len, n_heads, n_kv_heads, head_dim));

        let c_q = cpu.upload(&q_data);
        let c_k = cpu.upload(&kv_data);
        let c_v = cpu.upload(&v_data);
        let c_out = cpu.download(&cpu.kv_attention(&c_q, &c_k, &c_v, 0, q_len, n_heads, n_kv_heads, head_dim));

        assert_eq!(m_out.len(), c_out.len());
        for i in 0..m_out.len() {
            assert!(
                (m_out[i] - c_out[i]).abs() < 1e-3,
                "batched kv_attention mismatch at {i}: metal={} cpu={}",
                m_out[i],
                c_out[i]
            );
        }
    }
}
