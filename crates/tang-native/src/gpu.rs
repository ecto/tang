//! The training step on a GPU: compiled once (fusion with the residual
//! stream as barriers, a liveness memory plan), parameters and Adam state
//! resident, updated in place by donation. CUDA through NVRTC when built
//! with `cuda` and a driver is present (mew), wgpu otherwise.

use crate::train::Step;
use tang_compile::tensor::{Op, TensorId};
use tang_gpu::program::{Program, Runtime};
// BackendKind is matched by path above

pub use crate::compile::{compile, input_node};

pub struct GpuTrainer {
    pub rt: Runtime,
    pub prog: Program,
    pub step: Step,
    /// compiled outputs, in [`Step::outputs`] order
    outs: Vec<TensorId>,
    /// (updated output, its input) pairs the plan could not donate in place
    copies: Vec<(TensorId, TensorId)>,
    pub t: u64,
}

impl GpuTrainer {
    /// Compile `step` for the selected GPU, starting from `values` (every
    /// graph input, by input index).
    pub fn new(step: Step, values: &[Vec<f32>]) -> Result<GpuTrainer, String> {
        let rt = Runtime::select().map_err(|e| format!("no GPU runtime: {e:?}"))?;
        let outs = step.outputs();
        let barriers: Vec<TensorId> = step
            .model
            .boundaries
            .iter()
            .chain(&step.norm_partials)
            .copied()
            .collect();
        let (g, couts) = compile(&step.model.graph, &outs, &barriers);
        let np = step.slots.len();
        let mut pairs = Vec::new();
        for (i, (p, s)) in step.model.params.iter().zip(&step.slots).enumerate() {
            for (j, src) in [p.id, s.m, s.v].into_iter().enumerate() {
                pairs.push((couts[2 + j * np + i], input_node(&g, step.input_index(src))));
            }
        }
        let lowered = rt.lower(&g, &couts);
        // in-place updates need a backend that can bind one buffer read and
        // write in one dispatch: CUDA can, wgpu can't (it copies instead)
        let donate = matches!(rt.kind(), tang_gpu::program::BackendKind::Cuda);
        let plan = g.plan_memory(&lowered, &couts, if donate { &pairs } else { &[] });
        let copies = pairs
            .iter()
            .filter(|(o, i)| !plan.inplace.iter().any(|&(a, b)| a == *o && b == *i))
            .copied()
            .collect();
        let inputs: Vec<&[f32]> = values.iter().map(|v| v.as_slice()).collect();
        let prog = Program::planned(&rt, lowered, &inputs, &plan);
        Ok(GpuTrainer {
            rt,
            prog,
            step,
            outs: couts,
            copies,
            t: 0,
        })
    }

    pub fn backend(&self) -> String {
        format!("{:?} — {}", self.rt.kind(), self.rt.name())
    }

    /// One optimizer step. Returns `(loss, grad_norm)`.
    pub fn step(&mut self, tokens: &[u32], targets: &[u32], lr: f64) -> (f32, f32) {
        self.t += 1;
        let (s, rt, prog) = (&self.step, &self.rt, &self.prog);
        let h = s.hyper;
        let tk: Vec<f32> = tokens.iter().map(|&x| x as f32).collect();
        let tg: Vec<f32> = targets.iter().map(|&x| x as f32).collect();
        prog.write(rt, prog.input(s.input_index(s.model.tokens)), &tk);
        prog.write(rt, prog.input(s.input_index(s.model.targets)), &tg);
        let hyper = [
            lr as f32,
            (1.0 - h.beta1.powi(self.t as i32)) as f32,
            (1.0 - h.beta2.powi(self.t as i32)) as f32,
        ];
        prog.write(rt, prog.input(s.input_index(s.hyper_in)), &hyper);
        prog.run(rt);
        for &(out, input) in &self.copies {
            prog.copy(rt, input, prog, out);
        }
        (
            prog.read(rt, self.outs[0])[0],
            prog.read(rt, self.outs[1])[0],
        )
    }

    /// How many of the 3 × params updates run in place.
    pub fn donated(&self) -> usize {
        3 * self.step.slots.len() - self.copies.len()
    }

    /// Current value of graph input `t` (a parameter, `m` or `v`).
    pub fn read_input(&self, t: TensorId) -> Vec<f32> {
        let k = self.step.input_index(t);
        self.prog.read(&self.rt, self.prog.input(k))
    }

    /// Overwrite graph input `t` (resuming from a checkpoint).
    pub fn write_input(&self, t: TensorId, v: &[f32]) {
        let k = self.step.input_index(t);
        self.prog.write(&self.rt, self.prog.input(k), v);
    }
}

/// The model's loss alone (no update), compiled once, for validation.
pub struct GpuEval {
    rt: Runtime,
    prog: Program,
    model: crate::Model,
    loss: TensorId,
}

impl GpuEval {
    pub fn new(model: crate::Model, values: &[Vec<f32>]) -> Result<GpuEval, String> {
        let rt = Runtime::select().map_err(|e| format!("no GPU runtime: {e:?}"))?;
        let (g, outs) = compile(
            &model.graph,
            &[model.loss],
            &model.boundaries[..model.config.layers],
        );
        let lowered = rt.lower(&g, &outs);
        let plan = g.plan_memory(&lowered, &outs, &[]);
        let inputs: Vec<&[f32]> = values.iter().map(|v| v.as_slice()).collect();
        let prog = Program::planned(&rt, lowered, &inputs, &plan);
        Ok(GpuEval {
            rt,
            prog,
            loss: outs[0],
            model,
        })
    }

    fn input_index(&self, t: TensorId) -> usize {
        match self.model.graph.nodes()[t.index()].op {
            Op::Input(k) => k,
            _ => panic!("not an input"),
        }
    }

    /// Set a parameter by name (from the trainer).
    pub fn set(&self, name: &str, v: &[f32]) {
        if let Some(p) = self.model.params.iter().find(|p| p.name == name) {
            let k = self.input_index(p.id);
            self.prog.write(&self.rt, self.prog.input(k), v);
        }
    }

    pub fn loss(&self, tokens: &[u32], targets: &[u32]) -> f32 {
        let tk: Vec<f32> = tokens.iter().map(|&x| x as f32).collect();
        let tg: Vec<f32> = targets.iter().map(|&x| x as f32).collect();
        let (rt, p) = (&self.rt, &self.prog);
        p.write(rt, p.input(self.input_index(self.model.tokens)), &tk);
        p.write(rt, p.input(self.input_index(self.model.targets)), &tg);
        p.run(rt);
        p.read(rt, self.loss)[0]
    }
}
