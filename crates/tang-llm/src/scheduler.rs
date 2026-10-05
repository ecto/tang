//! Z-Image-Turbo's static-shift flow-matching Euler schedule.
use anyhow::{ensure, Result};
#[derive(Clone, Debug)]
pub struct Euler {
    pub sigmas: Vec<f32>,
}
impl Euler {
    pub fn new(steps: usize, shift: f32) -> Result<Self> {
        ensure!(
            (1..=100).contains(&steps) && shift.is_finite() && shift > 0.,
            "invalid steps or shift"
        );
        // Pipeline supplies linspace(1, 1/steps, steps), not training sigma_min.
        let mut sigmas: Vec<_> = (0..steps)
            .map(|i| {
                let raw = if steps == 1 {
                    1.
                } else {
                    1. - i as f32 / steps as f32
                };
                shift * raw / (1. + (shift - 1.) * raw)
            })
            .collect();
        sigmas.push(0.);
        Ok(Self { sigmas })
    }
    pub fn normalized_time(&self, step: usize) -> Result<f32> {
        ensure!(step + 1 < self.sigmas.len(), "invalid step");
        Ok(1. - self.sigmas[step])
    }
    /// The transformer predicts the negative of scheduler velocity.
    pub fn step(&self, step: usize, prediction: &[f32], latent: &mut [f32]) -> Result<()> {
        ensure!(
            step + 1 < self.sigmas.len() && prediction.len() == latent.len(),
            "invalid Euler step"
        );
        let delta = self.sigmas[step] - self.sigmas[step + 1];
        for (x, p) in latent.iter_mut().zip(prediction) {
            *x += delta * p;
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn matches_shifted_eight_step_schedule() {
        let e = Euler::new(8, 3.).unwrap();
        let reference = [
            1., 0.95454544, 0.9, 0.8333333, 0.75, 0.64285713, 0.5, 0.3, 0.,
        ];
        for (a, b) in e.sigmas.iter().zip(reference) {
            assert!((a - b).abs() < 1e-6);
        }
        let mut x = [2., 3.];
        e.step(0, &[1., -1.], &mut x).unwrap();
        assert!((x[0] - 2.0454545).abs() < 1e-6);
        assert!(Euler::new(0, 3.).is_err());
        assert!(e.normalized_time(8).is_err());
    }
}
