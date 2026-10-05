//! Compare every DiT boundary to the official fp32 diffusers fixture.
use anyhow::{ensure, Context, Result};
use serde::Deserialize;
use std::{collections::BTreeMap, path::Path};
use tang_compute::{ComputeDevice, CpuDevice};
use tang_llm::{dit::DiT, Dtype};
#[derive(Deserialize)]
struct Case {
    height: usize,
    width: usize,
    #[serde(default)]
    caption_tokens: usize,
    #[serde(default)]
    t: f32,
    latent: Vec<f32>,
    #[serde(default)]
    caption: Vec<f32>,
    traces: BTreeMap<String, Vec<f32>>,
    output: Vec<f32>,
}
fn compare(
    name: &str,
    actual: &[f32],
    expected: &[f32],
    tolerance: f32,
    relative: f32,
) -> Result<()> {
    ensure!(
        actual.len() == expected.len(),
        "{name}: shape {} != {}",
        actual.len(),
        expected.len()
    );
    let mut max = 0f32;
    let mut worst = 0usize;
    let mut max_ratio = 0f32;
    let mut squared_error = 0f64;
    let mut squared_reference = 0f64;
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        ensure!(a.is_finite(), "{name}: nonfinite value at {i}");
        let err = (a - e).abs();
        squared_error += (a as f64 - e as f64).powi(2);
        squared_reference += (e as f64).powi(2);
        max_ratio = max_ratio.max(err / (tolerance + relative * e.abs()));
        if err > max {
            max = err;
            worst = i;
        }
    }
    let relative_l2 = (squared_error / squared_reference.max(f64::MIN_POSITIVE)).sqrt();
    println!("{name}: max abs {max:.8} at {worst}; relative L2 {relative_l2:.8}; max tolerance ratio {max_ratio:.6}");
    ensure!(
        max_ratio <= 1.,
        "{name}: actual {} expected {} at {worst}",
        actual[worst],
        expected[worst]
    );
    Ok(())
}
fn run<D: ComputeDevice>(
    dev: D,
    root: &Path,
    file: &Path,
    tolerance: f32,
    relative: f32,
) -> Result<()> {
    let vae = root.join("vae-fixture").exists();
    let model = if vae {
        None
    } else {
        Some(DiT::load(&dev, root, Dtype::F32)?)
    };
    let case: Case = serde_json::from_slice(&std::fs::read(file)?)?;
    let mut seen = BTreeMap::new();
    let mut trace = |name: &str, tensor: &D::Buffer| {
        seen.insert(name.to_owned(), dev.download(tensor));
    };
    let out = if let Some(model) = model {
        let cap = dev.upload(&case.caption);
        model.forward_trace(
            &dev,
            &case.latent,
            case.height,
            case.width,
            &cap,
            case.caption_tokens,
            case.t,
            &mut trace,
        )?
    } else {
        tang_llm::vae::Vae::load(&dev, root)?.decode_trace(
            &dev,
            &case.latent,
            case.height,
            case.width,
            &mut trace,
        )?
    };
    let mut failures = 0;
    for (name, expected) in case.traces {
        if let Err(e) = compare(
            &name,
            seen.get(&name).context("missing trace")?,
            &expected,
            tolerance,
            relative,
        ) {
            eprintln!("{e:#}");
            failures += 1;
        }
    }
    if let Err(e) = compare("output", &out, &case.output, tolerance, relative) {
        eprintln!("{e:#}");
        failures += 1;
    }
    ensure!(failures == 0, "{failures} boundaries exceeded tolerance");
    Ok(())
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let root = Path::new(args.get(1).context("fixture directory required")?);
    let metal = args.iter().any(|a| a == "--metal");
    let relative = if let Some(index) = args.iter().position(|a| a == "--relative-tolerance") {
        args.get(index + 1)
            .context("relative tolerance missing")?
            .parse::<f32>()?
    } else {
        0.
    };
    ensure!(
        relative.is_finite() && relative >= 0.,
        "invalid relative tolerance"
    );
    let files: Vec<&str> = if let Some(index) = args.iter().position(|a| a == "--case") {
        vec![args
            .get(index + 1)
            .context("--case requires a filename")?
            .as_str()]
    } else if root.join("vae-fixture").exists() {
        vec!["case-4.json", "case-32.json"]
    } else {
        vec!["case-8.json", "case-32.json"]
    };
    for file in files {
        println!("{file}:");
        if metal {
            #[cfg(feature = "metal")]
            run(
                tang_compute::MetalDevice::new().context("Metal unavailable")?,
                root,
                &root.join(file),
                0.001,
                relative,
            )?;
            #[cfg(not(feature = "metal"))]
            anyhow::bail!("rebuild with --features metal");
        } else {
            run(CpuDevice::new(), root, &root.join(file), 0.0001, relative)?;
        }
    }
    Ok(())
}
