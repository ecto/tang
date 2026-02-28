fn main() {
    // Link Accelerate framework on macOS for BLAS (cblas_sgemm)
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}
