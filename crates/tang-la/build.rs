fn main() {
    #[cfg(feature = "accelerate")]
    {
        if std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default() == "macos" {
            println!("cargo:rustc-link-lib=framework=Accelerate");
        }
    }
}
