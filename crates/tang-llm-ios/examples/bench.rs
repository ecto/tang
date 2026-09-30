//! The phone bench, on this Mac: `cargo run --release -p tang-llm-ios --example bench -- <model dir> [prompt] [gen]`.
use std::ffi::{CStr, CString};
use tang_llm_ios::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = tang_llm::resolve_model(&args[1]).unwrap();
    let n = |i: usize, d: u32| args.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    let dir = CString::new(dir.to_str().unwrap()).unwrap();
    unsafe {
        let mut e = std::ptr::null_mut();
        let h = tang_load(dir.as_ptr(), 8192, false, &mut e);
        assert!(!h.is_null(), "{}", CStr::from_ptr(e).to_str().unwrap());
        for s in [tang_info(h), tang_bench(h, n(2, 2048), n(3, 128))] {
            println!("{}", CStr::from_ptr(s).to_str().unwrap());
            tang_string_free(s);
        }
        tang_free(h);
    }
}
