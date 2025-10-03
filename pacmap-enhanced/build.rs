use std::env;
use std::path::Path;

fn main() {
    let project_root = env::var("CARGO_MANIFEST_DIR").unwrap();
    let lapack_dir = Path::new(&project_root).parent().unwrap().join("lapack-binaries");

    // Use manually downloaded LAPACK binaries (best solution for ease of setup)
    if lapack_dir.exists() {
        if cfg!(target_os = "windows") {
            println!("cargo:rustc-link-search=native={}", lapack_dir.display());
            println!("cargo:rustc-link-lib=libopenblas");
            println!("cargo:warning=✅ Using manually downloaded OpenBLAS from: {}", lapack_dir.display());
        } else {
            // Linux can use system packages
            println!("cargo:rustc-link-lib=openblas");
            println!("cargo:rustc-link-lib=lapack");
            println!("cargo:warning=✅ Using system OpenBLAS/LAPACK on Linux");
        }
    } else {
        println!("cargo:warning=❌ No LAPACK binaries found. Download OpenBLAS from: https://github.com/OpenMathLib/OpenBLAS/releases");
        println!("cargo:warning=Extract to ../lapack-binaries/ directory");
    }
}