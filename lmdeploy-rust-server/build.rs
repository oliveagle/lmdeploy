use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    // CUDA library paths
    let cuda_lib = "/usr/local/cuda-12.5/lib64";
    let cuda_stubs = "/usr/local/cuda-12.5/targets/x86_64-linux/lib/stubs";
    let system_cuda = "/usr/lib/x86_64-linux-gnu";

    // Search for libraries in multiple locations (local lib first)
    let local_lib = manifest_dir.join("lib");
    println!("cargo:rustc-link-search=native={}", local_lib.display());
    // Also search lmdeploy/lib for shim library
    let lmdeploy_lib = manifest_dir.parent().unwrap().join("lmdeploy/lib");
    println!("cargo:rustc-link-search=native={}", lmdeploy_lib.display());
    let build_lib_dir = manifest_dir.parent().unwrap().join("build/lib");
    println!("cargo:rustc-link-search=native={}", build_lib_dir.display());
    println!("cargo:rustc-link-search=native={}", cuda_lib);
    println!("cargo:rustc-link-search=native={}", cuda_stubs);
    println!("cargo:rustc-link-search=native={}", system_cuda);

    // Link libraries
    println!("cargo:rustc-link-lib=dylib=turbomind_c");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublasLt");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cuda");

    // Set rpath to find CUDA and custom libraries at runtime
    // Order matters: local lib first (has latest fix), then others
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", local_lib.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lmdeploy_lib.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", build_lib_dir.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", cuda_lib);

    println!("cargo:rerun-if-changed={}", build_lib_dir.display());

    // Generate gRPC code from protos
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(&["proto/lmdeploy.proto"], &["proto/"])
        .expect("Failed to compile protos");
}
