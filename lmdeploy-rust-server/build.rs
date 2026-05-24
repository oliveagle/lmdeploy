use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let build_dir = manifest_dir.parent().unwrap().join("build/lib");

    // CUDA library paths
    let cuda_lib = "/usr/local/cuda/lib64";
    let cuda_stubs = "/usr/local/cuda/targets/x86_64-linux/lib/stubs";
    let system_cuda = "/usr/lib/x86_64-linux-gnu";

    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-search=native={}", cuda_lib);
    println!("cargo:rustc-link-search=native={}", cuda_stubs);
    println!("cargo:rustc-link-search=native={}", system_cuda);
    println!("cargo:rustc-link-lib=dylib=turbomind_c");
    println!("cargo:rustc-link-lib=dylib=tm_shim");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublasLt");
    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rerun-if-changed={}", build_dir.display());

    // Generate gRPC code from protos
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile(&["proto/lmdeploy.proto"], &["proto/"])
        .expect("Failed to compile protos");
}
