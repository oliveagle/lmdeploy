use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")?;
    let proto_file = PathBuf::from(&manifest_dir).join("proto").join("lmdeploy.proto");
    let proto_dir = proto_file.parent().unwrap();

    tonic_build::configure()
        .compile_protos(&[proto_file.as_path()], &[proto_dir])?;

    Ok(())
}
