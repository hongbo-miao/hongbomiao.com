use std::io::Result;

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=../proto/telemetry_record.proto");
    prost_build::compile_protos(&["../proto/telemetry_record.proto"], &["../proto/"])?;
    Ok(())
}
