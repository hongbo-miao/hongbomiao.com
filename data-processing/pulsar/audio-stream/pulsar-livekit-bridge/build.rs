use std::io::Result;

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=../schemas/audio_chunk.proto");
    prost_build::compile_protos(&["../schemas/audio_chunk.proto"], &["../schemas/"])?;
    Ok(())
}
