fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=../schemas/audio_ingest.proto");
    tonic_prost_build::compile_protos("../schemas/audio_ingest.proto")?;
    Ok(())
}
