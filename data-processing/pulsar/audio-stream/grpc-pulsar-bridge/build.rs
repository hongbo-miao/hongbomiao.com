fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=../schemas/audio_chunk.proto");
    println!("cargo:rerun-if-changed=../schemas/audio_ingest.proto");
    tonic_prost_build::configure()
        .build_server(false)
        .build_client(false)
        .compile_protos(&["../schemas/audio_chunk.proto"], &["../schemas/"])?;
    tonic_prost_build::compile_protos("../schemas/audio_ingest.proto")?;
    Ok(())
}
