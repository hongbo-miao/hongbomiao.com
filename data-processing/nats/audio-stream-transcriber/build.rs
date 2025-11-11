use std::io::Result;

fn main() -> Result<()> {
    capnpc::CompilerCommand::new()
        .src_prefix("src")
        .file("src/transcription.capnp")
        .run()
        .expect("Failed to compile Cap'n Proto schema");
    Ok(())
}
