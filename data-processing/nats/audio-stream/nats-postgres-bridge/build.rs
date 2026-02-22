use std::io::Result;

fn main() -> Result<()> {
    capnpc::CompilerCommand::new()
        .src_prefix("../schemas")
        .file("../schemas/transcription.capnp")
        .run()
        .map_err(std::io::Error::other)?;
    Ok(())
}
