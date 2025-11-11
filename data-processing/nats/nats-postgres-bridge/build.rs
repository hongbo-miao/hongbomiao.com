use std::io::Result;

fn main() -> Result<()> {
    capnpc::CompilerCommand::new()
        .src_prefix("src")
        .file("src/transcription.capnp")
        .run()
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::Other, error))?;
    Ok(())
}
