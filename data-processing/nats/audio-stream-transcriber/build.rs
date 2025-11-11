fn main() {
    capnpc::CompilerCommand::new()
        .src_prefix("src")
        .file("src/transcription.capnp")
        .run()
        .expect("Failed to compile Cap'n Proto schema");
}
