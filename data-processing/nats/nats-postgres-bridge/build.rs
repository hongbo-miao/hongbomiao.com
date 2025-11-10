fn main() {
    capnpc::CompilerCommand::new()
        .src_prefix("src")
        .file("src/transcript.capnp")
        .run()
        .expect("Failed to compile Cap'n Proto schema");
}
