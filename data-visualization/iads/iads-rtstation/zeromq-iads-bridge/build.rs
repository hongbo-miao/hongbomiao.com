use std::io::Result;

fn main() -> Result<()> {
    prost_build::compile_protos(
        &["src/protos/production.iot.signals.proto"],
        &["src/protos/"],
    )?;
    Ok(())
}
