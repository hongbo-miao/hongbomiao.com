use std::io::Result;

fn main() -> Result<()> {
    println!("cargo:warning=Build script is running!");
    let mut prost_build = prost_build::Config::new();
    prost_build.protoc_arg("--experimental_allow_proto3_optional");
    prost_build.compile_protos(&["src/protos/production.iot.motor.proto"], &["src/protos/"])?;
    Ok(())
}
