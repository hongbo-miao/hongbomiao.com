use std::{env, fs::File, io::Write, path::Path};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let generated_code = dust_dds_gen::compile_idl(Path::new("src/hm_message.idl"))?;

    let out_dir = env::var("OUT_DIR")?;
    let output_path = Path::new(&out_dir).join("hm_message.rs");

    let mut file = File::create(output_path)?;
    file.write_all(generated_code.as_bytes())?;

    Ok(())
}
