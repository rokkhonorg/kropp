//! Copy the bundled `onnx/` directory next to the built executable so the
//! default `--doc-orient-model onnx/bkori.onnx` (resolved relative to the exe)
//! is found during `cargo run`/`cargo test`, matching how it ships in releases.

use std::path::{Path, PathBuf};
use std::{env, fs};

fn main() {
    // Only re-run (and re-copy) when the model directory or this script changes.
    println!("cargo:rerun-if-changed=onnx");
    println!("cargo:rerun-if-changed=build.rs");

    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let src = manifest.join("onnx");
    if !src.is_dir() {
        return;
    }

    // OUT_DIR is `<target>/<profile>/build/<pkg>-<hash>/out`; the executable
    // lives at `<target>/<profile>`. Walk up to that directory.
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let Some(bin_dir) = out_dir.ancestors().nth(3) else {
        return;
    };

    if let Err(e) = copy_dir(&src, &bin_dir.join("onnx")) {
        // A failed copy isn't fatal: the binary still builds, the model just
        // won't be found in place during development.
        println!("cargo:warning=failed to copy onnx/ next to the binary: {e}");
    }
}

fn copy_dir(src: &Path, dst: &Path) -> std::io::Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let path = entry.path();
        let target = dst.join(entry.file_name());
        if path.is_dir() {
            copy_dir(&path, &target)?;
        } else {
            fs::copy(&path, &target)?;
        }
    }
    Ok(())
}
