#[macro_use]
mod logging;
mod angle;
mod cli;
mod components;
mod debug_overlay;
mod model;
mod output;
mod pipeline;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::Parser;

use crate::cli::Args;
use crate::model::read_original_image;
use crate::output::OutputPlan;
use crate::pipeline::{Pipeline, collect_input_paths, output_dir_has_prefix};

fn main() -> Result<()> {
    let args = Args::parse();
    let input_path = PathBuf::from(&args.input);

    // Install the log-file sink before any diagnostics are emitted. An explicit
    // --log-file wins; otherwise --debug writes to a default file.
    let log_path = args
        .log_file
        .clone()
        .map(PathBuf::from)
        .or_else(|| args.debug.then(|| PathBuf::from("kropp-debug.log")));
    if let Some(path) = &log_path {
        logging::init_log_file(path)?;
    }

    let cutoff = args.validate()?;

    if input_path.is_dir() {
        return run_directory(&args, &input_path, cutoff);
    }

    if args.output_dir.is_some() {
        bail!("--output-dir can only be used when the input is a directory");
    }

    run_single(&args, &input_path, cutoff)
}

/// Process a single input file: read it, plan its output, and run the pipeline.
fn run_single(args: &Args, input_path: &Path, cutoff: u8) -> Result<()> {
    let (src_original, input_format) = read_original_image(input_path)?;
    let output_plan = OutputPlan::new(
        input_path,
        args.output.as_deref(),
        None,
        args.alpha,
        args.allow_lossy_conversion,
        input_format,
        false,
    )?;

    let mut pipeline = Pipeline::new(args)?;
    pipeline.process_image(&src_original, &args.input, args, cutoff, &output_plan, None)
}

/// Process every image file in a directory, skipping inputs whose outputs are
/// already present unless `--overwrite` is set. The pipeline (and its lazily
/// loaded models) is built once and reused across files.
fn run_directory(args: &Args, input_path: &Path, cutoff: u8) -> Result<()> {
    if args.output.is_some() {
        bail!("--output cannot be used when the input is a directory; use --output-dir");
    }
    let output_dir = args
        .output_dir
        .as_deref()
        .context("input is a directory; --output-dir is required")?;
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("failed to create output dir: {output_dir}"))?;

    let inputs = collect_input_paths(input_path)?;
    let mut pipeline = Pipeline::new(args)?;

    for input_file in inputs {
        if !args.overwrite && output_dir_has_prefix(Path::new(output_dir), &input_file)? {
            if args.debug {
                elog!("skipping {} (already processed)", input_file.display());
            }
            continue;
        }

        let (src_original, input_format) = read_original_image(&input_file)?;
        let output_plan = OutputPlan::new(
            &input_file,
            None,
            Some(output_dir),
            args.alpha,
            args.allow_lossy_conversion,
            input_format,
            true,
        )?;
        let model_input = input_file.to_string_lossy();
        pipeline.process_image(
            &src_original,
            &model_input,
            args,
            cutoff,
            &output_plan,
            Some(&input_file),
        )?;
    }
    Ok(())
}
