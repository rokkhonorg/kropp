#[macro_use]
mod logging;
mod angle;
mod cli;
mod components;
mod debug_overlay;
mod extract;
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

fn main() {
    // When launched from Explorer (double-click or drag-and-drop) the process
    // gets its own console that vanishes on exit, so pause at the end to let the
    // user read the output. A run from an existing terminal must not pause.
    let standalone = launched_standalone();

    let code = match run() {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("error: {e:#}");
            1
        }
    };

    if standalone {
        wait_for_enter();
    }
    std::process::exit(code);
}

fn run() -> Result<()> {
    let args = Args::parse();

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
    let inputs = resolve_inputs(&args)?;

    // --output renames a single file's output; it can't address many inputs.
    if args.output.is_some() {
        if args.output_dir.is_some() {
            bail!("--output and --output-dir cannot be combined");
        }
        if inputs.len() != 1 || !inputs[0].is_file() {
            bail!(
                "--output is only valid with a single image file; \
                 use --output-dir, or omit both to write next to each input"
            );
        }
    }

    // Build the extractor (and lazy classifiers) once and reuse across every input.
    let mut pipeline = Pipeline::new(&args, cutoff)?;
    let multi = inputs.len() > 1;
    for input in &inputs {
        process_input(&mut pipeline, input, &args, multi)?;
    }
    Ok(())
}

/// Resolve the inputs to process: positional paths if any (the drag-and-drop
/// form), otherwise the single `--input`. Errors when neither is given or both.
fn resolve_inputs(args: &Args) -> Result<Vec<PathBuf>> {
    if !args.paths.is_empty() {
        if args.input.is_some() {
            bail!("provide inputs with --input or as positional paths, not both");
        }
        return Ok(args.paths.clone());
    }
    if let Some(input) = &args.input {
        return Ok(vec![PathBuf::from(input)]);
    }
    bail!(
        "no input given; pass an image or folder (e.g. `kropp photo.jpg`), \
         or drag files onto the executable"
    );
}

/// Dispatch one top-level input. A directory is processed in place (or into
/// `--output-dir`); a file is written next to itself (or per `--output` /
/// `--output-dir`). `label` adds the input path to the summary line when more
/// than one input is being processed.
fn process_input(pipeline: &mut Pipeline, input: &Path, args: &Args, label: bool) -> Result<()> {
    if input.is_dir() {
        // Default a folder's output to the folder itself, matching the
        // drag-a-folder-on-the-exe expectation.
        let output_dir = args
            .output_dir
            .as_deref()
            .map(PathBuf::from)
            .unwrap_or_else(|| input.to_path_buf());
        process_directory(pipeline, input, &output_dir, args)
    } else if input.is_file() {
        process_file(pipeline, input, args, label)
    } else {
        bail!(
            "input path is neither a file nor a directory: {}",
            input.display()
        );
    }
}

/// Process every image file in `dir`, writing crops into `output_dir` and
/// skipping inputs whose outputs are already present unless `--overwrite` is set.
fn process_directory(
    pipeline: &mut Pipeline,
    dir: &Path,
    output_dir: &Path,
    args: &Args,
) -> Result<()> {
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("failed to create output dir: {}", output_dir.display()))?;
    let output_dir_str = output_dir
        .to_str()
        .context("output directory path is not valid UTF-8")?;

    for input_file in collect_input_paths(dir)? {
        if !args.overwrite && output_dir_has_prefix(output_dir, &input_file)? {
            if args.debug {
                elog!("skipping {} (already processed)", input_file.display());
            }
            continue;
        }

        let (src_original, input_format) = read_original_image(&input_file)?;
        let output_plan = OutputPlan::new(
            &input_file,
            None,
            Some(output_dir_str),
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
            &output_plan,
            Some(&input_file),
        )?;
    }
    Ok(())
}

/// Process a single input file. Output goes to `--output-dir` if set, else the
/// explicit `--output`, else next to the input.
fn process_file(pipeline: &mut Pipeline, input: &Path, args: &Args, label: bool) -> Result<()> {
    let (src_original, input_format) = read_original_image(input)?;
    let output_plan = match &args.output_dir {
        Some(output_dir) => {
            std::fs::create_dir_all(output_dir)
                .with_context(|| format!("failed to create output dir: {output_dir}"))?;
            OutputPlan::new(
                input,
                None,
                Some(output_dir.as_str()),
                args.alpha,
                args.allow_lossy_conversion,
                input_format,
                true,
            )?
        }
        None => OutputPlan::new(
            input,
            args.output.as_deref(),
            None,
            args.alpha,
            args.allow_lossy_conversion,
            input_format,
            false,
        )?,
    };

    let model_input = input.to_string_lossy();
    let summary = label.then_some(input);
    pipeline.process_image(&src_original, &model_input, args, &output_plan, summary)
}

/// Whether this process owns a freshly allocated console — the signature of an
/// Explorer launch (double-click or drag-and-drop) rather than a run from an
/// existing terminal.
#[cfg(windows)]
fn launched_standalone() -> bool {
    // GetConsoleProcessList reports how many processes share our console. A
    // console spun up just for us has only this process attached; a terminal we
    // were launched from is also attached, giving a count of two or more.
    unsafe extern "system" {
        fn GetConsoleProcessList(process_list: *mut u32, count: u32) -> u32;
    }
    let mut pids = [0u32; 2];
    let count = unsafe { GetConsoleProcessList(pids.as_mut_ptr(), pids.len() as u32) };
    count == 1
}

#[cfg(not(windows))]
fn launched_standalone() -> bool {
    false
}

fn wait_for_enter() {
    use std::io::{Read, Write};
    let mut stdout = std::io::stdout();
    let _ = write!(stdout, "\nPress Enter to exit...");
    let _ = stdout.flush();
    let _ = std::io::stdin().read(&mut [0u8]);
}
