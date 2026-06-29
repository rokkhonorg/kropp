//! Diagnostic logging that mirrors stderr to an optional log file.
//!
//! Everything normally printed to stderr (warnings and `--debug` diagnostics)
//! goes through [`elog!`], which writes to stderr and, when a sink has been
//! installed via [`init_log_file`], appends the same line to that file.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result, anyhow};

static LOG_FILE: OnceLock<Mutex<BufWriter<File>>> = OnceLock::new();

/// Install the log-file sink. Subsequent [`elog!`] calls mirror their output
/// here in addition to stderr. Called once at startup; erroring if called twice.
pub fn init_log_file(path: &Path) -> Result<()> {
    let file = File::create(path)
        .with_context(|| format!("failed to create log file: {}", path.display()))?;
    LOG_FILE
        .set(Mutex::new(BufWriter::new(file)))
        .map_err(|_| anyhow!("log file already initialized"))
}

/// Append a preformatted line to the log file sink if one is installed. Flushes
/// each line so the log survives a crash. Used by [`elog!`]; not called directly.
pub fn write_line(line: &str) {
    if let Some(lock) = LOG_FILE.get()
        && let Ok(mut writer) = lock.lock()
    {
        let _ = writeln!(writer, "{line}");
        let _ = writer.flush();
    }
}

/// Like `eprintln!`, but also appends the line to the `--log-file` sink.
#[macro_export]
macro_rules! elog {
    ($($arg:tt)*) => {{
        let line = ::std::format!($($arg)*);
        ::std::eprintln!("{}", line);
        $crate::logging::write_line(&line);
    }};
}
