//! A live status line for the stages that take long enough to look hung.
//!
//! Merging a set of full-platen scans spends half a minute in feature
//! detection and another fifteen seconds in the compositor without printing
//! anything, and reading or writing a 350 MB TIFF is not instant either. This
//! paints a single self-updating line on stderr saying what is happening and
//! how long it has been happening for.
//!
//! Three properties it is built around:
//!
//! * **Silent when not watched.** The line is only painted when stderr is a
//!   terminal, so redirecting to a log or piping to another program yields
//!   exactly the bytes it did before — no escape codes, no carriage returns,
//!   no repeated half-lines. Nothing is ever written to the `--log-file` sink.
//! * **It never eats real output.** Every ordinary line goes out through
//!   [`emit_stdout`]/[`emit_stderr`], which blank the status line, print, and
//!   repaint underneath — all while holding the lock, so the repaint thread
//!   cannot interleave with a half-written line.
//! * **A stage cannot be left on screen.** [`stage`] returns a guard, so an
//!   early return or a `?` takes the line down on the way out.
//!
//! Progress is reported through a module-level handle rather than threaded
//! through every signature, the same way the log sink is: the stages that need
//! it sit several calls down inside the merge, and passing a reporter to each
//! would be a lot of plumbing for a diagnostic.

use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

/// How often the line is repainted. Fast enough to read as alive, slow enough
/// to be invisible next to the work it is reporting on.
const TICK: Duration = Duration::from_millis(125);

/// Plain ASCII, so the line renders the same in a legacy console as in a
/// modern terminal.
const SPINNER: [char; 4] = ['-', '\\', '|', '/'];

struct Current {
    label: &'static str,
    detail: String,
    started: Instant,
}

#[derive(Default)]
struct State {
    current: Option<Current>,
    /// Width of the line last painted, so it can be blanked by overwriting
    /// with spaces rather than by an erase-to-end-of-line escape the terminal
    /// may not honour.
    painted: usize,
}

static STATE: OnceLock<Mutex<State>> = OnceLock::new();
static ENABLED: OnceLock<bool> = OnceLock::new();
static TICKING: AtomicBool = AtomicBool::new(false);
static FRAME: AtomicUsize = AtomicUsize::new(0);

/// Whether to paint at all. Defaults to "only when stderr is a terminal", so
/// piped and redirected runs are unchanged; `KROPP_PROGRESS=always` forces it
/// on for a terminal kropp cannot detect it is attached to (through `tee`, or
/// inside a CI log viewer that renders carriage returns), and `never` forces it
/// off.
fn enabled() -> bool {
    *ENABLED.get_or_init(|| match std::env::var("KROPP_PROGRESS").as_deref() {
        Ok("always") => true,
        Ok("never") => false,
        _ => std::io::stderr().is_terminal(),
    })
}

/// A poisoned lock only means some thread panicked mid-paint; the status line
/// is not worth propagating that to an otherwise healthy run.
fn state() -> MutexGuard<'static, State> {
    STATE
        .get_or_init(|| Mutex::new(State::default()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The status line's text. Pure, so the formatting is testable without a
/// terminal to paint on.
fn render(label: &str, detail: &str, elapsed: Duration, frame: usize) -> String {
    let spinner = SPINNER[frame % SPINNER.len()];
    let seconds = elapsed.as_secs_f64();
    // Sub-second stages flick past; showing tenths there stops the timer
    // looking stuck at zero.
    let clock = if seconds < 10.0 {
        format!("{seconds:.1}s")
    } else {
        format!("{seconds:.0}s")
    };
    if detail.is_empty() {
        format!("  {spinner} {label} … {clock}")
    } else {
        format!("  {spinner} {label} {detail} … {clock}")
    }
}

/// Repaint the line, or blank it when nothing is in progress. Caller holds the
/// lock, so this never races the ticker.
fn paint(state: &mut State) {
    let line = match &state.current {
        Some(current) => render(
            current.label,
            &current.detail,
            current.started.elapsed(),
            FRAME.load(Ordering::Relaxed),
        ),
        None => String::new(),
    };

    let width = line.chars().count();
    let mut err = std::io::stderr().lock();
    let _ = write!(err, "\r{line}");
    // Cover whatever a previous, longer line left behind.
    if state.painted > width {
        let overhang = state.painted - width;
        let _ = write!(err, "{:overhang$}\r{line}", "");
    }
    let _ = err.flush();
    state.painted = width;
}

/// Blank the line without disturbing which stage is current.
fn blank(state: &mut State) {
    if state.painted == 0 {
        return;
    }
    let width = state.painted;
    let mut err = std::io::stderr().lock();
    let _ = write!(err, "\r{:width$}\r", "");
    let _ = err.flush();
    state.painted = 0;
}

/// Start the repaint thread on first use. Detached: it holds no resources
/// worth reclaiming and the process exits out from under it.
fn start_ticker() {
    if TICKING.swap(true, Ordering::SeqCst) {
        return;
    }
    std::thread::spawn(|| {
        loop {
            std::thread::sleep(TICK);
            FRAME.fetch_add(1, Ordering::Relaxed);
            let mut state = state();
            if state.current.is_some() {
                paint(&mut state);
            }
        }
    });
}

/// Print `line` above the status line, then put the status back.
///
/// The blank, the write and the repaint all happen under one lock. Doing them
/// separately would let the repaint thread paint the spinner into the middle
/// of the line being printed.
fn emit(line: &str, stderr: bool) {
    if !enabled() {
        if stderr {
            eprintln!("{line}");
        } else {
            println!("{line}");
        }
        return;
    }
    let mut state = state();
    blank(&mut state);
    if stderr {
        eprintln!("{line}");
    } else {
        println!("{line}");
    }
    paint(&mut state);
}

pub fn emit_stdout(line: &str) {
    emit(line, false);
}

pub fn emit_stderr(line: &str) {
    emit(line, true);
}

/// A stage in progress. Dropping it takes the line down, so a stage cannot
/// outlive its scope or be left on screen by an early return.
#[must_use = "the status line shows for as long as this is held"]
pub struct Stage(());

impl Drop for Stage {
    fn drop(&mut self) {
        if !enabled() {
            return;
        }
        let mut state = state();
        state.current = None;
        paint(&mut state);
    }
}

/// Announce what is happening now. The returned guard ends it.
pub fn stage(label: &'static str) -> Stage {
    if enabled() {
        start_ticker();
        let mut state = state();
        state.current = Some(Current {
            label,
            detail: String::new(),
            started: Instant::now(),
        });
        paint(&mut state);
    }
    Stage(())
}

/// Add or replace the trailing detail of the current stage — a counter, a file
/// name, whatever says it is still moving.
pub fn detail(text: impl Into<String>) {
    if !enabled() {
        return;
    }
    let mut state = state();
    if let Some(current) = &mut state.current {
        current.detail = text.into();
        paint(&mut state);
    }
}

/// `println!`, but takes the status line down first so results scroll cleanly
/// above it.
#[macro_export]
macro_rules! say {
    ($($arg:tt)*) => {
        $crate::progress::emit_stdout(&::std::format!($($arg)*))
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stage_without_detail_still_names_itself_and_ticks() {
        let line = render("blending", "", Duration::from_millis(2500), 0);
        assert!(line.contains("blending"), "{line}");
        assert!(line.contains("2.5s"), "{line}");
        assert!(line.contains('-'), "spinner missing from {line}");
    }

    #[test]
    fn detail_appears_between_the_label_and_the_clock() {
        let line = render("detecting features", "3/4 passes", Duration::from_secs(42), 1);
        let label = line.find("detecting features").expect("label");
        let detail = line.find("3/4 passes").expect("detail");
        let clock = line.find("42s").expect("clock");
        assert!(label < detail && detail < clock, "out of order: {line}");
    }

    /// Under ten seconds the timer shows tenths, so a short stage does not sit
    /// at a motionless "0s"; past that whole seconds are enough and the line
    /// stops jittering in width.
    #[test]
    fn the_clock_switches_from_tenths_to_whole_seconds() {
        assert!(render("x", "", Duration::from_millis(400), 0).contains("0.4s"));
        assert!(render("x", "", Duration::from_millis(9900), 0).contains("9.9s"));
        assert!(render("x", "", Duration::from_millis(10_400), 0).contains("10s"));
    }

    /// The spinner has to actually advance, or the line reads as frozen — the
    /// exact thing it exists to disprove.
    #[test]
    fn the_spinner_advances_and_wraps() {
        let at = |frame| render("x", "", Duration::ZERO, frame);
        assert_ne!(at(0), at(1));
        assert_eq!(at(0), at(SPINNER.len()));
    }
}
