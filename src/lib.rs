//! kropp as a library.
//!
//! The CLI in `main.rs` is one driver over this crate; the other is the Rokkhon
//! agent, which needs the same stages but split differently. The agent shows a
//! contributor every object found on a platen, lets them drop the ones that
//! aren't wanted and group scans that are overlapping passes of one original,
//! and only then commits to the final crops. The CLI's `process_image` fuses
//! extract → crop → deskew → orient → write into one call, which cannot be
//! interrupted for that review.
//!
//! So the modules stay exactly as the CLI left them and [`api`] adds a façade
//! over them that separates the stages the way an interactive caller needs:
//! extract once and hold the result, render an object loosely for review,
//! render it finally when it has been kept. Nothing in `api` reimplements the
//! pipeline — it reuses the same functions the CLI calls, in a different order.
//!
//! The one thing a library caller must know is that **extraction is the whole
//! cost**. Running RMBG or SAM3 over a full-platen scan dwarfs every other
//! stage, and both review renderings come from the same [`extract::Extraction`],
//! so it is the caller's job to hold onto one rather than call `extract` twice
//! for the same scan.

#[macro_use]
pub mod logging;
#[macro_use]
pub mod progress;
pub mod angle;
pub mod api;
pub mod cli;
pub mod components;
pub mod debug_overlay;
pub mod extract;
pub mod merge;
pub mod model;
pub mod output;
pub mod pipeline;

pub use api::{Kropp, ObjectRender};
pub use cli::Args;
pub use extract::Extraction;
pub use pipeline::Pipeline;
