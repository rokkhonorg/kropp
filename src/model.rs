//! Model construction and source-image loading.

use std::path::Path;

use anyhow::{Context, Result};
use image::{DynamicImage, ImageFormat};
use usls::{Config, Device, models::RMBG};

/// Build the RMBG model, trying CUDA -> DirectML -> CPU and using the first
/// device that initializes. usls registers a single execution provider and
/// errors when it is unavailable, so we drive the fallback ourselves.
pub fn build_rmbg() -> Result<RMBG> {
    let mut last_err = None;
    for (device, name) in [
        (Device::Cuda(0), "CUDA"),
        (Device::DirectMl(0), "DirectML"),
        (Device::Cpu(0), "CPU"),
    ] {
        let config = match Config::rmbg2_0().with_device_all(device).commit() {
            Ok(c) => c,
            Err(e) => {
                last_err = Some(e);
                continue;
            }
        };
        match RMBG::new(config) {
            Ok(model) => {
                elog!("RMBG running on {name}");
                return Ok(model);
            }
            Err(e) => {
                elog!("RMBG unavailable on {name}: {e:#}");
                last_err = Some(e);
            }
        }
    }
    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("no execution provider available")))
        .context("could not initialize RMBG on any device")
}

pub fn read_original_image(path: &Path) -> Result<(DynamicImage, ImageFormat)> {
    let reader = image::ImageReader::open(path)
        .with_context(|| format!("failed to open image: {}", path.display()))?
        .with_guessed_format()
        .with_context(|| format!("failed to guess image format: {}", path.display()))?;
    let format = reader
        .format()
        .or_else(|| ImageFormat::from_path(path).ok())
        .context("could not determine input image format")?;
    let image = reader
        .decode()
        .with_context(|| format!("failed to decode image: {}", path.display()))?;
    Ok((image, format))
}
