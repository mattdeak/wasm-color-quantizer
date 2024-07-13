#[cfg(target_arch = "wasm32")]
pub mod wasm;

#[cfg(feature = "python")]
pub mod python;

pub mod kmeans;
pub mod quantize;
pub mod types;
mod utils;
