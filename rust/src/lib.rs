#![feature(portable_simd)]

#[cfg(feature = "python")]
pub mod python;

pub mod kmeans;
pub mod quantize;
pub mod types;
mod utils;
pub mod wasm;
