#![cfg(target_arch = "wasm32")]
#![cfg(feature = "wasm")]
#![cfg(feature = "gpu")]

const RGBA_CHANNELS: usize = 4;
use js_sys::Uint8Array;

use crate::quantize::{ColorCruncher, ColorCruncherBuilder};
use console_error_panic_hook;
use console_log;
use log::Level;
use std;
use wasm_bindgen::prelude::*;

// On start, load the wasm stuff we need
#[wasm_bindgen(start)]
fn start() {
    // Load the wasm stuff we need
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init_with_level(Level::Warn).expect("Failed to initialize console log");
}

#[wasm_bindgen(js_name = ColorCruncher)]
pub struct WasmColorCruncher(ColorCruncher);

#[wasm_bindgen(js_name = ColorCruncherBuilder)]
pub struct WasmColorCruncherBuilder(ColorCruncherBuilder);

#[wasm_bindgen(typescript_custom_section)]
const TS_APPEND_CONTENT: &'static str = r#"
export type Algorithm = "lloyd" | "hamerly" | "lloyd-gpu"
export type Initializer = "kmeans++" | "random";
"#;

type Algorithm = String;
type Initializer = String;

#[wasm_bindgen(js_class = ColorCruncherBuilder)]
impl WasmColorCruncherBuilder {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self(ColorCruncherBuilder::new().with_channels(RGBA_CHANNELS))
    }

    #[wasm_bindgen(js_name = withMaxColors)]
    pub fn with_max_colors(self, max_colors: u32) -> Self {
        Self(self.0.with_max_colors(max_colors as usize))
    }

    #[wasm_bindgen(js_name = withSampleRate)]
    pub fn with_sample_rate(self, sample_rate: u32) -> Self {
        Self(self.0.with_sample_rate(sample_rate as usize))
    }

    #[wasm_bindgen(js_name = withTolerance)]
    pub fn with_tolerance(self, tolerance: f32) -> Self {
        Self(self.0.with_tolerance(tolerance))
    }

    #[wasm_bindgen(js_name = withMaxIterations)]
    pub fn with_max_iterations(self, max_iterations: u32) -> Self {
        Self(self.0.with_max_iterations(max_iterations as usize))
    }

    #[wasm_bindgen(js_name = withInitializer)]
    pub fn with_initializer(self, initializer: Initializer) -> Self {
        let init = match initializer.as_str() {
            "kmeans++" => crate::kmeans::Initializer::KMeansPlusPlus,
            "random" => crate::kmeans::Initializer::Random,
            _ => panic!("Invalid initializer: {}", initializer),
        };
        Self(self.0.with_initializer(init))
    }

    #[wasm_bindgen(js_name = withAlgorithm)]
    pub fn with_algorithm(self, algorithm: Algorithm) -> Self {
        let algo = match algorithm.as_str() {
            "lloyd" => crate::kmeans::KMeansAlgorithm::Lloyd,
            "hamerly" => crate::kmeans::KMeansAlgorithm::Hamerly,
            "lloyd-gpu" => crate::kmeans::KMeansAlgorithm::LloydGpu,
            _ => panic!("Invalid algorithm: {}", algorithm),
        };
        Self(self.0.with_algorithm(algo))
    }

    #[wasm_bindgen(js_name = withSeed)]
    pub fn with_seed(self, seed: u64) -> Self {
        Self(self.0.with_seed(seed))
    }

    #[wasm_bindgen(js_name = build)]
    pub async fn build(&self) -> WasmColorCruncher {
        WasmColorCruncher(self.0.build().await)
    }
}

#[wasm_bindgen(js_class = ColorCruncher)]
impl WasmColorCruncher {
    #[wasm_bindgen(js_name = quantizeImage)]
    pub async fn quantize_image(&self, data: &[u8]) -> Result<Uint8Array, String> {
        let result = self.0.quantize_image(data).await;
        Ok(Uint8Array::from(result.as_slice()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use js_sys::Uint8Array;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    async fn test_color_cruncher_builder() {
        let builder = WasmColorCruncherBuilder::new();
        let builder = builder
            .with_max_colors(16)
            .with_sample_rate(2)
            .with_tolerance(0.01)
            .with_max_iterations(100)
            .with_initializer("kmeans++".to_string())
            .with_algorithm("lloyd".to_string())
            .with_seed(42);

        let cruncher = builder.build().await;
        assert!(true); // If we got here, the test passed
    }

    #[wasm_bindgen_test]
    async fn test_quantize_image() {
        let builder = WasmColorCruncherBuilder::new()
            .with_max_colors(2)
            .with_sample_rate(1);
        let cruncher = builder.build().await;

        let input_data = vec![
            255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
        ];

        let result = cruncher.quantize_image(&input_data).await.unwrap();
        assert_eq!(result.length(), input_data.len() as u32);

        // Convert Uint8Array back to Vec<u8> for easier assertions
        let result_vec: Vec<u8> = result.to_vec();

        // Check that we have only two colors (plus alpha)
        let unique_colors: std::collections::HashSet<_> = result_vec
            .chunks_exact(4)
            .map(|chunk| (chunk[0], chunk[1], chunk[2]))
            .collect();
        assert!(unique_colors.len() <= 2);
    }
}
