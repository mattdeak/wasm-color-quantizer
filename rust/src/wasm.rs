#![cfg(target_arch = "wasm32")]
#![cfg(feature = "wasm")]
#![cfg(feature = "gpu")]

const RGBA_CHANNELS: usize = 4;
use js_sys::Uint8Array;

use crate::kmeans::gpu::GpuAlgorithm;
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
export type Algorithm = "lloyd" | "hamerly" | "lloyd-all-gpu" | "lloyd-assignment-gpu";
export type Initializer = "kmeans++" | "random";
"#;

type Algorithm = String;
type Initializer = String;

#[wasm_bindgen(js_class = ColorCruncherBuilder)]
impl WasmColorCruncherBuilder {
    #[wasm_bindgen(js_name = new)]
    pub fn new() -> Self {
        Self(ColorCruncherBuilder::new().with_channels(RGBA_CHANNELS))
    }

    #[wasm_bindgen(js_name = withMaxColors)]
    pub fn with_max_colors(self, max_colors: u32) -> Self {
        Self(self.0.with_max_colors(max_colors as usize))
    }

    #[wasm_bindgen(js_name = setMaxColors)]
    pub fn set_max_colors(&mut self, max_colors: u32) {
        self.0.max_colors = Some(max_colors as usize);
    }

    #[wasm_bindgen(js_name = withSampleRate)]
    pub fn with_sample_rate(self, sample_rate: u32) -> Self {
        Self(self.0.with_sample_rate(sample_rate as usize))
    }

    #[wasm_bindgen(js_name = setSampleRate)]
    pub fn set_sample_rate(&mut self, sample_rate: u32) {
        self.0.sample_rate = Some(sample_rate as usize);
    }

    #[wasm_bindgen(js_name = withTolerance)]
    pub fn with_tolerance(self, tolerance: f32) -> Self {
        Self(self.0.with_tolerance(tolerance))
    }

    #[wasm_bindgen(js_name = setTolerance)]
    pub fn set_tolerance(&mut self, tolerance: f32) {
        self.0.tolerance = Some(tolerance);
    }

    #[wasm_bindgen(js_name = withMaxIterations)]
    pub fn with_max_iterations(self, max_iterations: u32) -> Self {
        Self(self.0.with_max_iterations(max_iterations as usize))
    }

    #[wasm_bindgen(js_name = setMaxIterations)]
    pub fn set_max_iterations(&mut self, max_iterations: u32) {
        self.0.max_iterations = Some(max_iterations as usize);
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

    #[wasm_bindgen(js_name = setInitializer)]
    pub fn set_initializer(&mut self, initializer: Initializer) {
        let init = match initializer.as_str() {
            "kmeans++" => crate::kmeans::Initializer::KMeansPlusPlus,
            "random" => crate::kmeans::Initializer::Random,
            _ => panic!("Invalid initializer: {}", initializer),
        };
        self.0.initializer = Some(init);
    }

    #[wasm_bindgen(js_name = withAlgorithm)]
    pub fn with_algorithm(self, algorithm: Algorithm) -> Self {
        let algo = match algorithm.as_str() {
            "lloyd" => crate::kmeans::KMeansAlgorithm::Lloyd,
            "hamerly" => crate::kmeans::KMeansAlgorithm::Hamerly,
            "lloyd-all-gpu" => GpuAlgorithm::LloydAssignmentsAndCentroids.into(),
            "lloyd-assignment-gpu" => GpuAlgorithm::LloydAssignmentsOnly.into(),
            _ => panic!("Invalid algorithm: {}", algorithm),
        };
        Self(self.0.with_algorithm(algo))
    }

    #[wasm_bindgen(js_name = setAlgorithm)]
    pub fn set_algorithm(&mut self, algorithm: Algorithm) {
        let algo = match algorithm.as_str() {
            "lloyd" => crate::kmeans::KMeansAlgorithm::Lloyd,
            "hamerly" => crate::kmeans::KMeansAlgorithm::Hamerly,
            "lloyd-assignment-gpu-cube" => GpuAlgorithm::LloydAssignmentCubeCl.into(),
            "lloyd-assignment-gpu" => GpuAlgorithm::LloydAssignmentsOnly.into(),
            _ => panic!("Invalid algorithm: {}", algorithm),
        };
        self.0.algorithm = Some(algo);
    }

    #[wasm_bindgen(js_name = withSeed)]
    pub fn with_seed(self, seed: u64) -> Self {
        Self(self.0.with_seed(seed))
    }

    #[wasm_bindgen(js_name = setSeed)]
    pub fn set_seed(&mut self, seed: u64) {
        self.0.seed = Some(seed);
    }

    #[wasm_bindgen(js_name = build)]
    pub async fn build(&self) -> WasmColorCruncher {
        WasmColorCruncher(self.0.build().await)
    }
}

#[wasm_bindgen(js_class = ColorCruncher)]
impl WasmColorCruncher {
    #[wasm_bindgen(constructor)]
    pub fn builder(max_colors: u32, sample_rate: u32) -> WasmColorCruncherBuilder {
        WasmColorCruncherBuilder::new()
            .with_max_colors(max_colors)
            .with_sample_rate(sample_rate)
    }

    #[wasm_bindgen(js_name = quantizeImage)]
    pub async fn quantize_image(&self, data: &[u8]) -> Result<Uint8Array, String> {
        let result = self.0.quantize_image(data).await;
        Ok(Uint8Array::from(result.as_slice()))
    }

    // TODO
    // #[wasm_bindgen(js_name = createPalette)]
    // pub async fn create_palette(&self, data: &[u8]) -> Result<Vec<Uint8Array>, String> {
    //     // let palette = self.0.create_palette(data).await;
    //     // Ok(palette.iter().map(|color| Uint8Array::from(color.to_vec())).collect())
    // }
}
