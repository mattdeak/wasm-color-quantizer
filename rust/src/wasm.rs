#![cfg(target_arch = "wasm32")]

use wasm_bindgen::prelude::*;
use crate::quantize::Quantizer;

#[wasm_bindgen(js_name = "ColorQuantizer")]
pub struct ColorQuantizer(crate::quantize::Quantizer);

#[wasm_bindgen(typescript_custom_section)]
const TS_APPEND_CONTENT: &'static str = r#"
    export type Algorithm = "lloyd" | "hamerly";
    export type Channels = "RGB" | "RGBA";
"#;


type Algorithm = String;
type Channels = String;

#[wasm_bindgen]
impl ColorQuantizer {

    ///
    /// Constructor for the ColorQuantizer class.
    ///
    /// # Parameters
    ///
    /// @param {u32} max_colors - The maximum number of colors to quantize to.
    /// @param {u32} sample_rate - The sample rate for the quantization.
    /// @param {Channels} channels - The number of channels in the image.
    #[wasm_bindgen(constructor, skip_jsdoc)]
    pub fn new(max_colors: u32, sample_rate: u32, channels: Channels) -> Self {
        let channels = match channels.as_str() {
            "RGB" => 3,
            "RGBA" => 4,
            _ => panic!("Invalid channels: {}", channels),
        };
        let quantizer = Quantizer::new(max_colors.try_into().unwrap(), sample_rate.try_into().unwrap(), channels);
        Self(quantizer)
    }

    /// # Parameters
    ///
    /// @param {Algorithm} algorithm - The algorithm to use for quantization.
    /// @returns {ColorQuantizer} - The ColorQuantizer instance.
    #[wasm_bindgen(skip_jsdoc)]
    pub fn withAlgorithm(mut self, algorithm: Algorithm) -> Self {
        let converted_algorithm = match algorithm.as_str() {
            "lloyd" => crate::kmeans::KMeansAlgorithm::Lloyd,
            "hamerly" => crate::kmeans::KMeansAlgorithm::Hamerly,
            _ => panic!("Invalid algorithm: {}", algorithm),
        };
        self.0 = self.0.with_algorithm(converted_algorithm);
        self
    }

    pub fn withMaxIterations(mut self, max_iterations: u32) -> Self {
        self.0 = self.0.with_max_iterations(max_iterations.try_into().unwrap());
        self
    }

    pub fn withMaxColors(mut self, max_colors: u32) -> Self {
        self.0 = self.0.with_max_colors(max_colors.try_into().unwrap());
        self
    }

    /// Set the number of channels in the image.
    ///
    /// # Parameters
    ///
    /// @param {Channels} channels - The number of channels in the image.
    /// @returns {ColorQuantizer} - The ColorQuantizer instance.
    #[wasm_bindgen(skip_jsdoc)]
    pub fn withChannels(mut self, channels: Channels) -> Self {
        let converted_channels = match channels.as_str() {
            "RGB" => 3,
            "RGBA" => 4,
            _ => panic!("Invalid channels: {}", channels),
        };
        self.0 = self.0.with_channels(converted_channels);
        self
    }

    pub fn withTolerance(mut self, tolerance: f64) -> Self {
        self.0 = self.0.with_tolerance(tolerance);
        self
    }

    pub fn withSampleRate(mut self, sample_rate: u32) -> Self {
        self.0 = self.0.with_sample_rate(sample_rate.try_into().unwrap());
        self
    }

    pub fn quantize(&self, data: Vec<u8>) -> Vec<u8> {
        self.0.quantize(&data)
    }
}