#![cfg(target_arch = "wasm32")]

// use crate::quantize::{ColorCruncher, ColorCruncherBuilder};
// use wasm_bindgen::prelude::*;

// #[wasm_bindgen(js_name = ColorCruncher)]
// pub struct WasmColorCruncher(crate::quantize::ColorCruncher);

// #[wasm_bindgen(typescript_custom_section)]
// const TS_APPEND_CONTENT: &'static str = r#"
// export type Algorithm = "lloyd" | "hamerly";
// export type Format = "RGB" | "RGBA";
// "#;

// type Algorithm = String;
// type Format = String;

// #[wasm_bindgen(js_class = ColorCruncher)]
// impl WasmColorCruncher {
//     ///
//     /// Constructor for the ColorQuantizer class.
//     ///
//     /// # Parameters
//     ///
//     /// @param {u32} max_colors - The maximum number of colors to quantize to.
//     /// @param {u32} sample_rate - The sample rate for the quantization.
//     /// @param {Format} format - The format of the image.
//     #[wasm_bindgen(constructor, skip_jsdoc)]
//     pub fn new(max_colors: u32, sample_rate: u32, format: Format) -> Self {
//         let channels = match format.as_str() {
//             "RGB" => 3,
//             "RGBA" => 4,
//             _ => panic!("Invalid format: {}", format),
//         };
//         let quantizer = ColorCruncherBuilder::new(
//             max_colors.try_into().unwrap(),
//             sample_rate.try_into().unwrap(),
//             channels,
//         );
//         Self(quantizer)
//     }

//     /// # Parameters
//     ///
//     /// @param {Algorithm} algorithm - The algorithm to use for quantization.
//     /// @returns {ColorQuantizer} - The ColorQuantizer instance.
//     #[wasm_bindgen(skip_jsdoc, js_name = setAlgorithm)]
//     pub fn set_algorithm(&mut self, algorithm: Algorithm) {
//         let converted_algorithm = match algorithm.as_str() {
//             "lloyd" => crate::kmeans::KMeansAlgorithm::Lloyd,
//             "hamerly" => crate::kmeans::KMeansAlgorithm::Hamerly,
//             _ => panic!("Invalid algorithm: {}", algorithm),
//         };
//         self.0 = self.0.clone().with_algorithm(converted_algorithm);
//     }

//     #[wasm_bindgen(js_name = setMaxIterations)]
//     pub fn set_max_iterations(&mut self, max_iterations: u32) {
//         self.0 = self
//             .0
//             .clone()
//             .with_max_iterations(max_iterations.try_into().unwrap());
//     }

//     #[wasm_bindgen(js_name = setMaxColors)]
//     pub fn set_max_colors(&mut self, max_colors: u32) {
//         self.0 = self
//             .0
//             .clone()
//             .with_max_colors(max_colors.try_into().unwrap());
//     }

//     /// Set the number of channels in the image.
//     ///
//     /// # Parameters
//     ///
//     /// @param {Format} format - The format of the image.
//     /// @returns {ColorQuantizer} - The ColorQuantizer instance.
//     #[wasm_bindgen(skip_jsdoc, js_name = setFormat)]
//     pub fn set_format(&mut self, format: Format) {
//         let converted_channels = match format.as_str() {
//             "RGB" => 3,
//             "RGBA" => 4,
//             _ => panic!("Invalid format: {}", format),
//         };
//         self.0 = self.0.clone().with_channels(converted_channels);
//     }

//     #[wasm_bindgen(js_name = setTolerance)]
//     pub fn set_tolerance(&mut self, tolerance: f64) {
//         self.0 = self.0.clone().with_tolerance(tolerance);
//     }

//     #[wasm_bindgen(js_name = setSampleRate)]
//     pub fn set_sample_rate(&mut self, sample_rate: u32) {
//         self.0 = self
//             .0
//             .clone()
//             .with_sample_rate(sample_rate.try_into().unwrap());
//     }

//     #[wasm_bindgen(js_name = quantizeImage)]
//     pub fn quantize_image(&self, data: &[u8]) -> Vec<u8> {
//         self.0.clone().quantize_image(data)
//     }

//     #[wasm_bindgen(js_name = createPalette)]
//     pub fn create_palette(&self, data: &[u8]) -> Vec<u8> {
//         self.0
//             .create_palette(data)
//             .iter()
//             .map(|color| color.to_vec())
//             .flatten()
//             .collect()
//     }
// }
