#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "python")]
pub mod python;

pub mod types;
pub mod kmeans;
mod utils;

use types::ColorVec;


const MAX_ITERATIONS: usize = 300;
const TOLERANCE: f32 = 0.02;



/// A k-means optimized for 3 channel color images

#[cfg_attr(target_arch = "wasm32", wasm_bindgen, target_feature(enable = "simd128"))]
pub fn reduce_colorspace(
    pixels: &[u8],
    max_colors: usize,
    sample_rate: usize, // 1 = no sampling, 2 = sample every 2 pixels, 3 = sample every 3 pixels, etc
    channels: usize // 3 = RGB, 4 = RGBA
) -> Vec<u8> {
    let image_data: Vec<ColorVec> = pixels
        .chunks_exact(channels)
        .step_by(sample_rate)
        .map(|chunk| [chunk[0] as f32, chunk[1] as f32, chunk[2] as f32])
        .collect();

    if utils::num_distinct_colors(&image_data) <= max_colors {
        return pixels.to_vec();
    }

    let kmeans = kmeans::KMeans::new(max_colors).with_max_iterations(MAX_ITERATIONS).with_tolerance(TOLERANCE as f64);
    let (_, centroids) = kmeans.run(&image_data).unwrap();

    let mut new_image = Vec::with_capacity(pixels.len());
    for pixel in pixels.chunks_exact(channels) {
        let px_vec = [pixel[0] as f32, pixel[1] as f32, pixel[2] as f32];
        let closest_centroid = kmeans::find_closest_centroid(&px_vec, &centroids);
        let new_color = &centroids[closest_centroid];

        if channels == 3 {
            new_image.extend_from_slice(&[
                new_color[0] as u8,
                new_color[1] as u8,
                new_color[2] as u8,
            ]);
        } else {
            new_image.extend_from_slice(&[
                new_color[0] as u8,
                new_color[1] as u8,
                new_color[2] as u8,
                pixel[3]
        ]);
            }
    }

    new_image
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;


    #[wasm_bindgen_test]
    fn test_reduce_colorspace() {
        let data = vec![
            255, 0, 0, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255
        ];
        let max_colors = 2;
        let sample_rate = 1;
        let channels = 4;

        let result = reduce_colorspace(&data, max_colors, sample_rate, channels);
        assert_eq!(result.len(), data.len());
    }
}