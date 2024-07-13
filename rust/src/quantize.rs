use crate::kmeans::find_closest_centroid;
use crate::kmeans::KMeans;
use crate::kmeans::KMeansAlgorithm;
use crate::types::ColorVec;
use crate::utils::num_distinct_colors;

#[derive(Clone, Debug, Default)]
pub struct ColorCruncher {
    kmeans: KMeans,
    max_colors: usize,
    pub sample_rate: usize,
    pub channels: usize,
}

impl ColorCruncher {
    pub fn new(max_colors: usize, sample_rate: usize, channels: usize) -> Self {
        let kmeans = KMeans::new(max_colors);
        Self {
            kmeans,
            sample_rate,
            channels,
            max_colors,
        }
    }

    pub fn with_max_colors(mut self, max_colors: usize) -> Self {
        self.max_colors = max_colors;
        self.kmeans = self.kmeans.with_k(max_colors);
        self
    }

    pub fn with_algorithm(mut self, algorithm: KMeansAlgorithm) -> Self {
        self.kmeans = self.kmeans.with_algorithm(algorithm);
        self
    }

    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.kmeans = self.kmeans.with_max_iterations(max_iterations);
        self
    }

    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.kmeans = self.kmeans.with_tolerance(tolerance);
        self
    }

    pub fn with_sample_rate(mut self, sample_rate: usize) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    pub fn with_channels(mut self, channels: usize) -> Self {
        self.channels = channels;
        self
    }

    // Getters/setters because k means also needs to change
    pub fn max_colors(&self) -> usize {
        self.max_colors
    }

    pub fn quantize_image(&self, pixels: &[u8]) -> Vec<u8> {
        let image_data: Vec<ColorVec> = pixels
            .chunks_exact(self.channels)
            .step_by(self.sample_rate)
            .map(|chunk| [chunk[0] as f32, chunk[1] as f32, chunk[2] as f32])
            .collect();

        // If there's already less than or equal to the max number of colors, return the original pixels
        if num_distinct_colors(&image_data) <= self.max_colors {
            return pixels.to_vec();
        }

        let (_, centroids) = self.kmeans.run(&image_data).unwrap();

        let mut new_image = Vec::with_capacity(pixels.len());
        for pixel in pixels.chunks_exact(self.channels) {
            let px_vec = [pixel[0] as f32, pixel[1] as f32, pixel[2] as f32];
            let closest_centroid = find_closest_centroid(&px_vec, &centroids);
            let new_color = &centroids[closest_centroid];

            if self.channels == 3 {
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
                    pixel[3],
                ]);
            }
        }

        new_image
    }

    pub fn create_palette(&self, pixels: &[u8]) -> Vec<[u8; 3]> {
        let image_data: Vec<ColorVec> = pixels
            .chunks_exact(self.channels)
            .step_by(self.sample_rate)
            .map(|chunk| [chunk[0] as f32, chunk[1] as f32, chunk[2] as f32])
            .collect();

        // If there's already less than or equal to the max number of colors, return the original pixels
        if num_distinct_colors(&image_data) < self.max_colors {
            // todo
            todo!()
        }

        let (_, centroids) = self.kmeans.run(&image_data).unwrap();
        centroids
            .iter()
            .map(|color| [color[0] as u8, color[1] as u8, color[2] as u8])
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduce_colorspace() {
        let data = vec![
            255, 0, 0, 255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255,
        ];
        let max_colors = 2;
        let sample_rate = 1;
        let channels = 4;

        let quantizer = ColorCruncher::new(max_colors, sample_rate, channels);

        let result = quantizer.quantize_image(&data);
        assert_eq!(result.len(), data.len());
    }
}
