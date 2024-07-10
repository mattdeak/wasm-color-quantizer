use wasm_bindgen::prelude::*;
use std::collections::HashSet;
use rand::seq::SliceRandom;
use packed_simd::f32x4;
use rand::Rng;

const MAX_ITERATIONS: usize = 1000;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct RGBAPixel {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl RGBAPixel {
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        RGBAPixel { r, g, b, a }
    }
}

pub type Centroid = Vec<f64>;


// I honestly don't know if this is speeding it up
// should probably test
#[inline]
fn euclidean_distance_simd(a: &RGBAPixel, b: &Centroid) -> f32 {
    let a_simd = f32x4::new(a.r as f32, a.g as f32, a.b as f32, 0.0);
    let b_simd = f32x4::new(b[0] as f32, b[1] as f32, b[2] as f32, 0.0);
    let diff = a_simd - b_simd;
    let squared_diff = diff * diff;
    let sum_squared_diff = squared_diff.sum();
    sum_squared_diff.sqrt()
}

fn find_closest_centroid(pixel: &RGBAPixel, centroids: &[Centroid]) -> usize {
    centroids.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            euclidean_distance_simd(pixel, a)
                .partial_cmp(&euclidean_distance_simd(pixel, b))
                .unwrap()
        })
        .map(|(index, _)| index)
        .unwrap()
}

fn num_distinct_colors(data: &[RGBAPixel]) -> usize {
    let mut color_hashset = HashSet::new();
    for pixel in data {
        color_hashset.insert(*pixel);
    }
    color_hashset.len()
}

// Ok we're using the K-Means++ initialization
// I think this is right? Seems to work
pub fn initialize_centroids(data: &[RGBAPixel], k: usize) -> Vec<Centroid> {
    let mut centroids = Vec::with_capacity(k);
    let mut rng = rand::thread_rng();

    // Choose the first centroid randomly
    if let Some(first_centroid) = data.choose(&mut rng) {
        centroids.push(vec![first_centroid.r as f64, first_centroid.g as f64, first_centroid.b as f64]);
    } else {
        return centroids; 
    }

    // K-Means++
    while centroids.len() < k {
        let distances: Vec<f64> = data
            .iter()
            .map(|pixel| {
                centroids
                    .iter()
                    .map(|centroid| euclidean_distance_simd(pixel, centroid) as f64)
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap()
            })
            .collect();

        let total_distance: f64 = distances.iter().sum();
        let threshold = rng.gen::<f64>() * total_distance;

        let mut cumulative_distance = 0.0;
        for (i, distance) in distances.iter().enumerate() {
            cumulative_distance += distance;
            if cumulative_distance >= threshold {
                let pixel = &data[i];
                centroids.push(vec![pixel.r as f64, pixel.g as f64, pixel.b as f64]);
                break;
            }
        }
    }

    centroids
}

pub fn kmeans(data: &[RGBAPixel], k: usize) -> (Vec<Vec<usize>>, Vec<Centroid>) {
    let mut centroids = initialize_centroids(data, k);
    let mut clusters = vec![Vec::new(); k];
    let mut assignments = vec![0; data.len()];

    let mut converged = false;
    let mut iterations = 0;
    while !converged && iterations < MAX_ITERATIONS {
        converged = true;
        
        // Assign points to clusters
        for (i, pixel) in data.iter().enumerate() {
            let closest_centroid = find_closest_centroid(pixel, &centroids);
            if assignments[i] != closest_centroid {
                assignments[i] = closest_centroid;
                converged = false;
            }
        }
        clusters.iter_mut().for_each(|cluster| cluster.clear());
        assignments.iter().enumerate().for_each(|(i, &cluster)| {
            clusters[cluster].push(i);
        });

        // Update centroids
        for (i, cluster) in clusters.iter().enumerate() {
            if cluster.is_empty() {
                continue;
            }

            let mut sum_r = 0.0;
            let mut sum_g = 0.0;
            let mut sum_b = 0.0;
            let num_pixels = cluster.len() as f64;

            for &idx in cluster {
                let pixel = &data[idx];
                sum_r += pixel.r as f64;
                sum_g += pixel.g as f64;
                sum_b += pixel.b as f64;
            }

            centroids[i] = vec![sum_r / num_pixels, sum_g / num_pixels, sum_b / num_pixels];
        }

        iterations += 1;
    }
    
    (clusters, centroids)
}

#[wasm_bindgen]
pub fn reduce_colorspace(
    width: u32,
    height: u32,
    pixels: &[u8],
    max_colors: usize
) -> Vec<u8> {
    let image_data: Vec<RGBAPixel> = pixels
        .chunks_exact(4)
        .map(|chunk| RGBAPixel { r: chunk[0], g: chunk[1], b: chunk[2], a: chunk[3] })
        .collect();

    if num_distinct_colors(&image_data) <= max_colors {
        return pixels.to_vec();
    }

    let (_, centroids) = kmeans(&image_data, max_colors);

    let mut new_image = Vec::with_capacity(width as usize * height as usize * 4);
    for pixel in image_data.iter() {
        let closest_centroid = find_closest_centroid(pixel, &centroids);
        let new_color = &centroids[closest_centroid];
        new_image.extend_from_slice(&[
            new_color[0] as u8,
            new_color[1] as u8,
            new_color[2] as u8,
            pixel.a
        ]);
    }

    new_image
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_basic() {
        let data = vec![
            RGBAPixel::new(255, 0, 0, 255),
            RGBAPixel::new(0, 255, 0, 255),
            RGBAPixel::new(0, 0, 255, 255),
        ];
        let k = 3;
        let (clusters, centroids) = kmeans(&data, k);

        assert_eq!(clusters.len(), k);
        assert_eq!(centroids.len(), k);
        assert_eq!(clusters.iter().map(|c| c.len()).sum::<usize>(), 3);
    }

    #[test]
    fn test_kmeans_single_color() {
        let data = vec![
            RGBAPixel::new(100, 100, 100, 255),
            RGBAPixel::new(100, 100, 100, 255),
            RGBAPixel::new(100, 100, 100, 255),
        ];
        let k = 2;
        let (clusters, centroids) = kmeans(&data, k);

        assert_eq!(clusters.len(), k);
        assert_eq!(centroids.len(), k);
        assert_eq!(clusters.iter().filter(|c| !c.is_empty()).count(), 1);
    }

    #[test]
    fn test_kmeans_two_distinct_colors() {
        let data = vec![
            RGBAPixel::new(255, 0, 0, 255),
            RGBAPixel::new(0, 0, 255, 255),
        ];
        let k = 2;
        let (clusters, centroids) = kmeans(&data, k);

        assert_eq!(clusters.len(), k);
        assert_eq!(centroids.len(), k);
    }

    #[test]
    fn test_kmeans_more_clusters_than_colors() {
        let data = vec![
            RGBAPixel::new(255, 0, 0, 255),
            RGBAPixel::new(0, 255, 0, 255),
        ];
        let k = 3;
        let (clusters, centroids) = kmeans(&data, k);

        assert_eq!(clusters.len(), k);
        assert_eq!(centroids.len(), k);
        assert_eq!(clusters.iter().filter(|c| !c.is_empty()).count(), 2);
    }
}