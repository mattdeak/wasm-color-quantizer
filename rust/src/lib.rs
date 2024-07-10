#[cfg(target_arch = "wasm")]
use wasm_bindgen::prelude::*;

use std::collections::HashSet;
use rand::Rng;
use rand::seq::SliceRandom;
use packed_simd::f32x4;

mod utils;
mod types;

use types::{RGBAPixel, Centroid};
use utils::{check_convergence, find_closest_centroid};


const MAX_ITERATIONS: usize = 300;
const TOLERANCE: f32 = 0.02;


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
        centroids.push(f32x4::new(first_centroid.r as f32, first_centroid.g as f32, first_centroid.b as f32, 0.0));
    } else {
        return centroids; 
    }

    // K-Means++
    while centroids.len() < k {
        let distances: Vec<f32> = data
            .iter()
            .map(|pixel| {
                centroids
                    .iter()
                    .map(|centroid| utils::euclidean_distance_simd(pixel.into(), *centroid))
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap()
            })
            .collect();

        let total_distance: f32 = distances.iter().sum();
        let threshold = rng.gen::<f32>() * total_distance;

        let mut cumulative_distance = 0.0;
        for (i, distance) in distances.iter().enumerate() {
            cumulative_distance += distance;
            if cumulative_distance >= threshold {
                let pixel = &data[i];
                centroids.push(pixel.into());
                break;
            }
        }
    }

    centroids
}

pub fn kmeans(data: &[RGBAPixel], k: usize) -> (Vec<Vec<usize>>, Vec<Centroid>) {
    let mut centroids = initialize_centroids(data, k);
    println!("len centroids: {}", centroids.len());
    let mut new_centroids: Vec<Centroid> = centroids.clone();
    println!("len new_centroids: {}", new_centroids.len());

    let mut clusters = vec![Vec::new(); k];
    let mut assignments = vec![0; data.len()];

    // Define the convergence criterion percentage (e.g., 2%)
    let mut iterations = 0;
    let mut converged = false;
    while iterations < MAX_ITERATIONS && !converged {
        // Assign points to clusters
        for (i, pixel) in data.iter().enumerate() {
            let closest_centroid = find_closest_centroid(pixel, &centroids);
            if assignments[i] != closest_centroid {
                assignments[i] = closest_centroid;
            }
        }

        clusters.iter_mut().for_each(|cluster| cluster.clear());
        assignments.iter().enumerate().for_each(|(i, &cluster)| {
            clusters[cluster].push(i);
        });

        // Update centroids and check for convergence
        clusters.iter().zip(new_centroids.iter_mut()).for_each(|(cluster, new_centroid)| {
            if cluster.is_empty() {
                return; // centroid can't move if there are no points
            }

            let mut sum_r = 0.0;
            let mut sum_g = 0.0;
            let mut sum_b = 0.0;
            let num_pixels = cluster.len() as f32;

            for &idx in cluster {
                let pixel = &data[idx];
                sum_r += pixel.r as f32;
                sum_g += pixel.g as f32;
                sum_b += pixel.b as f32;
            }

            *new_centroid = f32x4::new(sum_r / num_pixels, sum_g / num_pixels, sum_b / num_pixels, 0.0);
        });


        converged = check_convergence(&centroids, &new_centroids, TOLERANCE);
        // Swap the centroids and new_centroid. We'll update the new centroids again before
        // we check for convergence.
        std::mem::swap(&mut centroids, &mut new_centroids);
        iterations += 1;
    }
    
    (clusters, centroids)
}

#[cfg_attr(target_arch = "wasm", wasm_bindgen)]
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
            new_color.extract(0) as u8,
            new_color.extract(1) as u8,
            new_color.extract(2) as u8,
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