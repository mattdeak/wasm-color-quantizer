use wasm_bindgen::prelude::*;
use std::collections::HashSet;
use rand::seq::SliceRandom;
use packed_simd::f32x4;
use rand::Rng;

const MAX_ITERATIONS: usize = 1000;

#[derive(Clone, Copy, Debug)]
pub struct RGBAPixel {
    pub data: f32x4,
}

impl RGBAPixel {
    pub fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        RGBAPixel { 
            data: f32x4::new(
                r as f32 / 255.0,
                g as f32 / 255.0,
                b as f32 / 255.0,
                a as f32 / 255.0
            )
        }
    }

    pub fn r(&self) -> f32 { self.data.extract(0) }
    pub fn g(&self) -> f32 { self.data.extract(1) }
    pub fn b(&self) -> f32 { self.data.extract(2) }
    pub fn a(&self) -> f32 { self.data.extract(3) }

    pub fn to_u8_array(&self) -> [u8; 4] {
        [
            (self.r() * 255.0) as u8,
            (self.g() * 255.0) as u8,
            (self.b() * 255.0) as u8,
            (self.a() * 255.0) as u8,
        ]
    }
}

impl PartialEq for RGBAPixel {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl Eq for RGBAPixel {}

impl std::hash::Hash for RGBAPixel {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (self.r() as usize * 2 + self.g() as usize * 3 + self.b() as usize * 5 + self.a() as usize * 7).hash(state);
    }
}

pub type Centroid = [f32; 3];

#[inline]
fn euclidean_distance_simd(a: &RGBAPixel, b: &Centroid) -> f32 {
    let b_simd = f32x4::new(b[0], b[1], b[2], 0.0);
    let diff = a.data - b_simd;
    let squared_diff = diff * diff;
    squared_diff.sum().sqrt()
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

pub fn initialize_centroids(data: &[RGBAPixel], k: usize) -> Vec<Centroid> {
    let mut centroids = Vec::with_capacity(k);
    let mut rng = rand::thread_rng();

    if let Some(first_centroid) = data.choose(&mut rng) {
        centroids.push([first_centroid.r(), first_centroid.g(), first_centroid.b()]);
    } else {
        return centroids; 
    }

    while centroids.len() < k {
        let distances: Vec<f32> = data
            .iter()
            .map(|pixel| {
                centroids
                    .iter()
                    .map(|centroid| euclidean_distance_simd(pixel, centroid))
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
                centroids.push([pixel.r(), pixel.g(), pixel.b()]);
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

        for (i, cluster) in clusters.iter().enumerate() {
            if cluster.is_empty() {
                continue;
            }
            let mut sum = f32x4::splat(0.0);
            for &idx in cluster {
                sum += data[idx].data;
            }
            let num_pixels = cluster.len() as f32;
            let new_centroid = sum / f32x4::splat(num_pixels);
            centroids[i] = [new_centroid.extract(0), new_centroid.extract(1), new_centroid.extract(2)];
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
        .map(|chunk| RGBAPixel::new(chunk[0], chunk[1], chunk[2], chunk[3]))
        .collect();

    if num_distinct_colors(&image_data) <= max_colors {
        return pixels.to_vec();
    }

    let (_, centroids) = kmeans(&image_data, max_colors);

    let mut new_image = Vec::with_capacity(width as usize * height as usize * 4);
    for pixel in image_data.iter() {
        let closest_centroid = find_closest_centroid(pixel, &centroids);
        let new_color = &centroids[closest_centroid];
        let rgba = RGBAPixel::new(
            (new_color[0] * 255.0) as u8,
            (new_color[1] * 255.0) as u8,
            (new_color[2] * 255.0) as u8,
            (pixel.a() * 255.0) as u8
        ).to_u8_array();
        new_image.extend_from_slice(&rgba);
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