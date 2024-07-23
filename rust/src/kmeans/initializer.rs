use crate::kmeans::distance::euclidean_distance_squared;
use crate::kmeans::distance::SquaredEuclideanDistance;
use crate::types::VectorExt;
use rand::prelude::*;
use rand::SeedableRng;

#[derive(Debug, Clone)]
pub enum Initializer {
    KMeansPlusPlus,
    Random,
}

impl Initializer {
    pub fn initialize_centroids<T: VectorExt>(
        &self,
        data: &[T],
        k: usize,
        seed: Option<u64>,
    ) -> Vec<T> {
        match self {
            Initializer::KMeansPlusPlus => kmeans_plus_plus(data, k, seed),
            Initializer::Random => initialize_random(data, k, seed),
        }
    }
}

fn get_seedable_rng(seed: Option<u64>) -> StdRng {
    if let Some(seed) = seed {
        rand::rngs::StdRng::seed_from_u64(seed)
    } else {
        rand::rngs::StdRng::from_entropy()
    }
}

// Ok we're using the K-Means++ initialization
// I think this is right? Seems to work
fn kmeans_plus_plus<T: VectorExt>(data: &[T], k: usize, seed: Option<u64>) -> Vec<T> {
    let mut centroids = Vec::with_capacity(k);

    // Seed the RNG if provided, otherwise use the current time
    let mut rng = get_seedable_rng(seed);

    // Choose the first centroid randomly
    if let Some(first_centroid) = data.choose(&mut rng) {
        centroids.push(*first_centroid);
    } else {
        return centroids;
    }

    // K-Means++
    while centroids.len() < k {
        let distances: Vec<SquaredEuclideanDistance> = data
            .iter()
            .map(|pixel| {
                centroids
                    .iter()
                    .map(|centroid| euclidean_distance_squared(pixel, centroid))
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap()
            })
            .collect();

        let total_distance: SquaredEuclideanDistance = distances.iter().sum();
        let threshold = rng.gen::<f32>() * total_distance.0;

        let mut cumulative_distance = 0.0;
        for (i, distance) in distances.iter().enumerate() {
            cumulative_distance += distance.0;
            if cumulative_distance >= threshold {
                let pixel = &data[i];
                centroids.push(*pixel);
                break;
            }
        }
    }
    centroids
}

pub fn initialize_random<T: Copy + PartialEq>(data: &[T], k: usize, seed: Option<u64>) -> Vec<T> {
    let mut rng = get_seedable_rng(seed);
    let mut centroids = Vec::with_capacity(k);
    let mut indices: Vec<usize> = (0..data.len()).collect();

    while centroids.len() < k && !indices.is_empty() {
        let idx = rng.gen_range(0..indices.len());
        let data_idx = indices.swap_remove(idx);
        let candidate = data[data_idx];
        
        if !centroids.contains(&candidate) {
            centroids.push(candidate);
        }
    }

    centroids
}