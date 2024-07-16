use crate::kmeans::distance::euclidean_distance_squared;
use crate::kmeans::distance::SquaredEuclideanDistance;
use crate::types::Vec3;
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
            Initializer::Random => random_initializer(data, k, seed),
        }
    }
}

// Ok we're using the K-Means++ initialization
// I think this is right? Seems to work
fn kmeans_plus_plus<T: VectorExt>(data: &[T], k: usize, seed: Option<u64>) -> Vec<T> {
    let mut centroids = Vec::with_capacity(k);

    // Seed the RNG if provided, otherwise use the current time
    let mut rng = {
        if let Some(seed) = seed {
            rand::rngs::StdRng::seed_from_u64(seed)
        } else {
            rand::rngs::StdRng::from_entropy()
        }
    };

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

fn random_initializer<T: VectorExt>(data: &[T], k: usize, seed: Option<u64>) -> Vec<T> {
    // Seed the RNG if provided, otherwise use the current time
    let mut rng = {
        if let Some(seed) = seed {
            rand::rngs::StdRng::seed_from_u64(seed)
        } else {
            rand::rngs::StdRng::from_entropy()
        }
    };

    let mut centroids = Vec::with_capacity(k);
    for centroid in data.choose_multiple(&mut rng, k) {
        centroids.push(*centroid);
    }

    centroids
}
