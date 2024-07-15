use crate::kmeans::distance::euclidean_distance_squared;
use crate::kmeans::distance::SquaredEuclideanDistance;
use crate::types::Vec3;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;

// Return the index of closest centroid and distance to that centroid
pub fn find_closest_centroid(pixel: &Vec3, centroids: &[Vec3]) -> usize {
    debug_assert!(!centroids.is_empty());
    let mut min_distance = euclidean_distance_squared(pixel, &centroids[0]);
    let mut min_index = 0;
    for (i, centroid) in centroids.iter().enumerate() {
        let distance = euclidean_distance_squared(pixel, centroid);
        if distance < min_distance {
            min_distance = distance;
            min_index = i;
        }
    }
    min_index
}

pub fn has_converged(initial_centroids: &[Vec3], final_centroids: &[Vec3], tolerance: f32) -> bool {
    let tolerance = SquaredEuclideanDistance(tolerance * tolerance);
    initial_centroids
        .iter()
        .zip(final_centroids.iter())
        .all(|(a, b)| euclidean_distance_squared(a, b) < tolerance)
}

// Ok we're using the K-Means++ initialization
// I think this is right? Seems to work
pub fn initialize_centroids(data: &[Vec3], k: usize, seed: Option<u64>) -> Vec<Vec3> {
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

#[cfg(test)]
mod tests {
    use statrs::assert_almost_eq;

    use super::*;

    #[test]
    fn test_find_closest_centroid() {
        let pixel = [100.0, 100.0, 100.0];
        let centroids = vec![
            [0.0, 0.0, 0.0],
            [100.0, 100.0, 100.0],
            [200.0, 200.0, 200.0],
        ];

        let closest_index = find_closest_centroid(&pixel, &centroids);
        assert_eq!(closest_index, 1);
    }
}
