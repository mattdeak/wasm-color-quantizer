use crate::kmeans::config::KMeansConfig;
use crate::kmeans::utils::{find_closest_centroid, has_converged, initialize_centroids};
use crate::types::ColorVec;

pub fn kmeans_lloyd(data: &[ColorVec], config: &KMeansConfig) -> (Vec<usize>, Vec<ColorVec>) {
    let mut centroids = initialize_centroids(data, config.k, config.seed);
    let mut new_centroids: Vec<ColorVec> = centroids.clone();

    let mut clusters = vec![Vec::new(); config.k];
    let mut assignments = vec![0; data.len()];

    // Define the convergence criterion percentage (e.g., 2%)
    let mut iterations = 0;
    let mut converged = false;
    while iterations < config.max_iterations && !converged {
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
        clusters
            .iter()
            .zip(new_centroids.iter_mut())
            .for_each(|(cluster, new_centroid)| {
                if cluster.is_empty() {
                    return; // centroid can't move if there are no points
                }

                let mut sum_r = 0.0;
                let mut sum_g = 0.0;
                let mut sum_b = 0.0;
                let num_pixels = cluster.len() as f32;

                for &idx in cluster {
                    let pixel = &data[idx];
                    sum_r += pixel[0];
                    sum_g += pixel[1];
                    sum_b += pixel[2];
                }

                *new_centroid = [sum_r / num_pixels, sum_g / num_pixels, sum_b / num_pixels];
            });
        converged = has_converged(&centroids, &new_centroids, config.tolerance);
        if converged {
            dbg!("Converged in {} iterations", iterations);
        }
        // Swap the centroids and new_centroid. We'll update the new centroids again before
        // we check for convergence.
        std::mem::swap(&mut centroids, &mut new_centroids);
        iterations += 1;
    }

    (assignments, centroids)
}
