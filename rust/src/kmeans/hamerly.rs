use crate::kmeans::config::KMeansConfig;
use crate::kmeans::distance::euclidean_distance_squared;
use crate::kmeans::utils::{has_converged, initialize_centroids};
use crate::types::{ColorVec, VectorExt};

// Some utility type aliases for readability
type Centroids = Vec<ColorVec>;
type CentroidSums = Vec<ColorVec>;
type UpperBounds = Vec<f32>;
type LowerBounds = Vec<f32>;
type Clusters = Vec<usize>;
type CentroidCounts = Vec<usize>;

pub fn kmeans_hamerly(data: &[ColorVec], config: &KMeansConfig) -> (Clusters, Centroids) {
    let (
        mut centroids,
        mut centroid_sums,
        mut centroid_counts,
        mut upper_bounds,
        mut lower_bounds,
        mut clusters,
    ) = initialize_hamerly(data, config);

    let mut centroid_move_distances = vec![0.0; config.k];
    let mut centroid_neighbor_distances = vec![f32::MAX; config.k];
    let mut new_centroids = centroids.clone();

    let num_pixels = data.len();
    let k = config.k;

    // I need to check this but I think it should ensure
    // we don't get any bounds checks?
    // If the compiler is smart enough, that is.
    assert!(num_pixels >= k);

    for iterations in 0..config.max_iterations {
        // Compute neighbor distances
        for (j, (centroid, neighbor_distance)) in centroids
            .iter()
            .zip(centroid_neighbor_distances.iter_mut())
            .enumerate()
        {
            for (k, other_centroid) in centroids.iter().enumerate() {
                if j != k {
                    let distance = euclidean_distance_squared(centroid, other_centroid);
                    if distance < *neighbor_distance {
                        *neighbor_distance = distance;
                    }
                }
            }
        }

        // Update bounds and clusters
        for i in 0..num_pixels {
            // Is this correct? hmm
            let m = upper_bounds[i].max(centroid_neighbor_distances[clusters[i]] / 2.);
            if upper_bounds[i] > m {
                // tighten upper bound here
                upper_bounds[i] = euclidean_distance_squared(&centroids[clusters[i]], &data[i]);
                if upper_bounds[i] > m {
                    let (best_distance, second_best_distance, best_index) =
                        find_best_and_second_best(&centroids, &data[i]);
                    if best_index != clusters[i] {
                        upper_bounds[clusters[i]] = best_distance;
                        lower_bounds[best_index] = second_best_distance;
                        clusters[i] = best_index;

                        // Update centroid sums and counts
                        centroid_sums[best_index] = centroid_sums[best_index].add(&data[i]);
                        centroid_sums[clusters[i]] = centroid_sums[clusters[i]].sub(&data[i]);
                        centroid_counts[best_index] += 1;
                        centroid_counts[clusters[i]] -= 1;
                    }
                }
            }
        }

        // Move centroids
        // Also, see TODO underneath
        for (j, (current_centroid, new_centroid)) in
            centroids.iter().zip(new_centroids.iter_mut()).enumerate()
        {
            let distance = euclidean_distance_squared(current_centroid, new_centroid);
            *new_centroid = centroid_sums[j].div_scalar(centroid_counts[j] as f32);
            centroid_move_distances[j] = distance;
        }

        // We can optimize this by keeping a running total, but I doubt it's a bottleneck so
        // TODO maybe look into it
        if has_converged(&centroids, &new_centroids, config.tolerance) {
            dbg!("Converged in {} iterations", iterations);
            std::mem::swap(&mut centroids, &mut new_centroids);
            break;
        }
        std::mem::swap(&mut centroids, &mut new_centroids);

        update_bounds(
            &mut upper_bounds,
            &mut lower_bounds,
            &centroid_move_distances,
            &clusters,
        )
    }
    (clusters, centroids)
}

fn initialize_hamerly(
    data: &[ColorVec],
    config: &KMeansConfig,
) -> (
    Centroids,
    CentroidSums,
    CentroidCounts,
    UpperBounds,
    LowerBounds,
    Clusters,
) {
    // indicex of the cluster each pixel belongs to
    let centroids = initialize_centroids(data, config.k, config.seed);

    let num_pixels = data.len();
    let mut clusters = vec![0; num_pixels];
    let mut upper_bounds = vec![0.0; num_pixels];
    let mut lower_bounds = vec![0.0; num_pixels];

    let mut centroid_sums = vec![[0., 0., 0.]; config.k];
    let mut centroid_counts = vec![0; config.k];

    assert!(data.len() >= config.k);
    assert!(centroid_sums.len() == config.k);

    for i in 0..num_pixels {
        let (best_distance, second_best_distance, best_index) =
            find_best_and_second_best(&centroids, &data[i]);

        upper_bounds[i] = best_distance;
        lower_bounds[i] = second_best_distance;
        clusters[i] = best_index;
        centroid_sums[best_index] = centroid_sums[best_index].add(&data[i]);
        centroid_counts[best_index] += 1;
    }
    (
        centroids,
        centroid_sums,
        centroid_counts,
        upper_bounds,
        lower_bounds,
        clusters,
    )
}

fn find_best_and_second_best(centroids: &[ColorVec], point: &ColorVec) -> (f32, f32, usize) {
    let mut best_distance = f32::MAX;
    let mut second_best_distance = f32::MAX;
    let mut best_index = 0;

    for (j, centroid) in centroids.iter().enumerate() {
        let distance = euclidean_distance_squared(centroid, point);
        if distance < best_distance {
            second_best_distance = best_distance;
            best_distance = distance;
            best_index = j;
        } else if distance < second_best_distance {
            second_best_distance = distance;
        }
    }

    (best_distance, second_best_distance, best_index)
}

fn update_bounds(
    upper_bounds: &mut UpperBounds,
    lower_bounds: &mut LowerBounds,
    distances: &[f32],
    clusters: &[usize],
) {
    let (r1, r1_distance, _, r2_distance) = {
        let mut r1_distance = f32::MIN;
        let mut r2_distance = f32::MIN;

        let mut r1_index = 0;
        let mut r2_index = 0;

        for (i, distance) in distances.iter().enumerate() {
            if *distance > r1_distance {
                r2_distance = r1_distance;
                r1_distance = *distance;
                r2_index = r1_index;
                r1_index = i;
            } else if *distance > r2_distance {
                r2_distance = *distance;
                r2_index = i;
            }
        }
        (r1_index, r1_distance, r2_index, r2_distance)
    };

    for i in 0..clusters.len() {
        upper_bounds[i] += distances[clusters[i]];
        if clusters[i] == r1 {
            lower_bounds[i] -= r2_distance;
        } else {
            lower_bounds[i] -= r1_distance;
        }
    }
}
