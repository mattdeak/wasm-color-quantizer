use crate::kmeans::config::KMeansConfig;
use crate::kmeans::distance::{
    euclidean_distance_squared, EuclideanDistance, SquaredEuclideanDistance,
};
use crate::kmeans::types::{Assignments, CentroidCounts, CentroidSums, Centroids};
use crate::kmeans::utils::has_converged;
use crate::types::VectorExt;
use itertools::izip;

type UpperBounds = Vec<EuclideanDistance>;
type LowerBounds = Vec<EuclideanDistance>;

pub fn kmeans_hamerly<T: VectorExt>(
    data: &[T],
    config: &KMeansConfig,
) -> (Assignments, Centroids<T>) {
    let (
        mut centroids,
        mut centroid_sums,
        mut centroid_counts,
        mut upper_bounds,
        mut lower_bounds,
        mut clusters,
    ) = initialize_hamerly(data, config);

    let mut centroid_move_distances = vec![EuclideanDistance(0.0); config.k];
    let mut centroid_neighbor_distances = vec![EuclideanDistance(f32::MAX); config.k];
    let mut new_centroids = centroids.clone();

    let num_pixels = data.len();
    let k = config.k;

    // I need to check this but I think it should ensure
    // we don't get any bounds checks?
    // If the compiler is smart enough, that is.
    assert!(num_pixels >= k);

    for _ in 0..config.max_iterations {
        compute_neighbor_distances(&centroids, &mut centroid_neighbor_distances);

        for (pixel, assigned_cluster, upper_bound, lower_bound) in
            izip!(data, &mut clusters, &mut upper_bounds, &mut lower_bounds)
        {
            let m = lower_bound.max(centroid_neighbor_distances[*assigned_cluster] / (2.).into());
            if *upper_bound <= m {
                continue;
            }

            *upper_bound = euclidean_distance_squared(&centroids[*assigned_cluster], pixel).sqrt();

            if *upper_bound <= m {
                continue;
            }

            let (best_distance, second_best_distance, best_index) =
                find_best_and_second_best(&centroids, pixel);
            *upper_bound = best_distance;
            *lower_bound = second_best_distance;
            if best_index != *assigned_cluster {
                centroid_sums[*assigned_cluster] = centroid_sums[*assigned_cluster].sub(pixel);
                centroid_counts[*assigned_cluster] -= 1;
                centroid_sums[best_index] = centroid_sums[best_index].add(pixel);
                centroid_counts[best_index] += 1;

                *assigned_cluster = best_index;
            }
        }

        // Move centroids into new_centroids (we swap later)
        move_centroids(
            &mut centroids,
            &mut new_centroids,
            &mut centroid_sums,
            &mut centroid_counts,
            &mut centroid_move_distances,
        );

        // We can optimize this by keeping a running total, but I doubt it's a bottleneck so
        // TODO maybe look into it
        if has_converged(&centroids, &new_centroids, config.tolerance) {
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
    (clusters.to_vec(), centroids.to_vec())
}

fn initialize_hamerly<T: VectorExt>(
    data: &[T],
    config: &KMeansConfig,
) -> (
    Centroids<T>,
    CentroidSums<T>,
    CentroidCounts,
    UpperBounds,
    LowerBounds,
    Assignments,
) {
    // indicex of the cluster each pixel belongs to
    let centroids = config
        .initializer
        .initialize_centroids(data, config.k, config.seed);

    let num_pixels = data.len();
    let mut clusters = vec![0; num_pixels];
    let mut upper_bounds = vec![EuclideanDistance(0.0); num_pixels];
    let mut lower_bounds = vec![EuclideanDistance(0.0); num_pixels];

    let mut centroid_sums = vec![T::zero(); config.k];
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

#[inline]
fn find_best_and_second_best<T: VectorExt>(
    centroids: &[T],
    point: &T,
) -> (EuclideanDistance, EuclideanDistance, usize) {
    let mut best_distance = SquaredEuclideanDistance(f32::MAX);
    let mut second_best_distance = SquaredEuclideanDistance(f32::MAX);
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

    // We need to square root before returning as we're comparing squared distances.
    (
        best_distance.sqrt(),
        second_best_distance.sqrt(),
        best_index,
    )
}

#[inline]
fn update_bounds(
    upper_bounds: &mut [EuclideanDistance],
    lower_bounds: &mut [EuclideanDistance],
    distances: &[EuclideanDistance],
    clusters: &[usize],
) {
    let (r1, r1_distance, _, r2_distance) = {
        let mut r1_distance = EuclideanDistance(f32::MIN);
        let mut r2_distance = EuclideanDistance(f32::MIN);

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

fn compute_neighbor_distances<T: VectorExt>(centroids: &[T], distances: &mut [EuclideanDistance]) {
    for (i, centroid) in centroids.iter().enumerate() {
        distances[i] = EuclideanDistance(f32::MAX);
        for (j, other_centroid) in centroids.iter().enumerate() {
            if i == j {
                continue;
            }
            // We need to square root here because the bounds check assumes true distances.
            distances[i] =
                distances[i].min(euclidean_distance_squared(centroid, other_centroid).sqrt());
        }
    }
}

fn move_centroids<T: VectorExt>(
    centroids: &mut [T],
    new_centroids: &mut [T],
    centroid_sums: &mut [T],
    centroid_counts: &mut [usize],
    centroid_move_distances: &mut [EuclideanDistance],
) {
    for (j, (current_centroid, new_centroid)) in
        centroids.iter().zip(new_centroids.iter_mut()).enumerate()
    {
        *new_centroid = centroid_sums[j].div_scalar(centroid_counts[j] as f32);
        // We need to square root here because the bounds check assumes true distances.
        centroid_move_distances[j] =
            euclidean_distance_squared(current_centroid, new_centroid).sqrt();
    }
}
