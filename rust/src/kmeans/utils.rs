use crate::kmeans::distance::euclidean_distance_squared;
use crate::kmeans::distance::SquaredEuclideanDistance;
use crate::types::VectorExt;

// Return the index of closest centroid and distance to that centroid
pub fn find_closest_centroid<T: VectorExt>(pixel: &T, centroids: &[T]) -> usize {
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

pub fn has_converged<T: VectorExt>(
    initial_centroids: &[T],
    final_centroids: &[T],
    tolerance: f32,
) -> bool {
    let tolerance = SquaredEuclideanDistance(tolerance * tolerance);
    initial_centroids
        .iter()
        .zip(final_centroids.iter())
        .all(|(a, b)| euclidean_distance_squared(a, b) < tolerance)
}

#[cfg(test)]
mod tests {

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
