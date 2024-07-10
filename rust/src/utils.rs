use crate::types::ColorVec;

// should probably test
#[inline]
pub fn euclidean_distance(a: &ColorVec, b: &ColorVec) -> f32 {
    a.iter().zip(b.iter()).map(|(a, b)| (a - b) * (a - b)).sum()
}

pub fn find_closest_centroid(pixel: &ColorVec, centroids: &[ColorVec]) -> usize {
    debug_assert!(centroids.len() > 0);
    centroids.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            euclidean_distance(pixel, *a)
                .partial_cmp(&euclidean_distance(pixel, *b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(index, _)| index)
        .unwrap_or(0)
}

pub fn check_convergence(initial_centroids: &[ColorVec], final_centroids: &[ColorVec], tolerance: f32) -> bool {
    let min_initial_distance = calculate_min_centroid_distance(initial_centroids);
    let max_movement = calculate_max_centroid_movement(initial_centroids, final_centroids);
    // We're squaring the tolerance because we aren't square-rooting the distances
    max_movement < (tolerance * tolerance) * min_initial_distance
}

pub fn calculate_max_centroid_movement(initial_centroids: &[ColorVec], final_centroids: &[ColorVec]) -> f32 {
    initial_centroids.iter().zip(final_centroids.iter()).map(|(a, b)| euclidean_distance(&a, &b)).reduce(f32::max).unwrap_or(0.0)
}


// Helper function to calculate the minimum distance between centroids
pub fn calculate_min_centroid_distance(centroids: &[ColorVec]) -> f32 {
    centroids.iter()
    .enumerate()
    .flat_map(|(i, &centroid_a)| 
        centroids[i+1..].iter().map(move |&centroid_b| 
            euclidean_distance(&centroid_a, &centroid_b)
        )
    )
    .fold(f32::MAX, f32::min)
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

    #[test]
    fn test_calculate_max_centroid_movement() {
        let initial_centroids = vec![
            [0.0, 0.0, 0.0],
            [100.0, 100.0, 100.0],
        ];
        let final_centroids = vec![
            [10.0, 10.0, 10.0],
            [90.0, 90.0, 90.0],
        ];
        
        let max_movement = calculate_max_centroid_movement(&initial_centroids, &final_centroids);
        assert!((max_movement - 300.0).abs() < 0.00001); // sqrt(300) ≈ 17.32051
    }

    #[test]
    fn test_calculate_min_centroid_distance() {
        let centroids = vec![
            [0.0, 0.0, 0.0],
            [100.0, 100.0, 100.0],
            [200.0, 200.0, 200.0],
        ];
        
        let min_distance = calculate_min_centroid_distance(&centroids) as f64;
        assert_almost_eq!(min_distance, 30000.0, 0.0001); // sqrt(30000) ≈ 173.2051
    }

}