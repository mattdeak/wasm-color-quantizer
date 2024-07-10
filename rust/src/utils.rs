use packed_simd::f32x4;
use crate::types::{RGBAPixel, Centroid};

// should probably test
#[inline]
pub fn euclidean_distance_simd(a: f32x4, b: f32x4) -> f32 {
    let diff = a - b;
    let squared_diff = diff * diff;
    let sum_squared_diff = squared_diff.sum();
    sum_squared_diff.sqrt()
}

pub fn find_closest_centroid(pixel: &RGBAPixel, centroids: &[Centroid]) -> usize {
    assert!(centroids.len() > 0);
    centroids.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            euclidean_distance_simd(pixel.into(), **a)
                .partial_cmp(&euclidean_distance_simd(pixel.into(), **b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(index, _)| index)
        .unwrap_or(0)
}

pub fn check_convergence(initial_centroids: &[Centroid], final_centroids: &[Centroid], tolerance: f32) -> bool {
    let min_initial_distance = calculate_min_centroid_distance(initial_centroids);
    let max_movement = calculate_max_centroid_movement(initial_centroids, final_centroids);
    max_movement < tolerance * min_initial_distance
}

pub fn calculate_max_centroid_movement(initial_centroids: &[Centroid], final_centroids: &[Centroid]) -> f32 {
    initial_centroids.iter().zip(final_centroids.iter()).map(|(a, b)| euclidean_distance_simd(*a, *b)).reduce(f32::max).unwrap_or(0.0)
}


// Helper function to calculate the minimum distance between centroids
pub fn calculate_min_centroid_distance(centroids: &[Centroid]) -> f32 {
    let mut min_distance = f32::MAX;
    for i in 0..centroids.len() {
        for j in (i + 1)..centroids.len() {
            let distance = euclidean_distance_simd(
                centroids[i],
                centroids[j]
            );
            min_distance = min_distance.min(distance);
        }
    }
    min_distance
}


#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_find_closest_centroid() {
        let pixel = RGBAPixel::new(100, 100, 100, 255);
        let centroids = vec![
            f32x4::new(0.0, 0.0, 0.0, 0.0),
            f32x4::new(100.0, 100.0, 100.0, 0.0),
            f32x4::new(200.0, 200.0, 200.0, 0.0),
        ];
        
        let closest_index = find_closest_centroid(&pixel, &centroids);
        assert_eq!(closest_index, 1);
    }

    #[test]
    fn test_calculate_max_centroid_movement() {
        let initial_centroids = vec![
            f32x4::new(0.0, 0.0, 0.0, 0.0),
            f32x4::new(100.0, 100.0, 100.0, 0.0),
        ];
        let final_centroids = vec![
            f32x4::new(10.0, 10.0, 10.0, 0.0),
            f32x4::new(90.0, 90.0, 90.0, 0.0),
        ];
        
        let max_movement = calculate_max_centroid_movement(&initial_centroids, &final_centroids);
        assert!((max_movement - 17.32051).abs() < 0.00001); // sqrt(300) ≈ 17.32051
    }

    #[test]
    fn test_calculate_min_centroid_distance() {
        let centroids = vec![
            f32x4::new(0.0, 0.0, 0.0, 0.0),
            f32x4::new(100.0, 100.0, 100.0, 0.0),
            f32x4::new(200.0, 200.0, 200.0, 0.0),
        ];
        
        let min_distance = calculate_min_centroid_distance(&centroids);
        assert!((min_distance - 173.2051).abs() < 0.0001); // sqrt(30000) ≈ 173.2051
    }

}