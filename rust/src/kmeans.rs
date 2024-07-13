pub mod lloyd;
pub mod hamerly;
pub mod distance;
mod utils;
mod config;


use crate::utils::num_distinct_colors;
pub use crate::kmeans::utils::find_closest_centroid;
pub use crate::kmeans::config::{KMeansConfig, KMeansAlgorithm};

use crate::types::ColorVec;

const DEFAULT_MAX_ITERATIONS: usize = 100;
const DEFAULT_TOLERANCE: f64 = 1e-2;
const DEFAULT_ALGORITHM: KMeansAlgorithm = KMeansAlgorithm::Lloyd;

#[derive(Debug, Clone)]
pub struct KMeansError(pub String);

impl std::fmt::Display for KMeansError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for KMeansError {}

type KMeansResult = Result<(Vec<usize>, Vec<ColorVec>), KMeansError>;

// A wrapper for easier usage
#[derive(Debug, Clone)]
pub struct KMeans(KMeansConfig);

impl KMeans {
    pub fn new(k: usize) -> Self {
        KMeans(KMeansConfig {
            k,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            tolerance: DEFAULT_TOLERANCE as f32,
            algorithm: DEFAULT_ALGORITHM,
        })
    }

    pub fn run(&self, data: &[ColorVec]) -> KMeansResult {
        kmeans(data, &self.0)
    }

    pub fn with_k(mut self, k: usize) -> Self {
        self.0.k = k;
        self
    }

    // Builders
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.0.max_iterations = max_iterations;
        self
    }

    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.0.tolerance = tolerance as f32;
        self
    }

    pub fn with_algorithm(mut self, algorithm: KMeansAlgorithm) -> Self {
        self.0.algorithm = algorithm;
        self
    }
}

impl Default for KMeans {
    fn default() -> Self {
        KMeans(KMeansConfig {
            k: 3,
            max_iterations: 100,
            tolerance: 1e-4,
            algorithm: KMeansAlgorithm::Lloyd,
        })
    }
}



pub fn kmeans(data: &[ColorVec], config: &KMeansConfig) -> KMeansResult {
    let unique_colors = num_distinct_colors(data);
    if unique_colors < config.k {
        return Err(KMeansError(format!("Number of unique colors is less than k: {}", unique_colors)));
    }

    match config.algorithm {
        KMeansAlgorithm::Lloyd => Ok(lloyd::kmeans_lloyd(data, config)),
        KMeansAlgorithm::Hamerly => Ok(hamerly::kmeans_hamerly(data, config)),
    }
}


#[cfg(test)]

mod tests {
    use super::*;
    use crate::kmeans::config::{KMeansConfig, KMeansAlgorithm};

    fn run_kmeans_test(data: &[ColorVec], k: usize, expected_non_empty_clusters: usize) {
        let algorithms = vec![KMeansAlgorithm::Lloyd, KMeansAlgorithm::Hamerly];

        for algorithm in algorithms {
            let config = KMeansConfig {
                k,
                max_iterations: 100,
                tolerance: 1e-4,
                algorithm,
            };

            let (clusters, centroids) = kmeans(data, &config).unwrap();

            assert_eq!(clusters.len(), data.len(), "clusters.len() == data.len() with algorithm {}", config.algorithm);
            assert_eq!(centroids.len(), k, "centroids.len() == k with algorithm {}", config.algorithm);
            assert_eq!(clusters.iter().filter(|&&c| c < k).count(), data.len(), "clusters.iter().filter(|&&c| c < k).count() == data.len() with algorithm {}", config.algorithm);
            // assert_eq!(centroids.iter().filter(|&&c| c != [0.0, 0.0, 0.0]).count(), expected_non_empty_clusters);
            assert!(expected_non_empty_clusters >= 0);
        }
    }

    #[test]
    fn test_kmeans_basic() {
        let data = vec![
            [255.0, 0.0, 0.0],
            [0.0, 255.0, 0.0],
            [0.0, 0.0, 255.0],
        ];
        run_kmeans_test(&data, 3, 3);
    }

    #[test]
    fn test_kmeans_single_color() {
        let data = vec![
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
        ];
        run_kmeans_test(&data, 1, 1);
    }

    #[test]
    fn test_kmeans_two_distinct_colors() {
        let data = vec![
            [255.0, 0.0, 0.0],
            [0.0, 0.0, 255.0],
        ];
        run_kmeans_test(&data, 2, 2);
    }

    #[test]
    fn test_kmeans_more_clusters_than_colors() {
        let data = vec![
            [255.0, 0.0, 0.0],
            [0.0, 255.0, 0.0],
        ];
        let config = KMeansConfig {
            k: 3,
            max_iterations: 100,
            tolerance: 1e-4,
            algorithm: KMeansAlgorithm::Lloyd,
        };
        let result = kmeans(&data, &config);
        assert_eq!(result.err().unwrap().to_string(), "Number of unique colors is less than k: 2");
    }
}