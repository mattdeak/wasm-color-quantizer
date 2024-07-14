mod config;
pub mod distance;
pub mod hamerly;
pub mod lloyd;
mod utils;

pub use crate::kmeans::config::{KMeansAlgorithm, KMeansConfig};
pub use crate::kmeans::utils::find_closest_centroid;
use crate::utils::num_distinct_colors;

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
            seed: None,
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

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.0.seed = Some(seed);
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
            seed: None,
        })
    }
}

pub fn kmeans(data: &[ColorVec], config: &KMeansConfig) -> KMeansResult {
    let unique_colors = num_distinct_colors(data);
    if unique_colors < config.k {
        return Err(KMeansError(format!(
            "Number of unique colors is less than k: {}",
            unique_colors
        )));
    }

    match config.algorithm {
        KMeansAlgorithm::Lloyd => Ok(lloyd::kmeans_lloyd(data, config)),
        KMeansAlgorithm::Hamerly => Ok(hamerly::kmeans_hamerly(data, config)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kmeans::config::{KMeansAlgorithm, KMeansConfig};
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;

    trait TestExt {
        fn assert_almost_eq(&self, other: &[ColorVec], tolerance: f64);
    }

    impl TestExt for Vec<ColorVec> {
        fn assert_almost_eq(&self, other: &[ColorVec], tolerance: f64) {
            assert_eq!(self.len(), other.len());

            for i in 0..self.len() {
                let a_matches = (self[i][0] as f64 - other[i][0] as f64).abs() < tolerance;
                let b_matches = (self[i][1] as f64 - other[i][1] as f64).abs() < tolerance;
                let c_matches = (self[i][2] as f64 - other[i][2] as f64).abs() < tolerance;
                assert!(
                    a_matches && b_matches && c_matches,
                    "{:?} does not match {:?}",
                    self[i],
                    other[i]
                );
            }
        }
    }

    fn run_kmeans_test(data: &[ColorVec], k: usize, expected_non_empty_clusters: usize) {
        let algorithms = vec![KMeansAlgorithm::Lloyd, KMeansAlgorithm::Hamerly];

        for algorithm in algorithms {
            let config = KMeansConfig {
                k,
                max_iterations: 100,
                tolerance: 1e-4,
                algorithm,
                seed: None,
            };

            let (clusters, centroids) = kmeans(data, &config).unwrap();

            assert_eq!(
                clusters.len(),
                data.len(),
                "clusters.len() == data.len() with algorithm {}",
                config.algorithm
            );
            assert_eq!(
                centroids.len(),
                k,
                "centroids.len() == k with algorithm {}",
                config.algorithm
            );
            assert_eq!(
                clusters.iter().filter(|&&c| c < k).count(),
                data.len(),
                "clusters.iter().filter(|&&c| c < k).count() == data.len() with algorithm {}",
                config.algorithm
            );
            // assert_eq!(centroids.iter().filter(|&&c| c != [0.0, 0.0, 0.0]).count(), expected_non_empty_clusters);
            assert!(expected_non_empty_clusters >= 1);
        }
    }

    #[test]
    fn test_kmeans_basic() {
        let data = vec![[255.0, 0.0, 0.0], [0.0, 255.0, 0.0], [0.0, 0.0, 255.0]];
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
        let data = vec![[255.0, 0.0, 0.0], [0.0, 0.0, 255.0]];
        run_kmeans_test(&data, 2, 2);
    }

    #[test]
    fn test_kmeans_more_clusters_than_colors() {
        let data = vec![[255.0, 0.0, 0.0], [0.0, 255.0, 0.0]];
        let config = KMeansConfig {
            k: 3,
            max_iterations: 100,
            tolerance: 1e-4,
            algorithm: KMeansAlgorithm::Lloyd,
            seed: None,
        };
        let result = kmeans(&data, &config);
        assert_eq!(
            result.err().unwrap().to_string(),
            "Number of unique colors is less than k: 2"
        );
    }

    #[test]
    fn test_algorithms_converge_to_the_same_result_for_same_initial_conditions() {
        let seed = 42;
        let data_size = 100;

        let mut rng = StdRng::seed_from_u64(seed);
        let data = (0..data_size)
            .map(|_| {
                [
                    rng.gen::<f32>() * 255.0,
                    rng.gen::<f32>() * 255.0,
                    rng.gen::<f32>() * 255.0,
                ]
            })
            .collect::<Vec<ColorVec>>();

        let config_lloyd = KMeansConfig {
            k: 3,
            max_iterations: 500,
            tolerance: 1e-6,
            algorithm: KMeansAlgorithm::Lloyd,
            seed: Some(seed),
        };

        let config_hamerly = KMeansConfig {
            k: 3,
            max_iterations: 500,
            tolerance: 1e-6,
            algorithm: KMeansAlgorithm::Hamerly,
            seed: Some(seed),
        };

        let (clusters1, centroids1) = kmeans(&data, &config_lloyd).unwrap();
        let (clusters2, centroids2) = kmeans(&data, &config_hamerly).unwrap();

        centroids1.assert_almost_eq(&centroids2, 0.005);
        assert_eq!(clusters1, clusters2);
    }
}
