mod config;
pub mod distance;
pub mod hamerly;
pub mod initializer;
pub mod lloyd;
mod types;
mod utils;

pub mod gpu;

#[cfg(feature = "gpu")]
use self::gpu::run_lloyd_gpu;

pub use crate::kmeans::config::{KMeansAlgorithm, KMeansConfig};
pub use crate::kmeans::initializer::Initializer;
pub use crate::kmeans::utils::find_closest_centroid;
use crate::utils::num_distinct_colors;

use crate::types::{Vec3, Vec4, Vec4u, VectorExt};

use self::types::{KMeansError, KMeansResult};

const DEFAULT_INITIALIZER: Initializer = Initializer::KMeansPlusPlus;

// A wrapper for easier usage
#[derive(Debug, Clone)]
pub struct KMeans(KMeansConfig);

impl KMeans {
    pub fn from_config(config: KMeansConfig) -> Self {
        Self(config)
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
            initializer: DEFAULT_INITIALIZER,
            seed: None,
        })
    }
}

impl KMeans {
    pub fn run<T: VectorExt>(&self, data: &[T]) -> KMeansResult<T> {
        let unique_colors = num_distinct_colors(data);
        if unique_colors < self.0.k {
            return Err(KMeansError(format!(
                "Number of unique colors is less than k: {}",
                unique_colors
            )));
        }

        match self.0.algorithm {
            KMeansAlgorithm::Lloyd => Ok(lloyd::kmeans_lloyd(data, &self.0)),
            KMeansAlgorithm::Hamerly => Ok(hamerly::kmeans_hamerly(data, &self.0)),
            #[cfg(feature = "gpu")]
            _ => Err(KMeansError(format!(
                "Algorithm not supported on cpu: {}",
                self.0.algorithm
            ))),
        }
    }
    pub async fn run_async(&self, data: &[Vec4u]) -> KMeansResult<Vec4> {
        match &self.0.algorithm {
            KMeansAlgorithm::Lloyd | KMeansAlgorithm::Hamerly => {
                let data = data
                    .iter()
                    .map(|v| [v[0] as f32, v[1] as f32, v[2] as f32, v[3] as f32])
                    .collect::<Vec<Vec4>>();
                self.run(&data)
            }
            #[cfg(feature = "gpu")]
            _ => run_lloyd_gpu(self.0.clone(), data)
                .await
                .map_err(|e| KMeansError(e.to_string())),
        }
    }
}

impl KMeans {
    pub async fn new(config: KMeansConfig) -> Self {
        KMeans(config)
    }

    pub fn run_vec4(&self, data: &[Vec4]) -> KMeansResult<Vec4> {
        match &self.0.algorithm {
            KMeansAlgorithm::Lloyd => self.run(data),
            KMeansAlgorithm::Hamerly => self.run(data),
            #[cfg(feature = "gpu")]
            _ => Err(KMeansError(
                "GPU not supported for vec4 float data. Convert to u8 data first.".to_string(),
            )),
        }
    }

    pub fn run_vec3(&self, data: &[Vec3]) -> KMeansResult<Vec3> {
        match &self.0.algorithm {
            KMeansAlgorithm::Lloyd => self.run(data),
            KMeansAlgorithm::Hamerly => self.run(data),
            #[cfg(feature = "gpu")]
            _ => Err(KMeansError(
                "GPU not supported for 3 channel data. Convert to 4 channel data first."
                    .to_string(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kmeans::config::KMeansAlgorithm;
    use futures::executor::block_on;
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;

    trait TestExt<T: VectorExt> {
        fn assert_almost_eq(&self, other: &[T], tolerance: f64);
    }

    impl<T: VectorExt> TestExt<T> for Vec<T> {
        fn assert_almost_eq(&self, other: &[T], tolerance: f64) {
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

    fn run_kmeans_test(data: &[Vec3], k: usize, expected_non_empty_clusters: usize) {
        let algorithms = vec![KMeansAlgorithm::Lloyd, KMeansAlgorithm::Hamerly];

        for algorithm in algorithms {
            let config = KMeansConfig {
                k,
                max_iterations: 100,
                tolerance: 1e-4,
                algorithm,
                initializer: DEFAULT_INITIALIZER,
                seed: None,
            };

            let kmeans = KMeans::from_config(config.clone());
            let (clusters, centroids) = kmeans.run(data).unwrap();

            assert_eq!(
                clusters.len(),
                data.len(),
                "clusters.len() == data.len() with algorithm {}",
                &config.algorithm
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
            initializer: DEFAULT_INITIALIZER,
            seed: None,
        };
        let kmeans = KMeans::from_config(config);
        let result = kmeans.run(&data);
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
                    0.0,
                ]
            })
            .collect::<Vec<Vec4>>();

        let config_lloyd = KMeansConfig {
            k: 3,
            max_iterations: 500,
            tolerance: 1e-6,
            algorithm: KMeansAlgorithm::Lloyd,
            initializer: DEFAULT_INITIALIZER,
            seed: Some(seed),
        };

        let config_hamerly = KMeansConfig {
            k: 3,
            max_iterations: 500,
            tolerance: 1e-6,
            algorithm: KMeansAlgorithm::Hamerly,
            initializer: DEFAULT_INITIALIZER,
            seed: Some(seed),
        };

        #[cfg(feature = "gpu")]
        let config_gpu = KMeansConfig {
            k: 3,
            max_iterations: 500,
            tolerance: 1e-6,
            algorithm: KMeansAlgorithm::LloydGpu,
            initializer: DEFAULT_INITIALIZER,
            seed: Some(seed),
        };

        let kmeans_lloyd = KMeans::from_config(config_lloyd);
        let kmeans_hamerly = KMeans::from_config(config_hamerly);

        let (clusters1, centroids1) = kmeans_lloyd.run(&data).unwrap();
        let (clusters2, centroids2) = kmeans_hamerly.run(&data).unwrap();

        #[cfg(feature = "gpu")]
        {
            let kmeans_gpu = KMeans::from_config(config_gpu);
            let u32_data: Vec<Vec4u> = data
                .iter()
                .map(|p| [p[0] as u32, p[1] as u32, p[2] as u32, p[3] as u32])
                .collect();
            let (clusters3, centroids3) = block_on(kmeans_gpu.run_async(&u32_data)).unwrap();

            dbg!(&centroids1);
            dbg!(&centroids3);

            centroids1.assert_almost_eq(&centroids3, 1.0);
        }

        centroids1.assert_almost_eq(&centroids2, 1.0);
        assert_eq!(clusters1, clusters2);
    }
}
