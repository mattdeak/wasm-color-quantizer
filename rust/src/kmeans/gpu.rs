#![cfg(feature = "gpu")]

pub mod cubecl;
pub mod wgpu;

use ::cubecl::wgpu::WgpuRuntime;
use futures::executor::block_on;

use crate::kmeans::config::KMeansConfig;
use crate::types::{Vec4, Vec4u};

use self::cubecl::CubeKMeans;
use self::wgpu::LloydAssignmentsAndCentroids;
use self::wgpu::LloydAssignmentsOnly;

use super::types::KMeansError;
use super::KMeansAlgorithm;

#[derive(Debug, Clone, Copy)]
pub enum GpuAlgorithm {
    LloydAssignmentsOnly,
    LloydAssignmentsAndCentroids,
    LloydAssignmentCubeCl,
}

impl TryFrom<KMeansAlgorithm> for GpuAlgorithm {
    type Error = &'static str;
    fn try_from(algorithm: KMeansAlgorithm) -> std::prelude::v1::Result<Self, Self::Error> {
        match algorithm {
            KMeansAlgorithm::Gpu(gpu) => Ok(gpu),
            KMeansAlgorithm::Lloyd => Ok(GpuAlgorithm::LloydAssignmentsOnly),
            _ => Err("Algorithm not supported on gpu"),
        }
    }
}

impl From<GpuAlgorithm> for KMeansAlgorithm {
    fn from(algorithm: GpuAlgorithm) -> Self {
        KMeansAlgorithm::Gpu(algorithm)
    }
}

#[derive(Debug)]
enum AlgorithmImpl {
    LloydAssignmentsOnly(LloydAssignmentsOnly),
    LloydAssignmentsAndCentroids(LloydAssignmentsAndCentroids),
    LloydAssignmentCubeClWgpu(CubeKMeans<WgpuRuntime>),
}

#[derive(Debug)]
pub struct KMeansGpu {
    config: KMeansConfig,
    algorithm: AlgorithmImpl,
}

impl KMeansGpu {
    pub async fn new(config: KMeansConfig) -> Self {
        if let Some(gpu_algorithm) = config.algorithm.gpu() {
            let algorithm = match gpu_algorithm {
                GpuAlgorithm::LloydAssignmentsOnly => AlgorithmImpl::LloydAssignmentsOnly(
                    LloydAssignmentsOnly::from_config(config.clone()).await,
                ),
                GpuAlgorithm::LloydAssignmentsAndCentroids => {
                    AlgorithmImpl::LloydAssignmentsAndCentroids(
                        LloydAssignmentsAndCentroids::from_config(config.clone()).await,
                    )
                }
                GpuAlgorithm::LloydAssignmentCubeCl => AlgorithmImpl::LloydAssignmentCubeClWgpu(
                    CubeKMeans::<WgpuRuntime>::from_device_and_config(
                        Default::default(),
                        config.clone(),
                    ),
                ),
            };
            Self {
                config: config.clone(),
                algorithm,
            }
        } else {
            // TODO: replace with error handling
            panic!("Gpu algorithm not found");
        }
    }

    pub fn run(&self, data: &[Vec4u]) -> Result<(Vec<usize>, Vec<Vec4>), KMeansError> {
        block_on(self.run_async(data))
    }

    pub async fn run_async(&self, data: &[Vec4u]) -> Result<(Vec<usize>, Vec<Vec4>), KMeansError> {
        match &self.algorithm {
            AlgorithmImpl::LloydAssignmentsOnly(lloyd) => lloyd.run_async(data).await,
            AlgorithmImpl::LloydAssignmentsAndCentroids(lloyd) => lloyd.run_async(data).await,
            AlgorithmImpl::LloydAssignmentCubeClWgpu(cube) => cube.run(data),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kmeans::config::KMeansConfig;
    use crate::types::Vec4u;
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;
    use statrs::assert_almost_eq;

    #[test]
    fn test_gpu_algorithms_convergence() {
        const K: usize = 5;
        const N: usize = 1000;
        const SEED: u64 = 42;

        // Generate random data
        let mut rng = StdRng::seed_from_u64(SEED);
        let data: Vec<Vec4u> = (0..N)
            .map(|_| {
                [
                    rng.gen_range(0..255),
                    rng.gen_range(0..255),
                    rng.gen_range(0..255),
                    rng.gen_range(0..255),
                ]
            })
            .collect();

        // Create configurations for each algorithm
        let config = KMeansConfig {
            seed: Some(SEED),
            k: K,
            max_iterations: 100,
            tolerance: 1.0,
            initializer: crate::kmeans::Initializer::Random,
            ..Default::default()
        };

        let algorithms = vec![
            GpuAlgorithm::LloydAssignmentsOnly,
            // GpuAlgorithm::LloydAssignmentsAndCentroids,
            GpuAlgorithm::LloydAssignmentCubeCl,
        ];

        let mut results = Vec::new();

        // Run each algorithm
        for alg in algorithms {
            dbg!("Running algorithm: {}", alg);
            let mut cfg = config.clone();
            cfg.algorithm = KMeansAlgorithm::Gpu(alg);
            let kmeans = block_on(KMeansGpu::new(cfg));
            let (assignments, centroids) = block_on(kmeans.run_async(&data)).unwrap();
            results.push((assignments, centroids));
        }

        // Compare results
        for i in 1..results.len() {
            let (assignments_a, centroids_a) = &results[0];
            let (assignments_b, centroids_b) = &results[i];

            // Check if assignments are identical
            // assert_eq!(assignments_a, assignments_b, "Assignments differ for algorithm {}", i);

            // Check if centroids are approximately equal
            for (ca, cb) in centroids_a.iter().zip(centroids_b.iter()) {
                assert_almost_eq!(ca[0] as f64, cb[0] as f64, 1.0);
                assert_almost_eq!(ca[1] as f64, cb[1] as f64, 1.0);
                assert_almost_eq!(ca[2] as f64, cb[2] as f64, 1.0);
                assert_almost_eq!(ca[3] as f64, cb[3] as f64, 1.0);
            }
        }
    }
}
