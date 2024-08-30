#![cfg(feature = "gpu")]

mod buffers;
mod common;
mod lloyd_gpu1;

use crate::kmeans::config::KMeansConfig;
use crate::types::{Vec4, Vec4u};

use self::lloyd_gpu1::LloydAssignmentsOnly;

use super::types::KMeansError;

pub async fn run_lloyd_gpu(
    config: KMeansConfig,
    data: &[Vec4u],
) -> Result<(Vec<usize>, Vec<Vec4>), KMeansError> {
    let lloyd_gpu = LloydAssignmentsOnly::from_config(config).await;
    lloyd_gpu.run_async(data).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kmeans::config::KMeansConfig;
    use crate::types::Vec4u;
    use futures::executor::block_on;
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;

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

        // Create configuration
        let config = KMeansConfig {
            seed: Some(SEED),
            k: K,
            max_iterations: 100,
            tolerance: 1.0,
            initializer: crate::kmeans::Initializer::Random,
            ..Default::default()
        };

        // Run the GPU algorithm
        let (assignments, centroids) = block_on(run_lloyd_gpu(config, &data)).unwrap();

        // Basic sanity checks
        assert_eq!(assignments.len(), N);
        assert_eq!(centroids.len(), K);

        // Check if all assignments are within the valid range
        for assignment in &assignments {
            assert!(*assignment < K);
        }

        // Check if all centroid components are within the valid range
        for centroid in &centroids {
            for component in centroid.iter() {
                assert!(*component >= 0.0 && *component <= 255.0);
            }
        }
    }
}
