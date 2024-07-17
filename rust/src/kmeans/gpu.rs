mod buffers;
mod common;
mod lloyd_gpu1;
mod lloyd_gpu2;
mod lloyd_gpu4;

use futures::executor::block_on;

use crate::kmeans::config::KMeansConfig;
use crate::kmeans::gpu::lloyd_gpu4::LloydAssignmentsAndCentroidInfo;
use crate::types::{Vec4, Vec4u};

use self::lloyd_gpu1::LloydAssignmentsOnly;
use self::lloyd_gpu2::LloydAssignmentsAndCentroids;

use super::types::KMeansError;
use super::KMeansAlgorithm;

#[derive(Debug, Clone, Copy)]
pub enum GpuAlgorithm {
    LloydAssignmentsOnly,
    LloydAssignmentsAndCentroids,
    LloydAssignmentsAndCentroidInfo,
}

impl TryFrom<KMeansAlgorithm> for GpuAlgorithm {
    type Error = &'static str;
    fn try_from(algorithm: KMeansAlgorithm) -> std::prelude::v1::Result<Self, Self::Error> {
        match algorithm {
            KMeansAlgorithm::Gpu(gpu) => Ok(gpu),
            KMeansAlgorithm::Lloyd => Ok(GpuAlgorithm::LloydAssignmentsAndCentroidInfo),
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
    LloydAssignmentsAndCentroidInfo(LloydAssignmentsAndCentroidInfo),
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
                GpuAlgorithm::LloydAssignmentsAndCentroidInfo => {
                    AlgorithmImpl::LloydAssignmentsAndCentroidInfo(
                        LloydAssignmentsAndCentroidInfo::from_config(config.clone()).await,
                    )
                }
                GpuAlgorithm::LloydAssignmentsAndCentroids => {
                    AlgorithmImpl::LloydAssignmentsAndCentroids(
                        LloydAssignmentsAndCentroids::from_config(config.clone()).await,
                    )
                }
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
            AlgorithmImpl::LloydAssignmentsAndCentroidInfo(lloyd) => lloyd.run_async(data).await,
        }
    }
}
