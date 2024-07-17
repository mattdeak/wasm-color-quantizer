#[cfg(feature = "gpu")]
use crate::kmeans::gpu::GpuAlgorithm;
use crate::kmeans::initializer::Initializer;
use std::fmt;

#[derive(Debug, Clone)]
pub enum KMeansAlgorithm {
    Lloyd,
    Hamerly,
    #[cfg(feature = "gpu")]
    Gpu(GpuAlgorithm),
}

#[cfg(feature = "gpu")]
impl KMeansAlgorithm {
    pub fn gpu(&self) -> Option<GpuAlgorithm> {
        match self {
            KMeansAlgorithm::Gpu(gpu) => Some(*gpu),
            _ => None,
        }
    }
}

impl fmt::Display for KMeansAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

#[derive(Debug, Clone)]
pub struct KMeansConfig {
    pub k: usize,
    pub max_iterations: usize,
    pub tolerance: f32,
    pub algorithm: KMeansAlgorithm,
    pub initializer: Initializer,
    pub seed: Option<u64>,
}

impl Default for KMeansConfig {
    fn default() -> Self {
        Self {
            k: 10,
            max_iterations: 100,
            tolerance: 0.02,
            algorithm: KMeansAlgorithm::Lloyd,
            initializer: Initializer::KMeansPlusPlus,
            seed: None,
        }
    }
}

impl fmt::Display for KMeansConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
