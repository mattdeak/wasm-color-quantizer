use std::fmt;

#[derive(Debug, Clone)]
pub enum KMeansAlgorithm {
    Lloyd,
    Hamerly,
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
}


impl Default for KMeansConfig {
    fn default() -> Self {
        Self {
            k: 10,
            max_iterations: 100,
            tolerance: 0.02,
            algorithm: KMeansAlgorithm::Lloyd,
        }
    }
}

impl fmt::Display for KMeansConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}