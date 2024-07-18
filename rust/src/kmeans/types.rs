// Some utility type aliases for readability
pub type Centroids<T> = Vec<T>;
pub type CentroidSums<T> = Vec<T>;
pub type Assignments = Vec<usize>;
pub type CentroidCounts = Vec<usize>;

// Result
pub type KMeansResult<T> = Result<(Assignments, Centroids<T>), KMeansError>;

#[derive(Debug, Clone)]
pub struct KMeansError(pub String);

impl std::fmt::Display for KMeansError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for KMeansError {
    fn from(s: &str) -> Self {
        KMeansError(s.to_string())
    }
}

impl std::error::Error for KMeansError {}
