use crate::types::ColorVec;

// should probably test
#[inline]
pub fn euclidean_distance_squared(a: &ColorVec, b: &ColorVec) -> f32 {
    a.iter().zip(b.iter()).map(|(a, b)| (a - b) * (a - b)).sum()
}
