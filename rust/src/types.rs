pub type ColorVec = [f32; 3];

pub trait VectorExt<T> {
    fn add(&self, other: &T) -> Self;
    fn sub(&self, other: &T) -> Self;
    fn div_scalar(&self, scalar: f32) -> Self;
}

impl VectorExt<ColorVec> for ColorVec {
    fn add(&self, other: &ColorVec) -> Self {
        let mut sum = [0.0; 3];
        for i in 0..3 {
            sum[i] = self[i] + other[i];
        }
        sum
    }

    fn div_scalar(&self, scalar: f32) -> Self {
        return [self[0] / scalar, self[1] / scalar, self[2] / scalar];
    }

    fn sub(&self, other: &ColorVec) -> Self {
        let mut sum = [0.0; 3];
        for i in 0..3 {
            sum[i] = self[i] - other[i];
        }
        sum
    }
}
