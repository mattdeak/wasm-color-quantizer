pub type Vec3 = [f32; 3];
pub type Vec4 = [f32; 4];

pub trait GPUVector {}
pub trait VectorExt:
    Clone + Copy + std::ops::Index<usize, Output = f32> + std::ops::IndexMut<usize> + std::fmt::Debug
{
    fn add(&self, other: &Self) -> Self;
    fn sub(&self, other: &Self) -> Self;
    fn div_scalar(&self, scalar: f32) -> Self;
    fn zero() -> Self;
}

impl VectorExt for Vec3 {
    fn zero() -> Self {
        [0.0; 3]
    }

    fn add(&self, other: &Vec3) -> Self {
        let mut sum = [0.0; 3];
        for i in 0..3 {
            sum[i] = self[i] + other[i];
        }
        sum
    }

    fn div_scalar(&self, scalar: f32) -> Self {
        [self[0] / scalar, self[1] / scalar, self[2] / scalar]
    }

    fn sub(&self, other: &Vec3) -> Self {
        let mut sum = [0.0; 3];
        for i in 0..3 {
            sum[i] = self[i] - other[i];
        }
        sum
    }
}

impl VectorExt for Vec4 {
    fn add(&self, other: &Vec4) -> Self {
        let mut sum = [0.0; 4];
        for i in 0..4 {
            sum[i] = self[i] + other[i];
        }
        sum
    }

    fn sub(&self, other: &Vec4) -> Self {
        let mut sum = [0.0; 4];
        for i in 0..4 {
            sum[i] = self[i] - other[i];
        }
        sum
    }

    fn div_scalar(&self, scalar: f32) -> Self {
        [
            self[0] / scalar,
            self[1] / scalar,
            self[2] / scalar,
            self[3] / scalar,
        ]
    }

    fn zero() -> Self {
        [0.0; 4]
    }
}

impl GPUVector for Vec4 {}
