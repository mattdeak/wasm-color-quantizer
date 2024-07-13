use crate::types::ColorVec;
use std::ops::Mul;
use std::ops::MulAssign;
use std::iter::Sum;
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Sub;
use std::ops::SubAssign;
use std::ops::Div;
use std::ops::DivAssign;

// should probably test
#[inline]
pub fn euclidean_distance_squared(a: &ColorVec, b: &ColorVec) -> SquaredEuclideanDistance {
    SquaredEuclideanDistance(a.iter().zip(b.iter()).map(|(a, b)| (a - b) * (a - b)).sum())
}


#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct EuclideanDistance(pub f32);
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct SquaredEuclideanDistance(pub f32);

macro_rules! impl_distance {
    ($name:ident) => {
        impl $name {
            pub fn min(&self, other: &Self) -> Self {
                Self(self.0.min(other.0))
            }

            pub fn max(&self, other: &Self) -> Self {
                Self(self.0.max(other.0))
            }

            pub fn max_f32(&self, other: f32) -> Self {
                Self(self.0.max(other))
            }
        }

        impl Sum for $name {
            fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold(Self(0.0), |acc, x| acc + x)
            }
        }

        impl<'a> Sum<&'a Self> for $name {
            fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
                iter.fold(Self(0.0), |acc, &x| acc + x)
            }
        }

        impl Add for $name {
            type Output = Self;
            fn add(self, other: Self) -> Self {
                Self(self.0 + other.0)
            }
        }

        impl AddAssign for $name {
            fn add_assign(&mut self, other: Self) {
                self.0 += other.0;
            }
        }

        impl Sub for $name {
            type Output = Self;
            fn sub(self, other: Self) -> Self {
                Self(self.0 - other.0)
            }
        }

        impl SubAssign for $name {
            fn sub_assign(&mut self, other: Self) {
                self.0 -= other.0;
            }
        }

        impl Div for $name {
            type Output = Self;
            fn div(self, other: Self) -> Self {
                Self(self.0 / other.0)
            }
        }

        impl DivAssign for $name {
            fn div_assign(&mut self, other: Self) {
                self.0 /= other.0;
            }
        }

        impl Mul for $name {
            type Output = Self;
            fn mul(self, other: Self) -> Self {
                Self(self.0 * other.0)
            }
        }

        impl MulAssign for $name {
            fn mul_assign(&mut self, other: Self) {
                self.0 *= other.0;
            }
        }

        impl From<f32> for $name {
            fn from(value: f32) -> Self {
                Self(value)
            }
        }
    };
}

impl_distance!(EuclideanDistance);
impl_distance!(SquaredEuclideanDistance);

impl SquaredEuclideanDistance {
    pub fn sqrt(&self) -> EuclideanDistance {
        EuclideanDistance(self.0.sqrt())
    }
}

impl From<EuclideanDistance> for SquaredEuclideanDistance {
    fn from(value: EuclideanDistance) -> Self {
        Self(value.0 * value.0)
    }
}

impl From<SquaredEuclideanDistance> for EuclideanDistance {
    fn from(value: SquaredEuclideanDistance) -> Self {
        Self(value.0.sqrt())
    }
}