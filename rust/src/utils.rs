use crate::types::{Vec4u, VectorExt};

use std::collections::HashSet;

pub fn num_distinct_colors<T: VectorExt>(data: &[T]) -> usize {
    let mut color_hashset = HashSet::new();
    for pixel in data {
        // hacky but its fine, it only occurs once at the beginning
        let hash_key = pixel[0] as usize * 2 + pixel[1] as usize * 3 + pixel[2] as usize * 5;
        color_hashset.insert(hash_key);
    }
    color_hashset.len()
}

pub fn num_distinct_colors_u32(data: &[Vec4u]) -> usize {
    let mut color_hashset = HashSet::new();
    for pixel in data {
        color_hashset.insert(pixel[0] * 2 + pixel[1] * 3 + pixel[2] * 5);
    }
    color_hashset.len()
}
