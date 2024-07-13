use std::collections::HashSet;
use crate::types::ColorVec;

pub fn num_distinct_colors(data: &[ColorVec]) -> usize {
    let mut color_hashset = HashSet::new();
    for pixel in data {
        // hacky but its fine, it only occurs once at the beginning
        let hash_key = pixel[0] as u8 * 2 + pixel[1] as u8 * 3 + pixel[2] as u8 * 5;
        color_hashset.insert(hash_key);
    }
    color_hashset.len()
}