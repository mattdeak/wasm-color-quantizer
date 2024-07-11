#[cfg(not(target_arch = "wasm32"))]
use criterion::{black_box, criterion_group, criterion_main, Criterion};

use kmeanspp::{types::ColorVec, kmeans_3chan, utils};
use rand::Rng;

#[cfg(not(target_arch = "wasm32"))]
fn generate_random_pixels(count: usize) -> Vec<ColorVec> {
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| [
            rng.gen(),
            rng.gen(),
            rng.gen(),
        ])
        .collect()
}

#[cfg(not(target_arch = "wasm32"))]
fn benchmark_kmeans(c: &mut Criterion) {
    let k_values = [2, 4, 8, 16];
    let data_sizes = [1000, 10000, 100000];

    for &size in &data_sizes {
        let data = generate_random_pixels(size);
        
        for &k in &k_values {
            let benchmark_name = format!("kmeans_size_{}_k_{}", size, k);
            c.bench_function(&benchmark_name, |b| {
                b.iter(|| kmeans_3chan(black_box(&data), black_box(k)))
            });
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn benchmark_euclidean_distance(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let a: ColorVec = [rng.gen(), rng.gen(), rng.gen()];
    let b: ColorVec = [rng.gen(), rng.gen(), rng.gen()];

    c.bench_function("euclidean_distance", |bencher| {
        bencher.iter(|| utils::euclidean_distance(black_box(&a), black_box(&b)))
    });
}

#[cfg(not(target_arch = "wasm32"))]
fn benchmark_find_closest_centroid(c: &mut Criterion) {
    let mut rng = rand::thread_rng();
    let pixel = [rng.gen(), rng.gen(), rng.gen()];
    let centroids: Vec<ColorVec> = (0..100).map(|_| [rng.gen(), rng.gen(), rng.gen()]).collect();

    c.bench_function("find_closest_centroid", |bencher| {
        bencher.iter(|| utils::find_closest_centroid(black_box(&pixel), black_box(&centroids)))
    });
}

#[cfg(not(target_arch = "wasm32"))]
criterion_group!(benches, benchmark_kmeans, benchmark_euclidean_distance, benchmark_find_closest_centroid);

#[cfg(not(target_arch = "wasm32"))]
criterion_main!(benches);

#[cfg(target_arch = "wasm32")]
pub fn main() {
    println!("This benchmark is only supported on non-wasm32 targets.");
}
