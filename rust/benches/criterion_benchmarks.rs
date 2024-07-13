#[cfg(not(target_arch = "wasm32"))]
use criterion::{black_box, criterion_group, criterion_main, Criterion};

use kmeanspp::{types::ColorVec, kmeans::{KMeans, KMeansAlgorithm}};
use rand::Rng;

#[cfg(not(target_arch = "wasm32"))]
fn generate_random_pixels(count: usize) -> Vec<ColorVec> {
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| [
            rng.gen::<f32>() * 255.0,
            rng.gen::<f32>() * 255.0,
            rng.gen::<f32>() * 255.0,
        ])
        .collect()
}

#[cfg(not(target_arch = "wasm32"))]
fn benchmark_kmeans_comparison(c: &mut Criterion) {
    let k_values = [2, 4, 8, 16];
    let data_sizes = [1000, 10000, 100000, 500000];

    for &size in &data_sizes {
        let data = generate_random_pixels(size);
        
        for &k in &k_values {
            let mut group = c.benchmark_group(format!("kmeans_size_{}_k_{}", size, k));

            group.bench_function("Hamerly", |b| {
                b.iter(|| black_box(KMeans::new(black_box(k)).with_algorithm(KMeansAlgorithm::Hamerly).run(black_box(&data))))
            });

            group.bench_function("Lloyd", |b| {
                b.iter(|| black_box(KMeans::new(black_box(k)).with_algorithm(KMeansAlgorithm::Lloyd).run(black_box(&data))))
            });

            group.finish();
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn benchmark_euclidean_distance(c: &mut Criterion) {
    use kmeanspp::kmeans;

    let mut rng = rand::thread_rng();
    let a: ColorVec = [rng.gen(), rng.gen(), rng.gen()];
    let b: ColorVec = [rng.gen(), rng.gen(), rng.gen()];

    c.bench_function("euclidean_distance", |bencher| {
        bencher.iter(|| kmeans::distance::euclidean_distance_squared(black_box(&a), black_box(&b)))
    });
}

#[cfg(not(target_arch = "wasm32"))]
fn benchmark_find_closest_centroid(c: &mut Criterion) {
    use kmeanspp::kmeans;
    let mut rng = rand::thread_rng();
    let pixel = [rng.gen(), rng.gen(), rng.gen()];
    let centroids: Vec<ColorVec> = (0..100).map(|_| [rng.gen(), rng.gen(), rng.gen()]).collect();

    c.bench_function("find_closest_centroid", |bencher| {
        bencher.iter(|| kmeans::find_closest_centroid(black_box(&pixel), black_box(&centroids)))
    });
}

#[cfg(not(target_arch = "wasm32"))]
criterion_group!(benches, benchmark_kmeans_comparison, benchmark_euclidean_distance, benchmark_find_closest_centroid);

#[cfg(not(target_arch = "wasm32"))]
criterion_main!(benches);

#[cfg(target_arch = "wasm32")]
pub fn main() {
    println!("This benchmark is only supported on non-wasm32 targets.");
}