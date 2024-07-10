use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rust::{RGBAPixel, kmeans};
use rand::Rng;

fn generate_random_pixels(count: usize) -> Vec<RGBAPixel> {
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| RGBAPixel::new(
            rng.gen(),
            rng.gen(),
            rng.gen(),
            255
        ))
        .collect()
}

fn benchmark_kmeans(c: &mut Criterion) {
    let pixel_counts = [1000, 10000, 100000];
    let k_values = [2, 4, 8, 16];

    for &pixel_count in &pixel_counts {
        let pixels = generate_random_pixels(pixel_count);

        for &k in &k_values {
            c.bench_function(&format!("kmeans_{}px_k{}", pixel_count, k), |b| {
                b.iter(|| kmeans(black_box(&pixels), black_box(k)))
            });
        }
    }
}

criterion_group!(benches, benchmark_kmeans);
criterion_main!(benches);
