use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kmeanspp::{types::RGBAPixel, kmeans};
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
    let small_data = generate_random_pixels(1000);
    let medium_data = generate_random_pixels(10000);
    let large_data = generate_random_pixels(100000);

    let k_values = [2, 4, 8, 16];

    for k in k_values.iter() {
        c.bench_function(&format!("kmeans small k={}", k), |b| {
            b.iter(|| kmeans(black_box(&small_data), black_box(*k)))
        });

        c.bench_function(&format!("kmeans medium k={}", k), |b| {
            b.iter(|| kmeans(black_box(&medium_data), black_box(*k)))
        });

        c.bench_function(&format!("kmeans large k={}", k), |b| {
            b.iter(|| kmeans(black_box(&large_data), black_box(*k)))
        });
    }
}

criterion_group!(benches, benchmark_kmeans);
criterion_main!(benches);