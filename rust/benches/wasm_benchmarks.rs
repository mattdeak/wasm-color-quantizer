// #![cfg(target_arch = "wasm32")]

use colorcrunch::{kmeans::{KMeans, KMeansConfig}, types::Vec3};
use rand::Rng;
use statrs::{self, statistics::Statistics};
use std::time::{Duration, Instant};

fn generate_random_pixels(count: usize) -> Vec<Vec3> {
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| {
            [
                rng.gen::<f32>() * 255.0,
                rng.gen::<f32>() * 255.0,
                rng.gen::<f32>() * 255.0,
            ]
        })
        .collect()
}

#[no_mangle]
pub extern "C" fn benchmark() -> f64 {
    let k_values = [2, 4, 8, 16];
    let data_sizes = [1000, 10000, 100000];
    let iterations = 100;
    let warmup_duration = Duration::from_secs(3);
    let mut total_time = 0.0;
    let algorithm = colorcrunch::kmeans::KMeansAlgorithm::Hamerly;

    for &size in &data_sizes {
        for &k in &k_values {
            let kmeans = KMeans::new_cpu(KMeansConfig {
                algorithm: algorithm.clone(),
                k: k as usize,
                max_iterations: 1000,
                tolerance: 0.02,
                seed: Some(0),
                ..Default::default()
            });
            // Warmup with new data each time
            let warmup_start = Instant::now();
            while warmup_start.elapsed() < warmup_duration {
                let warmup_data = generate_random_pixels(size);
                kmeans.run_vec3(&warmup_data).unwrap();
            }

            let mut times = Vec::with_capacity(iterations);

            for _ in 0..iterations {
                let data = generate_random_pixels(size);
                let start = Instant::now();
                kmeans.run_vec3(&data).unwrap();
                let duration = start.elapsed();
                times.push(duration.as_secs_f64());
            }

            let mean_time: f64 = (&times).mean();
            let std_dev: f64 = (&times).std_dev();
            total_time += mean_time * iterations as f64;
            let ci_lower = mean_time - (1.96 * std_dev / (iterations as f64).sqrt());
            let ci_upper = mean_time + (1.96 * std_dev / (iterations as f64).sqrt());
            println!(
                "Size: {}, K: {}, Mean Time: {:.6}s, CI: {:.6}s - {:.6}s",
                size, k, mean_time, ci_lower, ci_upper
            );
        }
    }
    total_time
}

#[cfg(target_arch = "wasm32")]
pub fn main() {
    let total_time = benchmark();
    println!("Total benchmark time: {:.6}s", total_time);
}

#[cfg(not(target_arch = "wasm32"))]
pub fn main() {
    println!("This benchmark is only supported on wasm32 targets.");
}
