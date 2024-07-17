// #![cfg(target_arch = "wasm32")]

use colorcrunch::{
    kmeans::{KMeans, KMeansAlgorithm, KMeansConfig},
    types::{Vec3, Vec4, Vec4u},
};
use futures::executor::block_on;
use rand::Rng;
use statrs::{self, statistics::Statistics};
use std::time::{Duration, Instant};

fn generate_random_pixels(count: usize) -> Vec<Vec4> {
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| {
            [
                rng.gen::<f32>() * 255.0,
                rng.gen::<f32>() * 255.0,
                rng.gen::<f32>() * 255.0,
                0.0,
            ]
        })
        .collect()
}

fn generate_random_pixels_u32(count: usize) -> Vec<Vec4u> {
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| {
            [
                rng.gen_range(0..=255),
                rng.gen_range(0..=255),
                rng.gen_range(0..=255),
                rng.gen_range(0..=255),
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
    let algorithms = vec![
        colorcrunch::kmeans::KMeansAlgorithm::Hamerly,
        colorcrunch::kmeans::KMeansAlgorithm::Lloyd,
    ];

    for &size in &data_sizes {
        for &k in &k_values {
            for algorithm in &algorithms {
                let kmeans_cpu = KMeans::new(KMeansConfig {
                    algorithm: algorithm.clone(),
                    k: k as usize,
                    max_iterations: 1000,
                    tolerance: 0.02,
                    seed: Some(0),
                    ..Default::default()
                });

                let kmeans_gpu = block_on(KMeans::new_gpu(KMeansConfig {
                    algorithm: algorithm.clone(),
                    k: k as usize,
                    max_iterations: 1000,
                    tolerance: 0.02,
                    seed: Some(0),
                    ..Default::default()
                }));

                // Warmup with new data each time
                let warmup_start = Instant::now();
                while warmup_start.elapsed() < warmup_duration {
                    let warmup_data = generate_random_pixels(size);
                    let warmup_u32_data = generate_random_pixels_u32(size);
                    kmeans_cpu.run_vec4(&warmup_data).unwrap();
                    block_on(kmeans_gpu.run_async(&warmup_u32_data)).unwrap();
                }

                let mut times_cpu = Vec::with_capacity(iterations);
                let mut times_gpu = Vec::with_capacity(iterations);

                for _ in 0..iterations {
                    let data = generate_random_pixels(size);
                    let u32_data = generate_random_pixels_u32(size);

                    let start = Instant::now();
                    kmeans_cpu.run_vec4(&data).unwrap();
                    let duration = start.elapsed();
                    times_cpu.push(duration.as_secs_f64());

                    let start = Instant::now();
                    block_on(kmeans_gpu.run_async(&u32_data)).unwrap();
                    let duration = start.elapsed();
                    times_gpu.push(duration.as_secs_f64());
                }

                let mean_time_cpu: f64 = (&times_cpu).mean();
                let std_dev_cpu: f64 = (&times_cpu).std_dev();
                let mean_time_gpu: f64 = (&times_gpu).mean();
                let std_dev_gpu: f64 = (&times_gpu).std_dev();

                total_time += mean_time_cpu * iterations as f64;
                total_time += mean_time_gpu * iterations as f64;

                let ci_lower_cpu =
                    mean_time_cpu - (1.96 * std_dev_cpu / (iterations as f64).sqrt());
                let ci_upper_cpu =
                    mean_time_cpu + (1.96 * std_dev_cpu / (iterations as f64).sqrt());
                let ci_lower_gpu =
                    mean_time_gpu - (1.96 * std_dev_gpu / (iterations as f64).sqrt());
                let ci_upper_gpu =
                    mean_time_gpu + (1.96 * std_dev_gpu / (iterations as f64).sqrt());

                println!(
                    "Size: {}, K: {}, Algorithm: {:?}, CPU Mean Time: {:.6}s, CI: {:.6}s - {:.6}s",
                    size, k, algorithm, mean_time_cpu, ci_lower_cpu, ci_upper_cpu
                );
                println!(
                    "Size: {}, K: {}, Algorithm: {:?}, GPU Mean Time: {:.6}s, CI: {:.6}s - {:.6}s",
                    size, k, algorithm, mean_time_gpu, ci_lower_gpu, ci_upper_gpu
                );
            }
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
