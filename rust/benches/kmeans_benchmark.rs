use kmeanspp::{types::RGBAPixel, kmeans};
use rand::Rng;
use std::time::{Instant, Duration};
use statrs::{self, statistics::Statistics};

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

#[no_mangle]
pub extern "C" fn benchmark() -> f64 {
    let k_values = [2, 4, 8, 16];
    let data_sizes = [1000, 10000, 100000];
    let iterations = 50;
    let warmup_duration = Duration::from_secs(3);
    let mut total_time = 0.0;

    for &size in &data_sizes {
        for &k in &k_values {
            // Warmup with new data each time
            let warmup_start = Instant::now();
            while warmup_start.elapsed() < warmup_duration {
                let warmup_data = generate_random_pixels(size);
                kmeans(&warmup_data, k);
            }

            let mut times = Vec::with_capacity(iterations);

            for _ in 0..iterations {
                let data = generate_random_pixels(size);
                let start = Instant::now();
                kmeans(&data, k);
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
                size,
                k,
                mean_time,
                ci_lower,
                ci_upper
            );
        }
    }
    total_time
}

pub fn main() {
    let total_time = benchmark();
    println!("Total benchmark time: {:.6}s", total_time);
}