use kmeanspp::{types::RGBAPixel, kmeans};
use rand::Rng;
use std::time::Instant;

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
    let iterations = 10; // Number of iterations for each configuration
    let mut total_time = 0.0;

    for &size in &data_sizes {
        let data = generate_random_pixels(size);
        for &k in &k_values {
            let mut times = Vec::with_capacity(iterations);

            for _ in 0..iterations {
                let start = Instant::now();
                kmeans(&data, k);
                let duration = start.elapsed();
                times.push(duration.as_secs_f64());
            }

            let mean_time: f64 = times.iter().sum::<f64>() / iterations as f64;
            total_time += mean_time * iterations as f64;
            println!(
                "Size: {}, K: {}, Mean Time: {:.6}s",
                size,
                k,
                mean_time
            );
        }
    }
    total_time
}

pub fn main() {
    let total_time = benchmark();
    println!("Total benchmark time: {:.6}s", total_time);
}