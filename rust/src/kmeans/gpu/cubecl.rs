use cubecl::{cuda::CudaRuntime, prelude::*, wgpu::WgpuRuntime};

use crate::{
    kmeans::{types::KMeansResult, utils::has_converged, KMeansConfig},
    types::{Vec4, Vec4u, VectorExt},
};
use bytemuck;

const CUBE_DIM_X: u32 = 256;

pub enum CubeKMeansImpl {
    Wgpu(CubeKMeans<WgpuRuntime>),
    Cuda(CubeKMeans<CudaRuntime>),
}

pub struct CubeKMeans<R: Runtime> {
    device: R::Device,
    client: ComputeClient<R::Server, R::Channel>,
    config: KMeansConfig,
}

impl<R: Runtime> std::fmt::Debug for CubeKMeans<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CubeKMeans")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl<R: Runtime> CubeKMeans<R> {
    pub fn from_device_and_config(device: R::Device, config: KMeansConfig) -> Self {
        let client = R::client(&device);
        Self {
            device,
            client,
            config,
        }
    }

    pub fn run(&self, pixels: &[Vec4u]) -> KMeansResult<Vec4> {
        let vec4_pixels = pixels
            .iter()
            .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32, p[3] as f32])
            .collect::<Vec<[f32; 4]>>();
        let mut centroids = self.config.initializer.initialize_centroids(
            &vec4_pixels,
            self.config.k,
            self.config.seed,
        );

        dbg!("Centroids", &centroids);
        let mut new_centroids = centroids.clone();
        let mut assignments = vec![0; pixels.len()];

        for _ in 0..self.config.max_iterations {
            self.run_iteration(
                &vec4_pixels,
                &centroids,
                &mut new_centroids,
                &mut assignments,
            );

            if has_converged(&centroids, &new_centroids, self.config.tolerance) {
                std::mem::swap(&mut centroids, &mut new_centroids);
                break;
            }

            std::mem::swap(&mut centroids, &mut new_centroids);
        }

        Ok((assignments, centroids))
    }

    pub fn run_iteration(
        &self,
        pixels: &[Vec4],
        centroids: &[Vec4],
        new_centroids: &mut [Vec4],
        assignments: &mut [usize],
    ) {
        let new_assignments = self.launch_assignment_update_kernel(pixels, centroids);
        assignments.copy_from_slice(&new_assignments);
        dbg!(&assignments);

        // update centroids
        new_centroids.fill([0.0; 4]);
        let mut centroid_counts = vec![0; self.config.k];
        for i in 0..pixels.len() {
            let point = pixels[i];
            let assignment = assignments[i];
            new_centroids[assignment].add(&point);
            centroid_counts[assignment] += 1;
        }

        for (centroid, count) in new_centroids.iter_mut().zip(centroid_counts.iter()) {
            *centroid = centroid.div_scalar(*count as f32);
        }
    }

    // GPU kernel launcher
    pub fn launch_assignment_update_kernel(
        &self,
        pixels: &[Vec4],
        centroids: &[Vec4],
    ) -> Vec<usize> {
        let centroid_handle = self.client.create(bytemuck::cast_slice(centroids));
        let pixel_handle = self.client.create(bytemuck::cast_slice(pixels));
        let assignment_handle = self.client.empty(pixels.len() * std::mem::size_of::<u32>());

        // Todo: We're remaking data on the gpu that can be left there
        // this is stupid
        let num_blocks = (pixels.len() as u32 + CUBE_DIM_X - 1) / CUBE_DIM_X;
        update_assignments::launch::<R>(
            &self.client,
            CubeCount::Static(num_blocks, 1, 1),
            CubeDim::new(CUBE_DIM_X, 1, 1),
            ArrayArg::new(&assignment_handle, pixels.len()),
            ArrayArg::vectorized(3, &centroid_handle, centroids.len()),
            ArrayArg::vectorized(3, &pixel_handle, pixels.len()),
            self.config.k as u32,
        );

        let assignments = self.client.read(assignment_handle.binding());
        let assignments = u32::from_bytes(&assignments)
            .iter()
            .map(|a| *a as usize)
            .collect();
        assignments
    }
}


// GPU kernel function
#[cube(launch)]
pub fn update_assignments(
    assignments: &mut Array<UInt>,
    centroids: &Array<F32>,
    points: &Array<F32>,
    k: Comptime<u32>,
) {
    if ABSOLUTE_POS < assignments.len() {
        let point_idx = ABSOLUTE_POS;
        let point = points[point_idx];

        let mut min_distance = squared_distance_to_centroid(point, centroids[0]);
        for i in range(1, k, Comptime::new(true)) {
            let distance = squared_distance_to_centroid(point, centroids[i]);
            if distance < min_distance {
                min_distance = distance;
                assignments[point_idx] = i;
            }
        }
    }
}

// Helper function for GPU kernel
#[cube]
fn squared_distance_to_centroid(point: F32, centroid: F32) -> F32 {
    let pow = F32::new(2.0);
    F32::powf(point[0] - centroid[0], pow)
        + F32::powf(point[1] - centroid[1], pow)
        + F32::powf(point[2] - centroid[2], pow)
}

#[cfg(test)]
mod tests {

    use super::*;
    use cubecl::wgpu::WgpuRuntime;

    #[test]
    fn test_kmeans() {
        let points = vec![
            [1, 1, 1, 1],
            [2, 3, 3, 1],
            [2, 3, 3, 1],
            [2, 3, 3, 1],
            [3, 3, 3, 1],
        ];

        let k = 2;
        let max_iterations = 10;
        let (assignments, centroids) = CubeKMeans::<WgpuRuntime>::from_device_and_config(
            Default::default(),
            KMeansConfig {
                initializer: crate::kmeans::Initializer::Random,
                k,
                max_iterations,
                tolerance: 0.001,
                seed: Some(42),
                ..Default::default()
            },
        )
        .run(&points)
        .unwrap();
        assert_eq!(assignments, vec![1, 0, 0, 0, 0]);
    }
}
