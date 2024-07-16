use crate::kmeans::config::KMeansConfig;
use crate::kmeans::utils::has_converged;
use crate::types::Vec4;
use futures::executor::block_on;
use itertools::izip;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    Device, DeviceDescriptor, PipelineCompilationOptions, PipelineLayout, PipelineLayoutDescriptor,
    Queue, RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use super::types::KMeansResult;

type Assignments = Vec<usize>;
type Centroids = Vec<Vec4>;

#[derive(Debug)]
pub struct KMeansGpu {
    device: Device,
    queue: Queue,
    compute_pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
    pipeline_layout: PipelineLayout,
    config: KMeansConfig,
}

struct ProcessBuffers {
    pixel_buffer: Buffer,
    centroid_buffer: Buffer,
    assignment_buffer: Buffer,
    staging_buffer: Buffer,
    bind_group: BindGroup,
}

impl KMeansGpu {
    pub async fn from_config(config: KMeansConfig) -> Self {
        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&RequestAdapterOptions::default())
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor::default(), None)
            .await
            .unwrap();

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("kmeans_bind_group_layout".into()),
            entries: &[
                // Pixel Group
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Centroids
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        // true for now, but we may want to also update centroids in a shader later
                        // TODO
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Assignments
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let shader_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("kmeans_shader".into()),
            source: ShaderSource::Wgsl(include_str!("lloyd_gpu.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("kmeans_pipeline_layout".into()),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("kmeans_compute_pipeline".into()),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "main",
            compilation_options: PipelineCompilationOptions::default(),
        });

        log::info!("Adapter: {:?}", adapter.get_info());

        Self {
            device,
            queue,
            compute_pipeline,
            bind_group_layout,
            pipeline_layout,
            config,
        }
    }

    fn prepare_buffers(
        &self,
        pixels: &[Vec4],
        centroids: &[Vec4],
        assignments: &[u32],
    ) -> Result<ProcessBuffers, &'static str> {
        let pixel_buffer = self.device.create_buffer(&BufferDescriptor {
            label: None,
            // 3 floats per pixel, 4 bytes per float (as they are f32)
            size: std::mem::size_of_val(pixels) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.queue
            .write_buffer(&pixel_buffer, 0, bytemuck::cast_slice(pixels));

        let centroid_buffer = self.device.create_buffer(&BufferDescriptor {
            label: None,
            // 3 floats per centroid, 4 bytes per float (as they are f32), but we have to align
            // to 16 bytes to match the alignment of the pixel buffer
            size: std::mem::size_of_val(centroids) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.queue
            .write_buffer(&centroid_buffer, 0, bytemuck::cast_slice(centroids));

        let assignment_size: u64 = (pixels.len() * std::mem::size_of::<u32>()) as u64;
        let assignment_buffer = self.device.create_buffer(&BufferDescriptor {
            label: None,
            // 1 int per assignment
            // technically I think we can get away with a u8 here
            // since our color space is limited to 256 colors
            // but alignment. we'll maybe do this later
            size: assignment_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // this should just be zeros but that's fine
        self.queue
            .write_buffer(&assignment_buffer, 0, bytemuck::cast_slice(assignments));

        let staging_buffer = self.device.create_buffer(&BufferDescriptor {
            label: None,
            size: assignment_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &self.bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: pixel_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: centroid_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: assignment_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(ProcessBuffers {
            pixel_buffer,
            centroid_buffer,
            assignment_buffer,
            staging_buffer,
            bind_group,
        })
    }

    pub fn run(&self, pixels: &[Vec4]) -> KMeansResult<Vec4> {
        block_on(self.run_async(pixels))
    }

    pub async fn run_async(&self, pixels: &[Vec4]) -> KMeansResult<Vec4> {
        // initialization
        let mut centroids =
            self.config
                .initializer
                .initialize_centroids(pixels, self.config.k, self.config.seed);
        dbg!("Initial centroids: {:?}", &centroids);
        let mut new_centroids = centroids.clone();

        let mut assignments: Vec<u32> = vec![0; pixels.len()];

        let process_buffers = self
            .prepare_buffers(pixels, &centroids, &assignments)
            .unwrap();
        let mut converged = false;
        let mut iterations = 0;

        while iterations < self.config.max_iterations && !converged {
            self.update_assignments_gpu(&process_buffers, pixels.len(), &mut assignments)
                .await?;
            self.recalculate_centroids(pixels, &assignments, &mut new_centroids);

            converged = has_converged(&centroids, &new_centroids, self.config.tolerance);

            std::mem::swap(&mut centroids, &mut new_centroids);
            self.queue.write_buffer(
                &process_buffers.centroid_buffer,
                0,
                bytemuck::cast_slice(&centroids),
            );
            iterations += 1;
        }

        let assignments = assignments.iter().map(|v| *v as usize).collect();
        Ok((assignments, centroids))
    }

    fn recalculate_centroids(
        &self,
        pixels: &[Vec4],
        assignments: &[u32],
        new_centroids: &mut [Vec4],
    ) {
        let mut cluster_sums = vec![[0.0; 3]; self.config.k];
        let mut cluster_counts = vec![0.; self.config.k];

        for (i, pixel) in pixels.iter().enumerate() {
            let cluster = assignments[i] as usize;
            cluster_sums[cluster] = [
                cluster_sums[cluster][0] + pixel[0],
                cluster_sums[cluster][1] + pixel[1],
                cluster_sums[cluster][2] + pixel[2],
            ];
            cluster_counts[cluster] += 1.;
        }

        // dbg!("Cluster sums: {:?}", cluster_sums);
        dbg!("Cluster counts: {:?}", &cluster_counts);

        for (centroid, sum, count) in izip!(
            new_centroids.iter_mut(),
            cluster_sums.into_iter(),
            cluster_counts.into_iter()
        ) {
            if count > 0. {
                *centroid = [sum[0] / count, sum[1] / count, sum[2] / count, 0.0];
            } else {
                // although this is technically wrong
                // since this represents a point in 3d color space
                // but idk
                *centroid = [0.0, 0.0, 0.0, 0.0];
            }
        }
    }

    async fn update_assignments_gpu(
        &self,
        process_buffers: &ProcessBuffers,
        pixel_count: usize,
        local_assignments: &mut [u32],
    ) -> Result<(), &'static str> {
        // one u32 for each pixel
        let assignment_size = (pixel_count * std::mem::size_of::<u32>()) as wgpu::BufferAddress;

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.compute_pipeline);
            pass.set_bind_group(0, &process_buffers.bind_group, &[]);
            pass.insert_debug_marker("update_assignments_gpu");
            pass.dispatch_workgroups((pixel_count as u32 + 63) / 64, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &process_buffers.assignment_buffer,
            0,
            &process_buffers.staging_buffer,
            0,
            assignment_size,
        );
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = process_buffers.staging_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait).panic_on_timeout();

        if let Ok(Ok(())) = receiver.await {
            let data = buffer_slice.get_mapped_range();
            dbg!("data: {:?}", &data);
            local_assignments.copy_from_slice(bytemuck::cast_slice(&data));
            drop(data);
            process_buffers.staging_buffer.unmap();
            Ok(())
        } else {
            Err("Failed to read back the assignment buffer")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kmeans::config::KMeansAlgorithm;
    use crate::kmeans::initializer::Initializer;
    use futures::executor::block_on;

    fn create_test_config() -> KMeansConfig {
        KMeansConfig {
            k: 3,
            max_iterations: 10,
            tolerance: 0.001,
            algorithm: KMeansAlgorithm::Lloyd,
            initializer: Initializer::Random,
            seed: Some(42),
        }
    }

    #[test]
    fn test_kmeans_gpu_basic() {
        let config = create_test_config();
        let pixels: Vec<Vec4> = vec![
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
            [10.0, 10.0, 10.0, 10.0],
            [11.0, 11.0, 11.0, 11.0],
            [12.0, 12.0, 12.0, 12.0],
            [20.0, 20.0, 20.0, 20.0],
            [21.0, 21.0, 21.0, 21.0],
            [22.0, 22.0, 22.0, 22.0],
        ];

        let kmeans = block_on(KMeansGpu::from_config(config));
        let (assignments, centroids) = block_on(kmeans.run_async(&pixels)).unwrap();

        assert_eq!(assignments.len(), pixels.len());

        // Check that pixels close to each other are in the same cluster
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[1], assignments[2]);
        assert_eq!(assignments[3], assignments[4]);
        assert_eq!(assignments[4], assignments[5]);
        assert_eq!(assignments[6], assignments[7]);
        assert_eq!(assignments[7], assignments[8]);

        // Check that pixels far from each other are in different clusters
        assert_ne!(assignments[0], assignments[3]);
        assert_ne!(assignments[3], assignments[6]);
        assert_ne!(assignments[0], assignments[6]);
    }

    #[test]
    fn test_kmeans_gpu_convergence() {
        let config = KMeansConfig {
            k: 2,
            max_iterations: 100,
            tolerance: 0.001,
            algorithm: KMeansAlgorithm::Lloyd,
            initializer: Initializer::Random,
            seed: Some(42),
        };

        let pixels: Vec<Vec4> = vec![
            [0.0, 0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2, 0.2],
            [10.0, 10.0, 10.0, 10.0],
            [10.1, 10.1, 10.1, 10.1],
            [10.2, 10.2, 10.2, 10.2],
        ];

        let kmeans = block_on(KMeansGpu::from_config(config));
        let (assignments, centroids) = block_on(kmeans.run_async(&pixels)).unwrap();

        assert_eq!(assignments.len(), pixels.len());

        // Check that the algorithm converged to two clusters
        assert_eq!(assignments[0], assignments[1]);
        assert_eq!(assignments[1], assignments[2]);
        assert_eq!(assignments[3], assignments[4]);
        assert_eq!(assignments[4], assignments[5]);
        assert_ne!(assignments[0], assignments[3]);
    }

    #[test]
    fn test_kmeans_gpu_empty_input() {
        let config = create_test_config();
        let pixels: Vec<Vec4> = vec![];

        let kmeans = block_on(KMeansGpu::from_config(config));
        let (assignments, centroids) = block_on(kmeans.run_async(&pixels)).unwrap();

        assert_eq!(assignments.len(), 0);
        assert_eq!(centroids.len(), 0);
    }

    #[test]
    fn test_update_assignments_gpu() {
        let config = create_test_config();
        let kmeans = block_on(KMeansGpu::from_config(config));

        let pixels: Vec<Vec4> = vec![
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [10.0, 10.0, 10.0, 10.0],
            [11.0, 11.0, 11.0, 11.0],
        ];
        let centroids: Vec<Vec4> = vec![[0.5, 0.5, 0.5, 0.5], [10.5, 10.5, 10.5, 10.5]];
        let mut initial_assignments: Vec<u32> = vec![0, 0, 0, 0];

        let process_buffers = kmeans
            .prepare_buffers(&pixels, &centroids, &initial_assignments)
            .unwrap();

        block_on(kmeans.update_assignments_gpu(
            &process_buffers,
            pixels.len(),
            &mut initial_assignments,
        ))
        .unwrap();

        dbg!("Updated assignments: {:?}", &initial_assignments);

        assert_eq!(initial_assignments.len(), pixels.len());
        assert_eq!(initial_assignments[0], 0);
        assert_eq!(initial_assignments[1], 0);
        assert_eq!(initial_assignments[2], 1);
        assert_eq!(initial_assignments[3], 1);
    }
}
