use crate::kmeans::config::KMeansConfig;
use crate::kmeans::utils::{find_closest_centroid, has_converged, initialize_centroids};
use crate::types::Vec3;
use std::sync::mpsc;
use futures::executor::block_on;
use itertools::izip;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, Device, DeviceDescriptor, PipelineCompilationOptions, PipelineLayout, PipelineLayoutDescriptor, Queue, RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource, ShaderStages
};

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
    pub async fn new(config: KMeansConfig) -> Self {
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


    fn prepare_buffers(&self, pixels: &[Vec3], centroids: &[Vec3], assignments: &[u32]) -> Result<ProcessBuffers, &'static str> {
        let pixel_buffer = self.device.create_buffer(&BufferDescriptor {
            label: None,
            // 3 floats per pixel, 4 bytes per float (as they are f32)
            size: (pixels.len() * 3 * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });

        let centroid_buffer = self.device.create_buffer(&BufferDescriptor {
            label: None,
            // 3 floats per centroid, 4 bytes per float (as they are f32)
            size: (centroids.len() * 3 * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });

        let assignment_size: u64 = assignments.len() as u64 * 4;
        let assignment_buffer = self.device.create_buffer(&BufferDescriptor {
            label: None,
            // 1 int per assignment
            // technically I think we can get away with a u8 here
            // since our color space is limited to 256 colors
            // but alignment. we'll maybe do this later
            size: assignment_size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

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

    pub fn quantize(&self, pixels: &[Vec3]) -> Result<Vec<usize>, &'static str> {
        block_on(self.quantize_async(pixels))
    }

    async fn quantize_async(&self, pixels: &[Vec3]) -> Result<Vec<usize>, &'static str> {
        // initialization
        let mut centroids = initialize_centroids(pixels, self.config.k, self.config.seed);
        let mut new_centroids = centroids.clone();

        let mut assignments: Vec<u32> = vec![0; pixels.len()];
        let assignment_size: u64 = assignments.len() as u64 * 4;

        let process_buffers = self.prepare_buffers(pixels, &centroids, &assignments).unwrap();
        let mut converged = false;
        let mut iterations = 0;


        while iterations < self.config.max_iterations && !converged {
            // Get all pixel assignments
            let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
                label: None,
            });
            {
                let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });

                pass.set_pipeline(&self.compute_pipeline);
                pass.set_bind_group(0, &process_buffers.bind_group, &[]);
                pass.dispatch_workgroups(pixels.len() as u32, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&process_buffers.assignment_buffer, 0, &process_buffers.staging_buffer, 0, assignment_size);
            self.queue.submit(Some(encoder.finish()));

            let buffer_slice = process_buffers.staging_buffer.slice(..);
            let (sender, receiver) = futures::channel::oneshot::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
            self.device.poll(wgpu::Maintain::Wait);

            if let Ok(Ok(())) = receiver.await {
                let data = buffer_slice.get_mapped_range();
                assignments = bytemuck::cast_slice(&data).to_vec();
                drop(data);
                process_buffers.staging_buffer.unmap();
            } else {
                return Err("Failed to read back the assignment buffer");
            }

            dbg!("Succesfully ran gpu loop");

            let mut cluster_sums = vec![[0.0; 3]; self.config.k];
            let mut cluster_counts   = vec![0.; self.config.k];

            for (i, pixel) in pixels.iter().enumerate() {
                let cluster = assignments[i] as usize;
                cluster_sums[cluster] = [cluster_sums[cluster][0] + pixel[0], cluster_sums[cluster][1] + pixel[1], cluster_sums[cluster][2] + pixel[2]];
                cluster_counts[cluster] += 1.;
            }

            for (centroid, sum, count) in izip!(new_centroids.iter_mut(), cluster_sums.iter(), cluster_counts.iter()) {
                *centroid = [sum[0] / count, sum[1] / count, sum[2] / count];
            }
            self.queue.write_buffer(&process_buffers.centroid_buffer, 0, bytemuck::cast_slice(&new_centroids));
            // Check for convergence
            converged = has_converged(&centroids, &new_centroids, self.config.tolerance);
            std::mem::swap(&mut centroids, &mut new_centroids);
            iterations += 1;
        }



        todo!()



    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kmeans::config::KMeansAlgorithm;
    use futures::executor::block_on;

    fn create_test_config() -> KMeansConfig {
        KMeansConfig {
            k: 3,
            max_iterations: 100,
            tolerance: 0.001,
            algorithm: KMeansAlgorithm::Lloyd,
            seed: Some(42),
        }
    }

    #[test]
    fn test_kmeans_gpu_basic() {
        let config = create_test_config();
        let pixels: Vec<Vec3> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [10.0, 10.0, 10.0],
            [11.0, 11.0, 11.0],
            [12.0, 12.0, 12.0],
            [20.0, 20.0, 20.0],
            [21.0, 21.0, 21.0],
            [22.0, 22.0, 22.0],
        ];

        let kmeans = block_on(KMeansGpu::new(config));
        let result = block_on(kmeans.quantize_async(&pixels)).unwrap();

        assert_eq!(result.len(), pixels.len());
        
        // Check that pixels close to each other are in the same cluster
        assert_eq!(result[0], result[1]);
        assert_eq!(result[1], result[2]);
        assert_eq!(result[3], result[4]);
        assert_eq!(result[4], result[5]);
        assert_eq!(result[6], result[7]);
        assert_eq!(result[7], result[8]);

        // Check that pixels far from each other are in different clusters
        assert_ne!(result[0], result[3]);
        assert_ne!(result[3], result[6]);
        assert_ne!(result[0], result[6]);
    }

    #[test]
    fn test_kmeans_gpu_convergence() {
        let config = KMeansConfig {
            k: 2,
            max_iterations: 100,
            tolerance: 0.001,
            algorithm: KMeansAlgorithm::Lloyd,
            seed: Some(42),
        };

        let pixels: Vec<Vec3> = vec![
            [0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [10.0, 10.0, 10.0],
            [10.1, 10.1, 10.1],
            [10.2, 10.2, 10.2],
        ];

        let kmeans = block_on(KMeansGpu::new(config));
        let result = block_on(kmeans.quantize_async(&pixels)).unwrap();

        assert_eq!(result.len(), pixels.len());
        
        // Check that the algorithm converged to two clusters
        assert_eq!(result[0], result[1]);
        assert_eq!(result[1], result[2]);
        assert_eq!(result[3], result[4]);
        assert_eq!(result[4], result[5]);
        assert_ne!(result[0], result[3]);
    }

    #[test]
    fn test_kmeans_gpu_empty_input() {
        let config = create_test_config();
        let pixels: Vec<Vec3> = vec![];

        let kmeans = block_on(KMeansGpu::new(config));
        let result = block_on(kmeans.quantize_async(&pixels)).unwrap();

        assert_eq!(result.len(), 0);
    }
}