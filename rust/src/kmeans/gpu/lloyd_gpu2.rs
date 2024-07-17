use super::buffers::MappableBuffer;
use crate::kmeans::types::KMeansResult;
use crate::kmeans::utils::has_converged;
use crate::kmeans::KMeansConfig;
use crate::types::{Vec4, Vec4u};
use futures::executor::block_on;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    Device, DeviceDescriptor, PipelineCompilationOptions, PipelineLayout, PipelineLayoutDescriptor,
    Queue, RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

type Centroids = Vec<Vec4>;

struct ProcessBuffers {
    pixel_buffer: Buffer,
    centroid_buffer: MappableBuffer,
    assignment_buffer: MappableBuffer,
    global_assignment_counts_buffer: Buffer,
    global_assignment_sums_buffer: Buffer,
    bind_group: BindGroup,
}

#[derive(Debug)]
pub struct LloydAssignmentsAndCentroids {
    device: Device,
    queue: Queue,
    compute_pipeline: ComputePipeline,
    bind_group_layout: BindGroupLayout,
    pipeline_layout: PipelineLayout,
    config: KMeansConfig,
}

impl LloydAssignmentsAndCentroids {
    pub fn set_k(&mut self, k: usize) {
        self.config.k = k;
    }

    fn make_bind_group_layout(device: &Device) -> BindGroupLayout {
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("kmeans_bind_group_layout"),
            entries: &[
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
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

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

        let bind_group_layout = Self::make_bind_group_layout(&device);
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("kmeans_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let shader_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("kmeans_shader"),
            source: ShaderSource::Wgsl(include_str!("lloyd_gpu2.wgsl").into()),
        });

        let compute_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("kmeans_compute_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "main",
            compilation_options: PipelineCompilationOptions::default(),
        });

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
        pixels: &[Vec4u],
        centroids: &[Vec4],
    ) -> Result<ProcessBuffers, &'static str> {
        let pixel_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("pixel_buffer"),
            size: std::mem::size_of_val(pixels) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&pixel_buffer, 0, bytemuck::cast_slice(pixels));

        let centroid_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("centroid_buffer"),
            size: std::mem::size_of_val(centroids) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.queue
            .write_buffer(&centroid_buffer, 0, bytemuck::cast_slice(centroids));

        let centroid_staging_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("centroid_staging_buffer"),
            size: std::mem::size_of_val(centroids) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let assignment_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("assignment_buffer"),
            size: (pixels.len() * std::mem::size_of::<u32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let assignment_staging_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("assignment_staging_buffer"),
            size: (pixels.len() * std::mem::size_of::<u32>()) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let global_assignment_counts_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("global_assignment_counts_buffer"),
            size: (self.config.k * std::mem::size_of::<u32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let global_assignment_sums_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("global_assignment_sums_buffer"),
            size: (self.config.k * 3 * std::mem::size_of::<u32>()) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("kmeans_bind_group"),
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
                BindGroupEntry {
                    binding: 3,
                    resource: global_assignment_counts_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: global_assignment_sums_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(ProcessBuffers {
            pixel_buffer,
            centroid_buffer: MappableBuffer {
                gpu_buffer: centroid_buffer,
                staging_buffer: centroid_staging_buffer,
                size: std::mem::size_of_val(centroids) as u64,
            },
            assignment_buffer: MappableBuffer {
                gpu_buffer: assignment_buffer,
                staging_buffer: assignment_staging_buffer,
                size: (pixels.len() * std::mem::size_of::<u32>()) as u64,
            },
            global_assignment_counts_buffer,
            global_assignment_sums_buffer,
            bind_group,
        })
    }

    pub fn run(&self, pixels: &[Vec4u]) -> KMeansResult<Vec4> {
        block_on(self.run_async(pixels))
    }

    pub async fn run_async(&self, pixels: &[Vec4u]) -> KMeansResult<Vec4> {
        let vec4_pixels: Vec<Vec4> = pixels
            .iter()
            .map(|v| [v[0] as f32, v[1] as f32, v[2] as f32, v[3] as f32])
            .collect();
        let mut centroids: Vec<Vec4> = self.config.initializer.initialize_centroids(
            &vec4_pixels,
            self.config.k,
            self.config.seed,
        );

        let process_buffers = self.prepare_buffers(pixels, &centroids).unwrap();

        let mut iterations = 0;

        while iterations < self.config.max_iterations {
            let new_centroids = self.run_iteration(&process_buffers, pixels.len()).await?;

            if has_converged(&centroids, &new_centroids, self.config.tolerance) {
                centroids = new_centroids;
                break;
            }

            self.queue.write_buffer(
                &process_buffers.centroid_buffer.gpu_buffer,
                0,
                bytemuck::cast_slice(&new_centroids),
            );
            centroids = new_centroids;

            iterations += 1;
        }

        let assignments = self.read_assignments(&process_buffers).await?;

        Ok((assignments, centroids))
    }

    async fn run_iteration(
        &self,
        process_buffers: &ProcessBuffers,
        pixel_count: usize,
    ) -> Result<Centroids, &'static str> {
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });

        let num_workgroups = ((pixel_count as u32 + 63) / 64) as u32;

        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });

            pass.set_pipeline(&self.compute_pipeline);
            pass.set_bind_group(0, &process_buffers.bind_group, &[]);
            pass.insert_debug_marker("kmeans_iteration");
            pass.dispatch_workgroups(num_workgroups, 1, 1);
        }

        process_buffers
            .centroid_buffer
            .copy_to_staging_buffer(&mut encoder);
        self.queue.submit(Some(encoder.finish()));

        let centroids = process_buffers
            .centroid_buffer
            .read_back(&self.device)
            .await?;
        Ok(centroids)
    }

    async fn read_assignments(
        &self,
        process_buffers: &ProcessBuffers,
    ) -> Result<Vec<usize>, &'static str> {
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });

        process_buffers
            .assignment_buffer
            .copy_to_staging_buffer(&mut encoder);
        self.queue.submit(Some(encoder.finish()));

        let assignments: Vec<u32> = process_buffers
            .assignment_buffer
            .read_back(&self.device)
            .await?;
        Ok(assignments.into_iter().map(|a| a as usize).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kmeans::gpu::GpuAlgorithm;
    use crate::kmeans::initializer::Initializer;
    use futures::executor::block_on;

    fn create_test_config() -> KMeansConfig {
        KMeansConfig {
            k: 3,
            max_iterations: 10,
            tolerance: 0.001,
            algorithm: GpuAlgorithm::LloydAssignmentsAndCentroids.into(),
            initializer: Initializer::Random,
            seed: Some(42),
        }
    }

    #[test]
    fn test_kmeans_gpu_basic() {
        let config = create_test_config();
        let pixels: Vec<Vec4u> = vec![
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [10, 10, 10, 10],
            [11, 11, 11, 11],
            [12, 12, 12, 12],
            [20, 20, 20, 20],
            [21, 21, 21, 21],
            [22, 22, 22, 22],
        ];

        let kmeans = block_on(LloydAssignmentsAndCentroids::from_config(config));
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
}
