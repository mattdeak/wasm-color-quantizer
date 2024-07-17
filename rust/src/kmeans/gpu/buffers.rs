use wgpu::Buffer;
use wgpu::CommandEncoder;
use wgpu::Device;

pub struct MappableBuffer {
    pub gpu_buffer: Buffer,
    pub staging_buffer: Buffer,
    pub size: u64,
}

// Provides some utilities to make it easier to read back a buffer from the GPU
impl MappableBuffer {
    pub async fn read_back<T: bytemuck::Pod>(
        &self,
        device: &Device,
    ) -> Result<Vec<T>, &'static str> {
        let buffer_slice = self.staging_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        device.poll(wgpu::Maintain::Wait).panic_on_timeout();

        if let Ok(Ok(())) = receiver.await {
            let buffer_data = buffer_slice.get_mapped_range();
            let data: Vec<T> = bytemuck::cast_slice(&buffer_data).to_vec();
            drop(buffer_data);
            self.staging_buffer.unmap();
            Ok(data)
        } else {
            Err("Failed to read back the buffer")
        }
    }

    pub fn copy_to_staging_buffer(&self, encoder: &mut CommandEncoder) {
        encoder.copy_buffer_to_buffer(&self.gpu_buffer, 0, &self.staging_buffer, 0, self.size);
    }
}
