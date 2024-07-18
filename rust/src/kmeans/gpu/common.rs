use wgpu::{Adapter, Device, DeviceDescriptor, Instance, Queue, RequestAdapterOptions};

pub async fn common_wgpu_setup() -> Result<(Instance, Adapter, Device, Queue), String> {
    let instance = wgpu::Instance::default();

    let adapter = instance
        .request_adapter(&RequestAdapterOptions::default())
        .await
        .expect("Failed to request adapter");

    let (device, queue) = adapter
        .request_device(&DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
        },
        None,
    )
    .await
    .expect("Failed to request device");
    Ok((instance, adapter, device, queue))
}
