use wgpu::{Adapter, Device, DeviceDescriptor, Instance, Queue, RequestAdapterOptions};

pub async fn common_wgpu_setup() -> Result<(Instance, Adapter, Device, Queue), String> {
    let instance = wgpu::Instance::default();

    let adapter = instance
        .request_adapter(&RequestAdapterOptions::default())
        .await
        .unwrap();

    let (device, queue) = adapter
        .request_device(&DeviceDescriptor::default(), None)
        .await
        .unwrap();
    Ok((instance, adapter, device, queue))
}
