//! wgpu device creation and adapter selection.
//!
//! Absorbed from `crates/spike/src/wgpu_setup.rs`, which proved that CubeCL
//! adopts a `wgpu::Device` we create: pack instance/adapter/device/queue into a
//! `WgpuSetup` and call `init_device`. That is what lets the GUI draw straight
//! from CubeCL's own buffers, with no host round trip.
//!
//! Nothing here creates a surface or a window; the instance is built with
//! `new_without_display_handle`. The GUI builds its own instance with a
//! display handle and hands the resulting setup to [`client_on`].

use cubecl::prelude::*;
use cubecl_runtime::server::Handle;
use cubecl_wgpu::{RuntimeOptions, WgpuDevice, WgpuRuntime, WgpuSetup};

pub fn instance(backends: wgpu::Backends) -> wgpu::Instance {
  wgpu::Instance::new(wgpu::InstanceDescriptor {
    backends,
    ..wgpu::InstanceDescriptor::new_without_display_handle()
  })
}

pub fn list_adapters(backends: wgpu::Backends) -> Vec<wgpu::AdapterInfo> {
  let instance = instance(backends);
  pollster::block_on(instance.enumerate_adapters(backends))
    .into_iter()
    .map(|a| a.get_info())
    .collect()
}

/// Build a `WgpuSetup` ourselves, picking the adapter by a substring of its
/// name, or the most capable one when no filter is given.
pub fn headless_setup(
  backends: wgpu::Backends,
  name_filter: Option<&str>,
) -> anyhow::Result<WgpuSetup> {
  let instance = instance(backends);
  let adapters = pollster::block_on(instance.enumerate_adapters(backends));
  let adapter = match name_filter {
    Some(filter) => adapters
      .into_iter()
      .find(|a| {
        a.get_info()
          .name
          .to_lowercase()
          .contains(&filter.to_lowercase())
      })
      .ok_or_else(|| anyhow::anyhow!("no wgpu adapter matching {filter:?}"))?,
    None => adapters
      .into_iter()
      .max_by_key(|a| match a.get_info().device_type {
        wgpu::DeviceType::DiscreteGpu => 3,
        wgpu::DeviceType::IntegratedGpu => 2,
        wgpu::DeviceType::VirtualGpu => 1,
        _ => 0,
      })
      .ok_or_else(|| anyhow::anyhow!("no wgpu adapter at all"))?,
  };
  Ok(setup_from_adapter(instance, adapter))
}

/// Request a device from an adapter the same way `cubecl_wgpu`'s own
/// `request_device` does, so the device we hand CubeCL has everything its own
/// path would have asked for.
pub fn setup_from_adapter(instance: wgpu::Instance, adapter: wgpu::Adapter) -> WgpuSetup {
  let backend = adapter.get_info().backend;
  let (device, queue) = pollster::block_on(
    adapter.request_device(&wgpu::DeviceDescriptor {
      label: Some("alife device"),
      required_features: adapter
        .features()
        .difference(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS),
      required_limits: adapter.limits(),
      memory_hints: wgpu::MemoryHints::MemoryUsage,
      trace: wgpu::Trace::Off,
      experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
    }),
  )
  .expect("request_device");

  WgpuSetup {
    instance,
    adapter,
    device,
    queue,
    backend,
  }
}

/// Hand an existing setup to CubeCL and get a client back.
pub fn client_on(setup: &WgpuSetup) -> (WgpuDevice, ComputeClient<WgpuRuntime>) {
  let device = cubecl_wgpu::init_device(setup.clone(), RuntimeOptions::default());
  let client = WgpuRuntime::client(&device);
  (device, client)
}

/// The `wgpu::Buffer` behind a CubeCL handle, plus the handle's window into it.
///
/// A handle is a slice of a pooled buffer, not a buffer: the offset is real and
/// bindings must use it rather than `as_entire_binding()`. CubeCL also
/// allocates without `VERTEX` usage, so a renderer reads these as storage
/// buffers indexed by `@builtin(vertex_index)`.
pub struct SharedBuffer {
  pub buffer: wgpu::Buffer,
  pub offset: u64,
  pub size: u64,
}

pub fn shared_buffer(client: &ComputeClient<WgpuRuntime>, handle: &Handle) -> SharedBuffer {
  let managed = client
    .get_resource(handle.clone())
    .expect("get_resource on a live handle");
  let res = managed.resource();
  SharedBuffer {
    buffer: res.buffer.clone(),
    offset: res.offset,
    size: res.size,
  }
}

impl SharedBuffer {
  pub fn binding(&self) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
      buffer: &self.buffer,
      offset: self.offset,
      size: core::num::NonZeroU64::new(self.size.next_multiple_of(4)),
    })
  }
}
