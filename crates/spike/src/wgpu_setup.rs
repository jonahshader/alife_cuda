//! Headless wgpu setup, adapter selection, and the device/buffer sharing proof.
//!
//! Nothing here creates a surface or a window: the instance is built with
//! `new_without_display_handle`, and the only render target is an offscreen
//! texture.

use cubecl::prelude::*;
use cubecl_runtime::server::Handle;
use cubecl_wgpu::{RuntimeOptions, WgpuDevice, WgpuRuntime, WgpuSetup};
use wgpu::util::DeviceExt;

pub fn list_adapters(backends: wgpu::Backends) -> Vec<wgpu::AdapterInfo> {
  let instance = instance(backends);
  pollster::block_on(instance.enumerate_adapters(backends))
    .into_iter()
    .map(|a| a.get_info())
    .collect()
}

fn instance(backends: wgpu::Backends) -> wgpu::Instance {
  wgpu::Instance::new(wgpu::InstanceDescriptor {
    backends,
    ..wgpu::InstanceDescriptor::new_without_display_handle()
  })
}

/// Build a `WgpuSetup` ourselves, exactly as a renderer would, picking the
/// adapter by a substring of its name. This is the "share an existing device"
/// path: CubeCL never creates the device.
pub fn headless_setup(backends: wgpu::Backends, name_filter: Option<&str>) -> WgpuSetup {
  let instance = instance(backends);
  let adapters = pollster::block_on(instance.enumerate_adapters(backends));
  let adapter = match name_filter {
    Some(f) => adapters
      .into_iter()
      .find(|a| a.get_info().name.to_lowercase().contains(&f.to_lowercase()))
      .unwrap_or_else(|| panic!("no wgpu adapter matching {f:?}")),
    None => adapters
      .into_iter()
      .max_by_key(|a| match a.get_info().device_type {
        wgpu::DeviceType::DiscreteGpu => 3,
        wgpu::DeviceType::IntegratedGpu => 2,
        wgpu::DeviceType::VirtualGpu => 1,
        _ => 0,
      })
      .expect("no wgpu adapter at all"),
  };

  let backend = adapter.get_info().backend;
  // Mirrors `cubecl_wgpu::backend::wgsl::request_device`, so the device we
  // hand CubeCL has everything its own path would have asked for.
  let (device, queue) = pollster::block_on(
    adapter.request_device(&wgpu::DeviceDescriptor {
      label: Some("spike shared device"),
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
  fn binding(&self) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
      buffer: &self.buffer,
      offset: self.offset,
      size: core::num::NonZeroU64::new(self.size.next_multiple_of(4)),
    })
  }
}

/// Copy `count` floats out of a CubeCL-owned buffer using only wgpu's own API:
/// our encoder, our queue, our staging buffer, our map.
pub fn read_back_with_wgpu(setup: &WgpuSetup, shared: &SharedBuffer, count: usize) -> Vec<f32> {
  let bytes = (count * size_of::<f32>()) as u64;
  let staging = setup.device.create_buffer(&wgpu::BufferDescriptor {
    label: Some("spike staging"),
    size: bytes,
    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    mapped_at_creation: false,
  });

  let mut encoder = setup
    .device
    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
  encoder.copy_buffer_to_buffer(&shared.buffer, shared.offset, &staging, 0, bytes);
  setup.queue.submit([encoder.finish()]);

  map_read_f32(&setup.device, &staging, count)
}

/// Bind the CubeCL buffer as a read-only storage buffer in a compute pass we
/// dispatch ourselves, and write a transformed copy into our own buffer.
pub fn compute_pass_on_shared(setup: &WgpuSetup, shared: &SharedBuffer, count: usize) -> Vec<f32> {
  let shader = setup
    .device
    .create_shader_module(wgpu::ShaderModuleDescriptor {
      label: Some("spike compute"),
      source: wgpu::ShaderSource::Wgsl(
        r#"
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < arrayLength(&dst)) {
        dst[i] = src[i] * 2.0;
    }
}
"#
        .into(),
      ),
    });

  let out = setup.device.create_buffer(&wgpu::BufferDescriptor {
    label: Some("spike compute out"),
    size: (count * size_of::<f32>()) as u64,
    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    mapped_at_creation: false,
  });

  let pipeline = setup
    .device
    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
      label: None,
      layout: None,
      module: &shader,
      entry_point: Some("main"),
      compilation_options: Default::default(),
      cache: None,
    });

  let bind_group = setup.device.create_bind_group(&wgpu::BindGroupDescriptor {
    label: None,
    layout: &pipeline.get_bind_group_layout(0),
    entries: &[
      wgpu::BindGroupEntry {
        binding: 0,
        resource: shared.binding(),
      },
      wgpu::BindGroupEntry {
        binding: 1,
        resource: out.as_entire_binding(),
      },
    ],
  });

  let staging = setup.device.create_buffer(&wgpu::BufferDescriptor {
    label: None,
    size: (count * size_of::<f32>()) as u64,
    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    mapped_at_creation: false,
  });

  let mut encoder = setup
    .device
    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
  {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
      label: None,
      timestamp_writes: None,
    });
    pass.set_pipeline(&pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(count.div_ceil(64) as u32, 1, 1);
  }
  encoder.copy_buffer_to_buffer(&out, 0, &staging, 0, (count * size_of::<f32>()) as u64);
  setup.queue.submit([encoder.finish()]);

  map_read_f32(&setup.device, &staging, count)
}

/// Bind the same CubeCL buffer in a real (offscreen) render pass: the fragment
/// shader reads it as a storage buffer and writes one of its values into a 1x1
/// R32Float attachment, which we then read back.
pub fn render_pass_on_shared(setup: &WgpuSetup, shared: &SharedBuffer, index: u32) -> f32 {
  let shader = setup
    .device
    .create_shader_module(wgpu::ShaderModuleDescriptor {
      label: Some("spike render"),
      source: wgpu::ShaderSource::Wgsl(
        r#"
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<uniform> pick: vec4<u32>;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    // Oversized triangle covering the whole (1x1) target.
    let xy = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return vec4<f32>(xy[vi], 0.0, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> {
    return vec4<f32>(src[pick.x], 0.0, 0.0, 0.0);
}
"#
        .into(),
      ),
    });

  let pick = setup
    .device
    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: None,
      contents: &[index, 0, 0, 0]
        .iter()
        .flat_map(|v: &u32| v.to_ne_bytes())
        .collect::<Vec<u8>>(),
      usage: wgpu::BufferUsages::UNIFORM,
    });

  let texture = setup.device.create_texture(&wgpu::TextureDescriptor {
    label: None,
    size: wgpu::Extent3d {
      width: 1,
      height: 1,
      depth_or_array_layers: 1,
    },
    mip_level_count: 1,
    sample_count: 1,
    dimension: wgpu::TextureDimension::D2,
    format: wgpu::TextureFormat::R32Float,
    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
    view_formats: &[],
  });
  let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

  let pipeline = setup
    .device
    .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
      label: None,
      layout: None,
      vertex: wgpu::VertexState {
        module: &shader,
        entry_point: Some("vs"),
        compilation_options: Default::default(),
        buffers: &[],
      },
      primitive: wgpu::PrimitiveState::default(),
      depth_stencil: None,
      multisample: wgpu::MultisampleState::default(),
      fragment: Some(wgpu::FragmentState {
        module: &shader,
        entry_point: Some("fs"),
        compilation_options: Default::default(),
        targets: &[Some(wgpu::ColorTargetState {
          format: wgpu::TextureFormat::R32Float,
          blend: None,
          write_mask: wgpu::ColorWrites::ALL,
        })],
      }),
      multiview_mask: None,
      cache: None,
    });

  let bind_group = setup.device.create_bind_group(&wgpu::BindGroupDescriptor {
    label: None,
    layout: &pipeline.get_bind_group_layout(0),
    entries: &[
      wgpu::BindGroupEntry {
        binding: 0,
        resource: shared.binding(),
      },
      wgpu::BindGroupEntry {
        binding: 1,
        resource: pick.as_entire_binding(),
      },
    ],
  });

  // copy_texture_to_buffer wants rows padded to 256 bytes.
  let staging = setup.device.create_buffer(&wgpu::BufferDescriptor {
    label: None,
    size: 256,
    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    mapped_at_creation: false,
  });

  let mut encoder = setup
    .device
    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
  {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
      label: None,
      color_attachments: &[Some(wgpu::RenderPassColorAttachment {
        view: &view,
        depth_slice: None,
        resolve_target: None,
        ops: wgpu::Operations {
          load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
          store: wgpu::StoreOp::Store,
        },
      })],
      depth_stencil_attachment: None,
      timestamp_writes: None,
      occlusion_query_set: None,
      multiview_mask: None,
    });
    pass.set_pipeline(&pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.draw(0..3, 0..1);
  }
  encoder.copy_texture_to_buffer(
    wgpu::TexelCopyTextureInfo {
      texture: &texture,
      mip_level: 0,
      origin: wgpu::Origin3d::ZERO,
      aspect: wgpu::TextureAspect::All,
    },
    wgpu::TexelCopyBufferInfo {
      buffer: &staging,
      layout: wgpu::TexelCopyBufferLayout {
        offset: 0,
        bytes_per_row: Some(256),
        rows_per_image: Some(1),
      },
    },
    wgpu::Extent3d {
      width: 1,
      height: 1,
      depth_or_array_layers: 1,
    },
  );
  setup.queue.submit([encoder.finish()]);

  map_read_f32(&setup.device, &staging, 1)[0]
}

fn map_read_f32(device: &wgpu::Device, staging: &wgpu::Buffer, count: usize) -> Vec<f32> {
  let slice = staging.slice(..);
  let (tx, rx) = std::sync::mpsc::channel();
  slice.map_async(wgpu::MapMode::Read, move |r| {
    let _ = tx.send(r);
  });
  device
    .poll(wgpu::PollType::Wait {
      submission_index: None,
      timeout: None,
    })
    .expect("poll");
  rx.recv().expect("map callback").expect("map");
  let data = slice.get_mapped_range().expect("mapped range");
  let out = data
    .chunks_exact(4)
    .take(count)
    .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
    .collect();
  drop(data);
  staging.unmap();
  out
}
