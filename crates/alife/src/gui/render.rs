//! Drawing the sim straight out of CubeCL's buffers.
//!
//! The CubeCL spike settled the two rules this has to respect (`docs/organism.md`,
//! decisions): a CubeCL handle is a slice of a pooled buffer, so a binding must
//! carry its offset and never use `as_entire_binding()`; and CubeCL allocates
//! without `VERTEX` usage, so there is no vertex buffer to bind — the shaders
//! read the same storage buffers the kernels wrote and index them by
//! `@builtin(vertex_index)`.
//!
//! Both passes therefore issue a single `draw` with no vertex or index buffer:
//! six vertices per soil cell and six per particle.

use std::sync::Arc;

use alife_sim::wgpu_backend::shared_buffer;
use cubecl::prelude::ComputeClient;
use cubecl_wgpu::WgpuRuntime;
use eframe::egui_wgpu::{CallbackResources, CallbackTrait, ScreenDescriptor};
use wgpu::util::DeviceExt as _;

/// World-to-clip transform and the sizes the shaders need.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ViewUniform {
  pub scale: [f32; 2],
  pub offset: [f32; 2],
  pub soil_cell_size: f32,
  pub particle_radius: f32,
  pub _pad: [f32; 2],
  pub soil_width: u32,
  pub soil_height: u32,
  pub particle_count: u32,
  pub debug_evap: u32,
}

const SHADER: &str = r#"
struct View {
  scale: vec2<f32>,
  offset: vec2<f32>,
  soil_cell_size: f32,
  particle_radius: f32,
  pad: vec2<f32>,
  soil_width: u32,
  soil_height: u32,
  particle_count: u32,
  debug_evap: u32,
};

@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<storage, read> sand: array<f32>;
@group(0) @binding(2) var<storage, read> silt: array<f32>;
@group(0) @binding(3) var<storage, read> clay: array<f32>;
@group(0) @binding(4) var<storage, read> pos: array<f32>;
@group(0) @binding(5) var<storage, read> state: array<u32>;
@group(0) @binding(6) var<storage, read> evap_prob: array<f32>;

fn to_clip(world: vec2<f32>) -> vec4<f32> {
  return vec4<f32>(world * view.scale + view.offset, 0.0, 1.0);
}

// The two triangles of a unit quad, as corner offsets.
fn quad_corner(i: u32) -> vec2<f32> {
  let corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
    vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0));
  return corners[i];
}

struct SoilOut {
  @builtin(position) clip: vec4<f32>,
  @location(0) color: vec4<f32>,
};

@vertex
fn soil_vs(@builtin(vertex_index) vi: u32) -> SoilOut {
  let cell = vi / 6u;
  let corner = quad_corner(vi % 6u);
  let cx = f32(cell % view.soil_width);
  let cy = f32(cell / view.soil_width);

  var s = sand[cell];
  var l = silt[cell];
  var c = clay[cell];
  let total = s + l + c;

  var color = vec3<f32>(0.0);
  if (total > 0.001) {
    let inv = 1.0 / total;
    s = s * inv;
    l = l * inv;
    c = c * inv;
    let sand_color = vec3<f32>(219.0 / 255.0, 193.0 / 255.0, 44.0 / 255.0);
    let silt_color = vec3<f32>(119.0 / 255.0, 143.0 / 255.0, 40.0 / 255.0);
    let clay_color = vec3<f32>(219.0 / 255.0, 41.0 / 255.0, 23.0 / 255.0);
    color = sand_color * s + silt_color * l + clay_color * c;
  }

  var out: SoilOut;
  out.clip = to_clip((vec2<f32>(cx, cy) + corner) * view.soil_cell_size);
  out.color = vec4<f32>(color, 1.0);
  return out;
}

@fragment
fn soil_fs(in: SoilOut) -> @location(0) vec4<f32> {
  return in.color;
}

struct ParticleOut {
  @builtin(position) clip: vec4<f32>,
  @location(0) color: vec4<f32>,
  // -1..1 across the quad, so the fragment can cut a disc out of it.
  @location(1) local: vec2<f32>,
};

// deep blue (0) -> teal (0.5) -> white (1.0)
fn evap_prob_to_color(raw: f32) -> vec3<f32> {
  let p = clamp(raw, 0.0, 1.0);
  if (p < 0.5) {
    let t = p * 2.0;
    return vec3<f32>(t * 68.0, t * 200.0, 140.0 + t * 115.0) / 255.0;
  }
  let t = (p - 0.5) * 2.0;
  return vec3<f32>(68.0 + t * 187.0, 200.0 + t * 55.0, 255.0) / 255.0;
}

@vertex
fn particle_vs(@builtin(vertex_index) vi: u32) -> ParticleOut {
  let id = vi / 6u;
  let corner = quad_corner(vi % 6u) * 2.0 - vec2<f32>(1.0, 1.0);
  let centre = vec2<f32>(pos[id * 2u], pos[id * 2u + 1u]);

  var radius = view.particle_radius;
  var color = vec4<f32>(1.0, 1.0, 1.0, 1.0);
  if (state[id] == 1u) {
    // 0x40CCCCCC in the C++ renderer's packed ABGR.
    color = vec4<f32>(0.8, 0.8, 0.8, 0.25);
    radius = radius * 0.5;
  } else if (view.debug_evap == 1u) {
    color = vec4<f32>(evap_prob_to_color(evap_prob[id]), 1.0);
  }

  var out: ParticleOut;
  out.clip = to_clip(centre + corner * radius);
  out.color = color;
  out.local = corner;
  return out;
}

@fragment
fn particle_fs(in: ParticleOut) -> @location(0) vec4<f32> {
  if (dot(in.local, in.local) > 1.0) {
    discard;
  }
  return in.color;
}
"#;

/// Pipelines and layout, built once.
pub struct Pipelines {
  layout: wgpu::BindGroupLayout,
  soil: wgpu::RenderPipeline,
  particles: wgpu::RenderPipeline,
}

impl Pipelines {
  pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
      label: Some("alife sim shaders"),
      source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });

    let storage = |binding: u32| wgpu::BindGroupLayoutEntry {
      binding,
      visibility: wgpu::ShaderStages::VERTEX,
      ty: wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only: true },
        has_dynamic_offset: false,
        min_binding_size: None,
      },
      count: None,
    };
    let mut entries = vec![wgpu::BindGroupLayoutEntry {
      binding: 0,
      visibility: wgpu::ShaderStages::VERTEX,
      ty: wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Uniform,
        has_dynamic_offset: false,
        min_binding_size: None,
      },
      count: None,
    }];
    entries.extend((1..=6).map(storage));

    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
      label: Some("alife sim bindings"),
      entries: &entries,
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
      label: None,
      bind_group_layouts: &[Some(&layout)],
      immediate_size: 0,
    });

    let make = |name: &str, vs: &str, fs: &str| {
      device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(name),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
          module: &shader,
          entry_point: Some(vs),
          compilation_options: Default::default(),
          // No vertex buffers: CubeCL's buffers have no VERTEX usage, so the
          // shaders read them as storage and index by vertex id.
          buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
          module: &shader,
          entry_point: Some(fs),
          compilation_options: Default::default(),
          targets: &[Some(wgpu::ColorTargetState {
            format: target_format,
            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
            write_mask: wgpu::ColorWrites::ALL,
          })],
        }),
        multiview_mask: None,
        cache: None,
      })
    };

    Self {
      soil: make("alife soil", "soil_vs", "soil_fs"),
      particles: make("alife particles", "particle_vs", "particle_fs"),
      layout,
    }
  }
}

/// What `prepare` hands to `paint`.
struct Frame {
  bind_group: wgpu::BindGroup,
  soil_vertices: u32,
  particle_vertices: u32,
}

/// One frame's worth of everything the callback needs.
pub struct SimCallback {
  pub pipelines: Arc<Pipelines>,
  pub client: ComputeClient<WgpuRuntime>,
  pub view: ViewUniform,
  pub sph: alife_sim::particles::SphDevice,
  pub soil: alife_sim::soil::SoilDevice,
  pub soil_cells: u32,
}

impl CallbackTrait for SimCallback {
  fn prepare(
    &self,
    device: &wgpu::Device,
    _queue: &wgpu::Queue,
    _screen: &ScreenDescriptor,
    _encoder: &mut wgpu::CommandEncoder,
    resources: &mut CallbackResources,
  ) -> Vec<wgpu::CommandBuffer> {
    let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("alife view"),
      contents: bytemuck::bytes_of(&self.view),
      usage: wgpu::BufferUsages::UNIFORM,
    });

    // Each handle's window into its pooled buffer, resolved fresh: the sim
    // swaps and reallocates handles between frames.
    let bound = [
      shared_buffer(&self.client, &self.soil.sand_density),
      shared_buffer(&self.client, &self.soil.silt_density),
      shared_buffer(&self.client, &self.soil.clay_density),
      shared_buffer(&self.client, &self.sph.pos),
      shared_buffer(&self.client, &self.sph.state),
      shared_buffer(&self.client, &self.sph.evap_prob),
    ];

    let mut entries = vec![wgpu::BindGroupEntry {
      binding: 0,
      resource: uniform.as_entire_binding(),
    }];
    for (i, buffer) in bound.iter().enumerate() {
      entries.push(wgpu::BindGroupEntry {
        binding: 1 + i as u32,
        resource: buffer.binding(),
      });
    }

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: Some("alife sim frame"),
      layout: &self.pipelines.layout,
      entries: &entries,
    });

    resources.insert(Frame {
      bind_group,
      soil_vertices: self.soil_cells * 6,
      particle_vertices: self.view.particle_count * 6,
    });
    Vec::new()
  }

  fn paint(
    &self,
    _info: eframe::egui::epaint::PaintCallbackInfo,
    render_pass: &mut wgpu::RenderPass<'static>,
    resources: &CallbackResources,
  ) {
    let Some(frame) = resources.get::<Frame>() else {
      return;
    };
    render_pass.set_bind_group(0, &frame.bind_group, &[]);
    render_pass.set_pipeline(&self.pipelines.soil);
    render_pass.draw(0..frame.soil_vertices, 0..1);
    render_pass.set_pipeline(&self.pipelines.particles);
    render_pass.draw(0..frame.particle_vertices, 0..1);
  }
}

#[cfg(test)]
mod tests {
  use super::Pipelines;

  /// Build the shaders and pipelines on a headless device.
  ///
  /// wgpu validates the WGSL when the module is created, and the bind group
  /// layout and entry points when the pipeline is. That is the whole of what
  /// can be checked without a window, and the window is the user's to open.
  #[test]
  fn shaders_and_pipelines_build() {
    let Ok(setup) = alife_sim::wgpu_backend::headless_setup(wgpu::Backends::PRIMARY, None) else {
      eprintln!("skipping: no wgpu adapter on this box");
      return;
    };
    let scope = setup.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let _pipelines = Pipelines::new(&setup.device, wgpu::TextureFormat::Bgra8UnormSrgb);
    let error = pollster::block_on(scope.pop());
    assert!(error.is_none(), "{error:?}");
  }
}
