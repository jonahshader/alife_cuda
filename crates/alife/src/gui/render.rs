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

use alife_sim::wgpu_backend::{SharedBuffer, shared_buffer};
use cubecl::prelude::ComputeClient;
use cubecl_wgpu::WgpuRuntime;
use eframe::egui_wgpu::{CallbackResources, CallbackTrait, ScreenDescriptor};

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
@group(0) @binding(7) var<storage, read> part_type: array<u32>;

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

// Body particles are coloured by the part type their limb carries, so a
// plant reads as a plant without looking anything up: root brown, stem
// green-brown, leaf green, seed yellow. The reserved codes stay grey.
fn part_type_to_color(part: u32) -> vec3<f32> {
  switch part {
    case 1u: { return vec3<f32>(0.42, 0.28, 0.14); }
    case 2u: { return vec3<f32>(0.36, 0.42, 0.18); }
    case 3u: { return vec3<f32>(0.20, 0.68, 0.24); }
    case 4u: { return vec3<f32>(0.95, 0.83, 0.25); }
    default: { return vec3<f32>(0.6, 0.6, 0.6); }
  }
}

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
  let kind = state[id];
  if (kind == 1u) {
    // 0x40CCCCCC in the C++ renderer's packed ABGR.
    color = vec4<f32>(0.8, 0.8, 0.8, 0.25);
    radius = radius * 0.5;
  } else if (kind == 2u) {
    color = vec4<f32>(part_type_to_color(part_type[id]), 1.0);
    radius = radius * 1.5;
  } else if (view.debug_evap == 1u) {
    color = vec4<f32>(evap_prob_to_color(evap_prob[id]), 1.0);
  }

  var out: ParticleOut;
  out.clip = to_clip(centre + corner * radius);
  out.color = color;
  // An unallocated body slot is parked outside the world and is not a
  // particle at all: send its corners past the disc cutoff so every fragment
  // is discarded.
  if (kind == 3u) {
    out.local = corner * 4.0;
  } else {
    out.local = corner;
  }
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
    entries.extend((1..=7).map(storage));

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

/// What `prepare` hands to `paint`, and what it keeps between frames.
///
/// The uniform buffer and the bind group outlive a frame. The bind group is
/// rebuilt only when a bound window moves: CubeCL hands out slices of pooled
/// buffers, and the sim swaps `vel` and can reallocate between frames, so the
/// binding a handle resolves to is not stable — but it is also not usually
/// different, and rebuilding it unconditionally made egui's device do that
/// work every frame.
struct Frame {
  uniform: wgpu::Buffer,
  bindings: Vec<BoundWindow>,
  bind_group: wgpu::BindGroup,
  soil_vertices: u32,
  particle_vertices: u32,
}

/// The identity of one bound storage window: which buffer, and where in it.
#[derive(PartialEq, Eq)]
struct BoundWindow {
  buffer: wgpu::Buffer,
  offset: u64,
  size: u64,
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
    queue: &wgpu::Queue,
    _screen: &ScreenDescriptor,
    _encoder: &mut wgpu::CommandEncoder,
    resources: &mut CallbackResources,
  ) -> Vec<wgpu::CommandBuffer> {
    // Each handle's window into its pooled buffer, resolved fresh: the sim
    // swaps and reallocates handles between frames.
    let bound: Vec<SharedBuffer> = [
      &self.soil.sand_density,
      &self.soil.silt_density,
      &self.soil.clay_density,
      &self.sph.pos,
      &self.sph.state,
      &self.sph.evap_prob,
      &self.sph.part_type,
    ]
    .into_iter()
    .map(|handle| shared_buffer(&self.client, handle))
    .collect();
    let bindings: Vec<BoundWindow> = bound
      .iter()
      .map(|b| BoundWindow {
        buffer: b.buffer.clone(),
        offset: b.offset,
        size: b.size,
      })
      .collect();

    let uniform = match resources.get::<Frame>() {
      Some(frame) => frame.uniform.clone(),
      None => device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("alife view"),
        size: size_of::<ViewUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
      }),
    };
    queue.write_buffer(&uniform, 0, bytemuck::bytes_of(&self.view));

    let reusable = resources
      .get::<Frame>()
      .filter(|frame| frame.bindings == bindings);
    let bind_group = match reusable {
      Some(frame) => frame.bind_group.clone(),
      None => {
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
        device.create_bind_group(&wgpu::BindGroupDescriptor {
          label: Some("alife sim bind group"),
          layout: &self.pipelines.layout,
          entries: &entries,
        })
      }
    };

    resources.insert(Frame {
      uniform,
      bindings,
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
  use super::{Frame, Pipelines, SimCallback, ViewUniform};
  use alife_sim::SimParams;
  use alife_sim::particles::SphDevice;
  use alife_sim::sim::Sim;
  use alife_sim::wgpu_backend::{client_on, headless_setup};
  use cubecl_wgpu::WgpuSetup;
  use eframe::egui_wgpu::{CallbackResources, CallbackTrait, ScreenDescriptor};

  fn setup() -> Option<WgpuSetup> {
    match headless_setup(wgpu::Backends::PRIMARY, None) {
      Ok(setup) => Some(setup),
      Err(_) => {
        eprintln!("skipping: no wgpu adapter on this box");
        None
      }
    }
  }

  /// Build the shaders and pipelines on a headless device.
  ///
  /// wgpu validates the WGSL when the module is created, and the bind group
  /// layout and entry points when the pipeline is. That is the whole of what
  /// can be checked without a window, and the window is the user's to open.
  #[test]
  fn shaders_and_pipelines_build() {
    let Some(setup) = setup() else { return };
    let scope = setup.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let _pipelines = Pipelines::new(&setup.device, wgpu::TextureFormat::Bgra8UnormSrgb);
    let error = pollster::block_on(scope.pop());
    assert!(error.is_none(), "{error:?}");
  }

  /// `prepare` is the only part of the render path a test can reach — `paint`
  /// needs a render pass, which needs a surface. This drives it across steps,
  /// which is where the reuse it now does has to hold: the uniform is written
  /// rather than recreated, and the bind group survives as long as every
  /// bound window does.
  #[test]
  fn prepare_reuses_the_uniform_and_the_bind_group_across_steps() {
    let Some(setup) = setup() else { return };
    let (_device, client) = client_on(&setup);

    // A small world: this is about buffer identity, not about the sim.
    let params = SimParams {
      smoothing_radius: 2.0,
      particles_per_cell: 2,
      ..SimParams::default()
    };
    let mut sim = Sim::new(client, params, 42, None);
    let pipelines = std::sync::Arc::new(Pipelines::new(
      &setup.device,
      wgpu::TextureFormat::Bgra8UnormSrgb,
    ));

    let callback = |sim: &Sim<cubecl_wgpu::WgpuRuntime>, sph: SphDevice| SimCallback {
      pipelines: pipelines.clone(),
      client: sim.client().clone(),
      view: ViewUniform {
        scale: [1.0, 1.0],
        offset: [0.0, 0.0],
        soil_cell_size: sim.geometry().soil_cell_size,
        particle_radius: 0.01,
        _pad: [0.0; 2],
        soil_width: sim.geometry().soil_width as u32,
        soil_height: sim.geometry().soil_height as u32,
        particle_count: sim.geometry().num_particles as u32,
        debug_evap: 0,
      },
      sph,
      soil: sim.device_soil().clone(),
      soil_cells: sim.geometry().num_soil_cells() as u32,
    };

    let screen = ScreenDescriptor {
      size_in_pixels: [64, 64],
      pixels_per_point: 1.0,
    };
    let mut resources = CallbackResources::default();
    let prepare =
      |sim: &Sim<cubecl_wgpu::WgpuRuntime>, sph: SphDevice, resources: &mut CallbackResources| {
        let mut encoder = setup
          .device
          .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let scope = setup.device.push_error_scope(wgpu::ErrorFilter::Validation);
        callback(sim, sph).prepare(
          &setup.device,
          &setup.queue,
          &screen,
          &mut encoder,
          resources,
        );
        let error = pollster::block_on(scope.pop());
        assert!(error.is_none(), "{error:?}");
      };

    prepare(&sim, sim.device_particles().clone(), &mut resources);
    let first = resources.get::<Frame>().expect("prepare stored a frame");
    let (uniform, bind_group) = (first.uniform.clone(), first.bind_group.clone());

    for _ in 0..3 {
      sim.step();
      prepare(&sim, sim.device_particles().clone(), &mut resources);
      let frame = resources.get::<Frame>().unwrap();
      assert_eq!(frame.uniform, uniform, "the uniform buffer was recreated");
      assert_eq!(
        frame.bind_group, bind_group,
        "the bind group was rebuilt although nothing it binds moved"
      );
    }

    // And the other way: a bound handle pointing somewhere else has to be
    // noticed. `pos` and `ppos` are two windows into the pool, so swapping
    // them is the cheapest stand-in for the reallocation this guards against.
    let mut moved = sim.device_particles().clone();
    std::mem::swap(&mut moved.pos, &mut moved.ppos);
    prepare(&sim, moved, &mut resources);
    let frame = resources.get::<Frame>().unwrap();
    assert_eq!(frame.uniform, uniform, "the uniform buffer was recreated");
    assert_ne!(
      frame.bind_group, bind_group,
      "a bound window moved and the bind group was kept anyway"
    );
  }
}
