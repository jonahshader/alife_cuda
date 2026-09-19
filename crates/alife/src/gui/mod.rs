//! The windowed front end: an eframe window with the sim drawn into it and an
//! egui panel carrying the controls the ImGui `FluidSoil` screen had.
//!
//! The sim runs on the wgpu runtime, on the very `wgpu::Device` eframe created,
//! so the renderer binds CubeCL's buffers directly and nothing round-trips
//! through the host.

mod render;

use alife_sim::runtime::RuntimeKind;
use alife_sim::sim::{InitialState, Sim};
use alife_sim::wgpu_backend::client_on;
use alife_sim::{SimParams, dump};
use anyhow::{Context, Result};
use cubecl_wgpu::WgpuSetup;
use eframe::egui;
use eframe::egui_wgpu;
use std::sync::Arc;

use render::{Pipelines, SimCallback, ViewUniform};

// Unqualified so the slider table below fits one parameter per line.
use Effect::{Reshape, Retune};

pub fn run(
  params: SimParams,
  initial: Option<InitialState>,
  requested: Option<RuntimeKind>,
  founders: usize,
) -> Result<()> {
  // Only an explicit `--runtime` is worth a warning; without one there is no
  // choice being overridden, and the GUI never asked what this box can run.
  if let Some(kind) = requested.filter(|kind| *kind != RuntimeKind::Wgpu) {
    // Nothing stops the sim from running on another backend, but then the
    // renderer would have to copy every buffer through the host every frame,
    // which is exactly what this port set out to remove.
    tracing::warn!(
      "the GUI runs the sim on the wgpu runtime so the renderer can read its buffers directly; \
       ignoring --runtime {}",
      kind.name()
    );
  }

  let options = eframe::NativeOptions {
    viewport: egui::ViewportBuilder::default()
      .with_title("ALife")
      .with_inner_size([1600.0, 900.0]),
    ..Default::default()
  };

  // eframe owns the error type; carry ours through a slot instead of unwrapping
  // inside the closure.
  eframe::run_native(
    "alife",
    options,
    Box::new(move |cc| Ok(Box::new(SimApp::new(cc, params, initial, founders)?))),
  )
  .map_err(|err| anyhow::anyhow!("{err}"))
}

/// What moving a parameter costs.
#[derive(Clone, Copy)]
enum Effect {
  /// Re-uploads the params buffer and the world keeps running, so the slider
  /// can act while it is being dragged.
  Retune,
  /// Resizes a buffer or moves a constant compiled into the kernels, so the
  /// world has to be rebuilt — which is why these act on `drag_stopped`
  /// rather than every frame of a drag. The C++ slider rebuilt nothing at
  /// all and left the device grid at the old size while kernels launched
  /// with the new one.
  Reshape,
}

/// Which field a slider moves, and between what.
///
/// A function pointer rather than an offset: the borrow is handed out at the
/// moment of use, so the table itself is a `const`.
enum Field {
  Float {
    at: fn(&mut SimParams) -> &mut f32,
    range: (f32, f32),
  },
  Int {
    at: fn(&mut SimParams) -> &mut i32,
    range: (i32, i32),
  },
}

struct Control {
  label: &'static str,
  field: Field,
  effect: Effect,
}

impl Control {
  const fn float(
    label: &'static str,
    at: fn(&mut SimParams) -> &mut f32,
    lo: f32,
    hi: f32,
    effect: Effect,
  ) -> Self {
    Self {
      label,
      field: Field::Float {
        at,
        range: (lo, hi),
      },
      effect,
    }
  }

  const fn int(
    label: &'static str,
    at: fn(&mut SimParams) -> &mut i32,
    lo: i32,
    hi: i32,
    effect: Effect,
  ) -> Self {
    Self {
      label,
      field: Field::Int {
        at,
        range: (lo, hi),
      },
      effect,
    }
  }

  /// Draw the slider; true when its [`Effect`] should be applied now.
  fn show(&self, ui: &mut egui::Ui, params: &mut SimParams) -> bool {
    let response = match &self.field {
      Field::Float { at, range } => {
        ui.add(egui::Slider::new(at(params), range.0..=range.1).text(self.label))
      }
      Field::Int { at, range } => {
        ui.add(egui::Slider::new(at(params), range.0..=range.1).text(self.label))
      }
    };
    match self.effect {
      Effect::Retune => response.changed(),
      Effect::Reshape => response.drag_stopped(),
    }
  }
}

/// Every parameter slider the panel shows, under the heading it sits below.
/// One line per parameter is the point, so rustfmt is kept off it; adding a
/// parameter to the GUI is adding a line here.
#[rustfmt::skip]
const SLIDERS: &[(&str, &[Control])] = &[
  ("Fluid", &[
    Control::float("dt", |p| &mut p.dt, 0.0, 0.1, Retune),
    Control::float("dt_predict", |p| &mut p.dt_predict, 0.0, 0.1, Retune),
    Control::float("gravity", |p| &mut p.gravity, -30.0, 0.0, Retune),
    Control::float("collision_damping", |p| &mut p.collision_damping, 0.0, 1.0, Retune),
    Control::float("smoothing_radius", |p| &mut p.smoothing_radius, 0.001, 0.5, Reshape),
    Control::float("target_density", |p| &mut p.target_density, 0.0, 400.0, Retune),
    Control::float("pressure_mult", |p| &mut p.pressure_mult, 0.0, 1200.0, Retune),
    Control::float("near_pressure_mult", |p| &mut p.near_pressure_mult, 0.0, 100.0, Retune),
    Control::float("viscosity_strength", |p| &mut p.viscosity_strength, 0.0, 10.0, Retune),
    Control::float("capillary_mult", |p| &mut p.capillary_mult, 0.0, 5.0, Retune),
    Control::int("particles_per_cell", |p| &mut p.particles_per_cell, 1, 32, Reshape),
    Control::int("max_particles_per_cell", |p| &mut p.max_particles_per_cell, 1, 1024, Reshape),
  ]),
  ("Evaporation / condensation", &[
    Control::float("evap_rate", |p| &mut p.evap_rate, 0.0, 0.1, Retune),
    Control::float("condense_rate", |p| &mut p.condense_rate, 0.0, 0.1, Retune),
    Control::float("vapor_buoyancy", |p| &mut p.vapor_buoyancy, 0.0, 20.0, Retune),
    Control::float("vapor_drift", |p| &mut p.vapor_drift, 0.0, 5.0, Retune),
    Control::float("condense_alt_power", |p| &mut p.condense_altitude_power, 0.5, 5.0, Retune),
  ]),
];

struct SimApp {
  sim: Sim<cubecl_wgpu::WgpuRuntime>,
  setup: WgpuSetup,
  pipelines: Arc<Pipelines>,
  params: SimParams,
  seed: u64,

  /// Founders seeded at startup and again on every rebuild, so Reset gives
  /// back the same world.
  founders: usize,

  running: bool,
  step_once: bool,
  steps_per_frame: u32,
  debug_evap: bool,

  /// World point at the centre of the view, and metres across the viewport.
  camera_centre: egui::Vec2,
  camera_height: f32,
}

impl SimApp {
  fn new(
    cc: &eframe::CreationContext<'_>,
    params: SimParams,
    initial: Option<InitialState>,
    founders: usize,
  ) -> Result<Self> {
    let render_state = cc
      .wgpu_render_state
      .as_ref()
      .context("eframe was built without the wgpu backend")?;

    // eframe already made an instance, adapter, device and queue; that is
    // exactly a `WgpuSetup`, so CubeCL adopts it rather than making its own.
    let setup = WgpuSetup {
      instance: render_state.instance.clone(),
      adapter: render_state.adapter.clone(),
      device: render_state.device.clone(),
      queue: render_state.queue.clone(),
      backend: render_state.adapter.get_info().backend,
    };
    let (_device, client) = client_on(&setup);

    let seed = params.resolve_seed();
    let mut sim = Sim::new(client, params, seed, initial);
    alife_sim::bodies::spawn_founders(&mut sim, founders);
    let pipelines = Arc::new(Pipelines::new(
      &render_state.device,
      render_state.target_format,
    ));

    let bounds = sim.geometry().bounds;
    Ok(Self {
      sim,
      setup,
      pipelines,
      params,
      seed,
      founders,
      running: true,
      step_once: false,
      steps_per_frame: 1,
      debug_evap: false,
      camera_centre: egui::vec2(bounds.x * 0.5, bounds.y * 0.5),
      camera_height: bounds.y,
    })
  }

  /// Rebuild the world. Needed for any parameter that changes a buffer's shape
  /// — the grid dimensions are compiled into the kernels.
  fn rebuild(&mut self) {
    let (_device, client) = client_on(&self.setup);
    self.sim = Sim::new(client, self.params, self.seed, None);
    alife_sim::bodies::spawn_founders(&mut self.sim, self.founders);
  }

  fn controls(&mut self, ui: &mut egui::Ui) {
    let mut reshape = false;
    let mut retune = false;
    let p = &mut self.params;

    ui.heading("Simulation");
    ui.horizontal(|ui| {
      if ui
        .button(if self.running { "Pause" } else { "Run" })
        .clicked()
      {
        self.running = !self.running;
      }
      if ui.button("Step").clicked() {
        self.step_once = true;
      }
      if ui.button("Reset").clicked() {
        reshape = true;
      }
    });
    ui.add(egui::Slider::new(&mut self.steps_per_frame, 1..=16).text("steps per frame"));
    // Nothing here reads the particle buffers back: counting the vapor would
    // cost a device round trip every frame.
    ui.label(format!(
      "step {}  ·  {} particle slots  ·  {} organisms",
      self.sim.step_count(),
      self.sim.geometry().num_particles,
      self.sim.organism_count(),
    ));

    for (heading, controls) in SLIDERS {
      ui.separator();
      ui.heading(*heading);
      for control in *controls {
        if control.show(ui, p) {
          match control.effect {
            Retune => retune = true,
            Reshape => reshape = true,
          }
        }
      }
    }

    ui.separator();
    ui.heading("View");
    ui.checkbox(&mut self.debug_evap, "Debug evap colors");
    if ui.button("Frame the world").clicked() {
      let bounds = self.sim.geometry().bounds;
      self.camera_centre = egui::vec2(bounds.x * 0.5, bounds.y * 0.5);
      self.camera_height = bounds.y;
    }
    if ui.button("Dump particles").clicked() {
      let particles = self.sim.read_particles();
      match dump::write("dump.bin", &particles, self.sim.step_count(), self.seed) {
        Ok(()) => tracing::info!("wrote dump.bin"),
        Err(err) => tracing::error!("dump failed: {err}"),
      }
    }

    if reshape {
      self.rebuild();
    } else if retune {
      self.sim.set_params(self.params);
    }
  }
}

impl eframe::App for SimApp {
  fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
    if self.running || self.step_once {
      let steps = if self.step_once {
        1
      } else {
        self.steps_per_frame
      };
      for _ in 0..steps {
        self.sim.step();
      }
      // The renderer reads the same buffers the kernels just wrote, on the
      // same queue; syncing here keeps the frame's contents well-defined.
      self.sim.sync();
      self.step_once = false;
    }

    egui::Panel::right("controls")
      .default_size(320.0)
      .show(ui, |ui| {
        egui::ScrollArea::vertical().show(ui, |ui| self.controls(ui));
      });

    egui::CentralPanel::no_frame()
      .frame(egui::Frame::NONE.fill(egui::Color32::from_rgb(8, 10, 20)))
      .show(ui, |ui| {
        let (rect, response) =
          ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());

        // Pan with a drag, zoom with the wheel, both in world units.
        let metres_per_point = self.camera_height / rect.height().max(1.0);
        if response.dragged() {
          let delta = response.drag_delta();
          self.camera_centre.x -= delta.x * metres_per_point;
          // Screen y grows downward, world y grows upward.
          self.camera_centre.y += delta.y * metres_per_point;
        }
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        // egui zeroes the smoothed delta exactly today; a tolerance keeps a
        // sub-epsilon residual from zooming every frame if that ever changes
        if response.hovered() && scroll.abs() > 1.0e-4 {
          self.camera_height = (self.camera_height * (1.0 - scroll * 0.002)).clamp(0.05, 1.0e4);
        }

        let half_height = self.camera_height * 0.5;
        let half_width = half_height * (rect.width() / rect.height().max(1.0));
        let geom = *self.sim.geometry();
        let view = ViewUniform {
          scale: [1.0 / half_width, 1.0 / half_height],
          offset: [
            -self.camera_centre.x / half_width,
            -self.camera_centre.y / half_height,
          ],
          soil_cell_size: geom.soil_cell_size,
          particle_radius: self.params.smoothing_radius * 0.1,
          _pad: [0.0; 2],
          soil_width: geom.soil_width as u32,
          soil_height: geom.soil_height as u32,
          // Only the live prefix: unclaimed body slots are `Free` and would
          // be a discarded quad each, per frame.
          particle_count: self.sim.live_particles().0 as u32,
          debug_evap: u32::from(self.debug_evap),
        };

        ui.painter().add(egui_wgpu::Callback::new_paint_callback(
          rect,
          SimCallback {
            pipelines: self.pipelines.clone(),
            client: self.sim.client().clone(),
            view,
            sph: self.sim.device_particles().clone(),
            soil: self.sim.device_soil().clone(),
            soil_cells: geom.num_soil_cells() as u32,
          },
        ));
      });

    if self.running {
      ui.ctx().request_repaint();
    }
  }
}
