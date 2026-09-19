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

pub fn run(params: SimParams, initial: Option<InitialState>, requested: RuntimeKind) -> Result<()> {
  if requested != RuntimeKind::Wgpu {
    // Nothing stops the sim from running on another backend, but then the
    // renderer would have to copy every buffer through the host every frame,
    // which is exactly what this port set out to remove.
    tracing::warn!(
      "the GUI runs the sim on the wgpu runtime so the renderer can read its buffers directly; \
       ignoring --runtime {}",
      requested.name()
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
    Box::new(move |cc| Ok(Box::new(SimApp::new(cc, params, initial)?))),
  )
  .map_err(|err| anyhow::anyhow!("{err}"))
}

struct SimApp {
  sim: Sim<cubecl_wgpu::WgpuRuntime>,
  setup: WgpuSetup,
  pipelines: Arc<Pipelines>,
  params: SimParams,
  seed: u64,

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
    let sim = Sim::new(client, params, seed, initial);
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
      "step {}  ·  {} particles",
      self.sim.step_count(),
      self.sim.geometry().num_particles,
    ));

    ui.separator();
    ui.heading("Fluid");
    retune |= ui
      .add(egui::Slider::new(&mut p.dt, 0.0..=0.1).text("dt"))
      .changed();
    retune |= ui
      .add(egui::Slider::new(&mut p.dt_predict, 0.0..=0.1).text("dt_predict"))
      .changed();
    retune |= ui
      .add(egui::Slider::new(&mut p.gravity, -30.0..=0.0).text("gravity"))
      .changed();
    retune |= ui
      .add(egui::Slider::new(&mut p.collision_damping, 0.0..=1.0).text("collision_damping"))
      .changed();
    // Changing the smoothing radius resizes the neighbour grid, so it cannot
    // be a live tweak: the C++ slider did exactly that and left the device
    // grid at the old size while kernels launched with the new one.
    reshape |= ui
      .add(egui::Slider::new(&mut p.smoothing_radius, 0.001..=0.5).text("smoothing_radius"))
      .drag_stopped();
    retune |= ui
      .add(egui::Slider::new(&mut p.target_density, 0.0..=400.0).text("target_density"))
      .changed();
    retune |= ui
      .add(egui::Slider::new(&mut p.pressure_mult, 0.0..=1200.0).text("pressure_mult"))
      .changed();
    retune |= ui
      .add(egui::Slider::new(&mut p.near_pressure_mult, 0.0..=100.0).text("near_pressure_mult"))
      .changed();
    retune |= ui
      .add(egui::Slider::new(&mut p.viscosity_strength, 0.0..=10.0).text("viscosity_strength"))
      .changed();
    retune |= ui
      .add(egui::Slider::new(&mut p.capillary_mult, 0.0..=5.0).text("capillary_mult"))
      .changed();
    reshape |= ui
      .add(egui::Slider::new(&mut p.particles_per_cell, 1..=32).text("particles_per_cell"))
      .drag_stopped();
    reshape |= ui
      .add(
        egui::Slider::new(&mut p.max_particles_per_cell, 1..=1024).text("max_particles_per_cell"),
      )
      .drag_stopped();

    ui.separator();
    ui.heading("Evaporation / condensation");
    retune |= ui
      .add(egui::Slider::new(&mut p.evap_rate, 0.0..=0.1).text("evap_rate"))
      .changed();
    retune |= ui
      .add(egui::Slider::new(&mut p.condense_rate, 0.0..=0.1).text("condense_rate"))
      .changed();
    retune |= ui
      .add(egui::Slider::new(&mut p.vapor_buoyancy, 0.0..=20.0).text("vapor_buoyancy"))
      .changed();
    retune |= ui
      .add(egui::Slider::new(&mut p.vapor_drift, 0.0..=5.0).text("vapor_drift"))
      .changed();
    retune |= ui
      .add(egui::Slider::new(&mut p.condense_altitude_power, 0.5..=5.0).text("condense_alt_power"))
      .changed();

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
        if response.hovered() && scroll != 0.0 {
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
          particle_count: geom.num_particles as u32,
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
