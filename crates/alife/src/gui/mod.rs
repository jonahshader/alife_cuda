//! The windowed front end: an eframe window with the sim drawn into it and an
//! egui panel carrying the controls the ImGui `FluidSoil` screen had.
//!
//! The sim runs on the wgpu runtime, on the very `wgpu::Device` eframe created,
//! so the renderer binds CubeCL's buffers directly and nothing round-trips
//! through the host.

mod frame_stats;
mod profiler;
mod render;
mod rolling;

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

/// Density overlay samples per metre — the C++ screen's `pixels_per_meter`,
/// whose density texture was one texel per pixel of the world.
const DENSITY_SAMPLES_PER_METRE: f32 = 10.0;

/// The density the overlay draws as white; above it a sample is blue
/// (`alife_sim::kernels::density_grid`). The C++ screen's display scale.
const DENSITY_DISPLAY_SCALE: f32 = 300.0;

/// The mouse tool's starting radius (metres) and strength (velocity added
/// per step), from the C++ `FluidSoil` screen.
const TOOL_RADIUS: f32 = 2.0;
const TOOL_STRENGTH: f32 = 0.15;

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

  // eframe's default device request takes `wgpu::Limits::default()`, which
  // caps a compute stage at eight storage buffers; the kernels bind up to
  // sixteen. Ask for the device the sim needs, since the sim runs on it.
  let mut wgpu_options = egui_wgpu::WgpuConfiguration::default();
  if let egui_wgpu::WgpuSetup::CreateNew(create) = &mut wgpu_options.wgpu_setup {
    create.device_descriptor = Arc::new(alife_sim::wgpu_backend::device_descriptor);
  }
  let options = eframe::NativeOptions {
    viewport: egui::ViewportBuilder::default()
      .with_title("ALife")
      .with_inner_size([1600.0, 900.0]),
    wgpu_options,
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

/// One point of the population plot: the counts at a step.
#[derive(Clone, Copy)]
struct PopSample {
  step: u32,
  organisms: usize,
  births: u64,
  deaths: u64,
}

/// The population over the run, for the plot under the world.
///
/// Every count here is host-side already — the life tick owns `alive` and
/// the counters — so a sample costs nothing on the device. The series is
/// kept to a bounded number of points: when it fills, every other point is
/// dropped and the sampling interval doubles, so a run of any length plots
/// at the same cost and the same apparent resolution.
struct PopHistory {
  samples: Vec<PopSample>,
  /// Steps between samples; doubles as the series thins.
  every: u32,
}

impl PopHistory {
  const CAPACITY: usize = 2048;

  fn new() -> Self {
    Self {
      samples: Vec::with_capacity(Self::CAPACITY),
      every: 1,
    }
  }

  /// Record the current counts if this step is due one.
  fn record(&mut self, sim: &Sim<cubecl_wgpu::WgpuRuntime>) {
    let step = sim.step_count();
    let due = match self.samples.last() {
      Some(last) => step >= last.step + self.every,
      None => true,
    };
    if !due {
      return;
    }
    let counters = sim.counters();
    self.samples.push(PopSample {
      step,
      organisms: sim.organism_count(),
      births: counters.births,
      deaths: counters.deaths,
    });
    if self.samples.len() >= Self::CAPACITY {
      let mut keep = 0;
      self.samples.retain(|_| {
        keep += 1;
        keep % 2 == 1
      });
      self.every *= 2;
    }
  }

  fn series(&self, y: impl Fn(&PopSample) -> f64) -> Vec<[f64; 2]> {
    self
      .samples
      .iter()
      .map(|s| [f64::from(s.step), y(s)])
      .collect()
  }
}

struct SimApp {
  sim: Sim<cubecl_wgpu::WgpuRuntime>,
  setup: WgpuSetup,
  pipelines: Arc<Pipelines>,
  params: SimParams,
  seed: u64,

  /// Founders seeded on every rebuild, so Reset gives back the same world.
  /// The panel edits this; it takes effect on the next Reset.
  founders: usize,
  history: PopHistory,

  running: bool,
  step_once: bool,
  steps_per_frame: u32,
  debug_evap: bool,
  /// Whether the density overlay is computed and drawn.
  show_density: bool,

  /// The mouse tool: left button attracts, right repels, at the cursor.
  tool_radius: f32,
  tool_strength: f32,
  /// Where the tool acts and with what signed strength, while a button is
  /// held on the world. Set while the world is laid out, which is after this
  /// frame's steps, so the steps of the next frame apply it — a frame late,
  /// which is invisible, and it keeps the steps ahead of the layout that
  /// reads their results.
  tool_target: Option<(glam::Vec2, f32)>,

  /// World point at the centre of the view, and metres across the viewport.
  camera_centre: egui::Vec2,
  camera_height: f32,

  profiler: profiler::Profiler,
  frame_stats: frame_stats::FrameStats,
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
    let mut history = PopHistory::new();
    history.record(&sim);
    Ok(Self {
      sim,
      setup,
      pipelines,
      params,
      seed,
      founders,
      history,
      running: true,
      step_once: false,
      steps_per_frame: 1,
      debug_evap: false,
      show_density: false,
      tool_radius: TOOL_RADIUS,
      tool_strength: TOOL_STRENGTH,
      tool_target: None,
      camera_centre: egui::vec2(bounds.x * 0.5, bounds.y * 0.5),
      camera_height: bounds.y,
      profiler: profiler::Profiler::new(),
      frame_stats: frame_stats::FrameStats::new(),
    })
  }

  /// Rebuild the world. Needed for any parameter that changes a buffer's shape
  /// — the grid dimensions are compiled into the kernels.
  fn rebuild(&mut self) {
    let (_device, client) = client_on(&self.setup);
    self.sim = Sim::new(client, self.params, self.seed, None);
    alife_sim::bodies::spawn_founders(&mut self.sim, self.founders);
    self.history = PopHistory::new();
    self.history.record(&self.sim);
  }

  fn population_plot(&self, ui: &mut egui::Ui) {
    use egui_plot::{Legend, Line, Plot};
    let h = &self.history;
    Plot::new("population")
      .height(150.0)
      .legend(Legend::default())
      .include_y(0.0)
      .allow_drag(false)
      .allow_zoom(false)
      .allow_scroll(false)
      .x_axis_label("step")
      .show(ui, |plot| {
        plot.line(Line::new("organisms", h.series(|s| s.organisms as f64)));
        plot.line(Line::new("births", h.series(|s| s.births as f64)));
        plot.line(Line::new("deaths", h.series(|s| s.deaths as f64)));
      });
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
    ui.horizontal(|ui| {
      let max = usize::try_from(p.max_organisms).unwrap_or(0);
      ui.add(egui::DragValue::new(&mut self.founders).range(0..=max));
      ui.label("founders (on Reset)");
    });
    // Nothing here reads the particle buffers back: counting the vapor would
    // cost a device round trip every frame. The organism counts are host-side.
    let counters = self.sim.counters();
    ui.label(format!(
      "step {}  ·  {} particle slots\n{} organisms  ·  {} births  ·  {} deaths",
      self.sim.step_count(),
      self.sim.geometry().num_particles,
      self.sim.organism_count(),
      counters.births,
      counters.deaths,
    ));
    self.frame_stats.show(ui);

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
    ui.checkbox(&mut self.show_density, "Density grid")
      .on_hover_text(
        "SPH density, white at 300. Blue: denser than that. Red: a cell holding more than \
         max_particles_per_cell, whose excess every neighbour sum drops.",
      );
    ui.label(format!(
      "Tool: left drag attracts, right repels, middle pans.\n\
       radius {:.2} m (+/-), strength {:.3} ([/])",
      self.tool_radius, self.tool_strength
    ));
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

/// Esc closes the window, as it stopped the C++ game — unless a text field
/// has the keyboard, where Esc means "stop editing". Checked before any
/// widget runs, while the field still holds the focus it gives up on Esc.
fn quit_on_escape(ctx: &egui::Context) {
  if ctx.input(|i| i.key_pressed(egui::Key::Escape)) && !ctx.text_edit_focused() {
    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
  }
}

impl SimApp {
  /// The tool's keys, as the C++ screen had them: `+`/`=` and `-` scale the
  /// radius, `]` and `[` the strength. Not while egui has the keyboard (a
  /// focused slider or field), and not with Ctrl/Cmd, which is egui's own
  /// zoom on the same keys.
  fn tool_keys(&mut self, ui: &egui::Ui) {
    if ui.ctx().egui_wants_keyboard_input() {
      return;
    }
    let (grow, shrink, stronger, weaker) = ui.input(|i| {
      if i.modifiers.command {
        return (0, 0, 0, 0);
      }
      (
        // Shift+= arrives as Plus, the keypad's + as Plus too.
        i.num_presses(egui::Key::Plus) + i.num_presses(egui::Key::Equals),
        i.num_presses(egui::Key::Minus),
        i.num_presses(egui::Key::CloseBracket),
        i.num_presses(egui::Key::OpenBracket),
      )
    });
    self.tool_radius *= 1.25f32.powi(grow as i32) * 0.8f32.powi(shrink as i32);
    // Bounded where the C++ was not: past half the world's width the disc
    // overlaps itself across the seam and pulls a particle through two of
    // its images, and the kernel is one thread walking every cell in reach.
    let bounds = self.sim.geometry().bounds;
    self.tool_radius = self.tool_radius.clamp(0.01, 0.5 * bounds.x);
    self.tool_strength *= 1.25f32.powi(stronger as i32) * 0.8f32.powi(weaker as i32);
  }

  /// The world point under a screen point, for a view of `rect`.
  fn screen_to_world(&self, rect: egui::Rect, point: egui::Pos2) -> glam::Vec2 {
    let metres_per_point = self.camera_height / rect.height().max(1.0);
    let from_centre = point - rect.center();
    glam::Vec2::new(
      self.camera_centre.x + from_centre.x * metres_per_point,
      // Screen y grows downward, world y grows upward.
      self.camera_centre.y - from_centre.y * metres_per_point,
    )
  }

  /// The tool's circle and its strength label, as the C++ screen drew them:
  /// fainter while idle, stronger while a button is held.
  fn draw_tool(&self, painter: &egui::Painter, rect: egui::Rect, at: egui::Pos2) {
    let points_per_metre = rect.height().max(1.0) / self.camera_height;
    let radius = self.tool_radius * points_per_metre;
    let alpha = if self.tool_target.is_some() {
      0.25
    } else {
      0.1
    };
    painter.circle_filled(
      at,
      radius,
      egui::Rgba::from_rgba_unmultiplied(0.5, 0.5, 0.8, alpha),
    );

    // The C++ fitted the label to the radius's width and drew it black at
    // half alpha; the label is white here because this view's background is
    // near-black, and its size is bounded so a zoomed-in view does not
    // rasterise glyphs hundreds of points tall.
    let text = format!("Strength: {:.6}", self.tool_strength);
    let colour = egui::Color32::from_white_alpha(128);
    let nominal = 14.0;
    let width = painter
      .layout_no_wrap(text.clone(), egui::FontId::proportional(nominal), colour)
      .size()
      .x
      .max(1.0);
    let size = (nominal * radius / width).clamp(6.0, 64.0);
    painter.text(
      at,
      egui::Align2::CENTER_BOTTOM,
      text,
      egui::FontId::proportional(size),
      colour,
    );
  }
}

impl eframe::App for SimApp {
  fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
    quit_on_escape(ui.ctx());
    self.tool_keys(ui);

    if self.running || self.step_once {
      let steps = if self.step_once {
        1
      } else {
        self.steps_per_frame
      };
      for _ in 0..steps {
        self.sim.step();
        // After every step, against the grid it built — where the C++
        // `FluidSoil::update` applied it.
        if let Some((at, strength)) = self.tool_target {
          self.sim.attract(at, strength, self.tool_radius);
        }
      }
      // The renderer reads the same buffers the kernels just wrote, on the
      // same queue; syncing here keeps the frame's contents well-defined.
      self.sim.sync();
      self.step_once = false;
      self.history.record(&self.sim);
    }
    self.frame_stats.tick(self.sim.step_count());

    // Every frame while it is on, paused or not: a frame is one dispatch
    // over a few tens of thousands of samples.
    let density = self.show_density.then(|| {
      let bounds = self.sim.geometry().bounds;
      self.sim.density_grid(
        ((bounds.x * DENSITY_SAMPLES_PER_METRE).round() as u32).max(1),
        ((bounds.y * DENSITY_SAMPLES_PER_METRE).round() as u32).max(1),
        DENSITY_DISPLAY_SCALE,
      )
    });

    egui::Panel::right("controls")
      .default_size(320.0)
      .show(ui, |ui| {
        egui::ScrollArea::vertical().show(ui, |ui| self.controls(ui));
      });

    egui::Panel::bottom("population").show(ui, |ui| self.population_plot(ui));

    egui::CentralPanel::no_frame()
      .frame(egui::Frame::NONE.fill(egui::Color32::from_rgb(8, 10, 20)))
      .show(ui, |ui| {
        let (rect, response) =
          ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());

        // Pan with a middle drag, zoom with the wheel, both in world units.
        // The left and right buttons are the tool's.
        let metres_per_point = self.camera_height / rect.height().max(1.0);
        if response.dragged_by(egui::PointerButton::Middle) {
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

        // Only a press that began on the world counts, so a drag off a
        // slider never pulls the water, and egui's own widgets keep the
        // pointer while they hold it. Right wins when both are down, as in
        // the C++ (`repelling ? -strength : strength`).
        let (primary, secondary, latest) = ui.input(|i| {
          (
            i.pointer.primary_down(),
            i.pointer.secondary_down(),
            i.pointer.latest_pos(),
          )
        });
        let held = response.is_pointer_button_down_on() && (primary || secondary);
        let cursor = if held { latest } else { response.hover_pos() };
        self.tool_target = match cursor {
          Some(at) if held => {
            let sign = if secondary { -1.0 } else { 1.0 };
            Some((self.screen_to_world(rect, at), sign * self.tool_strength))
          }
          _ => None,
        };

        let half_height = self.camera_height * 0.5;
        let half_width = half_height * (rect.width() / rect.height().max(1.0));
        let geom = *self.sim.geometry();
        let body_cfg = self.sim.body_cfg();
        let view = ViewUniform {
          scale: [1.0 / half_width, 1.0 / half_height],
          offset: [
            -self.camera_centre.x / half_width,
            -self.camera_centre.y / half_height,
          ],
          soil_cell_size: geom.soil_cell_size,
          particle_radius: self.params.smoothing_radius * 0.1,
          world_width: geom.bounds.x,
          _pad: 0.0,
          soil_width: geom.soil_width as u32,
          soil_height: geom.soil_height as u32,
          // Only the live prefix: unclaimed body slots are `Free` and would
          // be a discarded quad each, per frame.
          particle_count: self.sim.live_particles().0 as u32,
          debug_evap: u32::from(self.debug_evap),
          max_limbs: body_cfg.max_limbs,
          per_limb: body_cfg.max_particles_per_limb,
          segment_count: body_cfg.particle_map_len() as u32,
          _pad2: 0,
          density_width: density.as_ref().map_or(0, |d| d.width),
          density_height: density.as_ref().map_or(0, |d| d.height),
          density_interval: density.as_ref().map_or(0.0, |d| d.sample_interval),
          _pad3: 0,
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
            bodies: self.sim.bodies().device.clone(),
            limbs: self.sim.population().device.limbs.clone(),
            density,
          },
        ));
        // Over the sim, which the callback paints in this same layer.
        if let Some(at) = cursor {
          self.draw_tool(ui.painter(), rect, at);
        }
      });

    let ctx = ui.ctx().clone();
    self.profiler.show(&ctx, &mut self.sim);

    if self.running {
      ui.ctx().request_repaint();
    }
  }
}
