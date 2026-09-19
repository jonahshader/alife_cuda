//! Runtime selection: one `Sim` per CubeCL backend behind one enum.
//!
//! `Sim<R>` is generic over the runtime because CubeCL's kernels are, and the
//! runtime is a startup choice, so the three instantiations are erased here
//! rather than in every caller.

use cubecl::prelude::*;
use cubecl_cpu::{CpuDevice, CpuRuntime};
use cubecl_cuda::{CudaDevice, CudaRuntime};
use cubecl_wgpu::WgpuRuntime;

use crate::SimParams;
use crate::particles::SphHost;
use crate::sim::{InitialState, Sim};
use crate::soil::SoilGrid;
use crate::timing::KernelTimings;
use crate::world::WorldGeometry;

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum RuntimeKind {
  /// `cubecl-cpu`, LLVM JIT over ~one thread per core. Not a fallback: it
  /// beats software Vulkan and a single-threaded Rust loop by a wide margin
  /// (`docs/perf.md`).
  Cpu,
  Cuda,
  Wgpu,
}

impl RuntimeKind {
  pub fn name(self) -> &'static str {
    match self {
      RuntimeKind::Cpu => "cpu",
      RuntimeKind::Cuda => "cuda",
      RuntimeKind::Wgpu => "wgpu",
    }
  }
}

pub enum AnySim {
  Cpu(Sim<CpuRuntime>),
  Cuda(Sim<CudaRuntime>),
  Wgpu(Sim<WgpuRuntime>),
}

macro_rules! dispatch {
  ($self:expr, $sim:ident => $body:expr) => {
    match $self {
      AnySim::Cpu($sim) => $body,
      AnySim::Cuda($sim) => $body,
      AnySim::Wgpu($sim) => $body,
    }
  };
}

/// How the wgpu runtime picks its adapter.
#[derive(Debug, Clone, Default)]
pub struct WgpuOptions {
  /// Substring of the adapter name, case-insensitive. Without one the most
  /// capable adapter wins (discrete > integrated > virtual > software).
  pub adapter: Option<String>,
}

impl AnySim {
  pub fn new(
    kind: RuntimeKind,
    params: SimParams,
    seed: u64,
    initial: Option<InitialState>,
    wgpu_options: &WgpuOptions,
  ) -> anyhow::Result<Self> {
    Ok(match kind {
      RuntimeKind::Cpu => AnySim::Cpu(Sim::new(
        CpuRuntime::client(&CpuDevice),
        params,
        seed,
        initial,
      )),
      RuntimeKind::Cuda => AnySim::Cuda(Sim::new(
        CudaRuntime::client(&CudaDevice::new(0)),
        params,
        seed,
        initial,
      )),
      RuntimeKind::Wgpu => {
        let setup = crate::wgpu_backend::headless_setup(
          wgpu::Backends::PRIMARY,
          wgpu_options.adapter.as_deref(),
        )?;
        let (_device, client) = crate::wgpu_backend::client_on(&setup);
        AnySim::Wgpu(Sim::new(client, params, seed, initial))
      }
    })
  }

  /// CUDA if this box can run it, else wgpu, else the CPU runtime — keeping
  /// the client the winning backend handed back. Which one that was is
  /// [`AnySim::kind`].
  ///
  /// Creating a client initializes the whole backend, so asking
  /// [`available`] first and then building the same backend again paid that
  /// cost twice. Call [`AnySim::new`] instead when the user named a runtime.
  pub fn new_auto(
    params: SimParams,
    seed: u64,
    initial: Option<InitialState>,
    wgpu_options: &WgpuOptions,
  ) -> anyhow::Result<Self> {
    if let Some(client) = probe(|| CudaRuntime::client(&CudaDevice::new(0))) {
      return Ok(AnySim::Cuda(Sim::new(client, params, seed, initial)));
    }

    // A named adapter is the one case where wgpu not working is an error
    // rather than a reason to try the next backend: the user asked for that
    // device, so running on the CPU instead would be answering a different
    // question.
    match probe(|| {
      crate::wgpu_backend::headless_setup(wgpu::Backends::PRIMARY, wgpu_options.adapter.as_deref())
    }) {
      Some(Ok(setup)) => {
        let (_device, client) = crate::wgpu_backend::client_on(&setup);
        return Ok(AnySim::Wgpu(Sim::new(client, params, seed, initial)));
      }
      Some(Err(err)) if wgpu_options.adapter.is_some() => return Err(err),
      Some(Err(err)) => tracing::debug!("no wgpu backend ({err:#}); falling back"),
      None => tracing::debug!("the wgpu backend panicked while starting; falling back"),
    }

    Ok(AnySim::Cpu(Sim::new(
      CpuRuntime::client(&CpuDevice),
      params,
      seed,
      initial,
    )))
  }

  pub fn kind(&self) -> RuntimeKind {
    match self {
      AnySim::Cpu(_) => RuntimeKind::Cpu,
      AnySim::Cuda(_) => RuntimeKind::Cuda,
      AnySim::Wgpu(_) => RuntimeKind::Wgpu,
    }
  }

  pub fn step(&mut self) {
    dispatch!(self, sim => sim.step())
  }

  /// Seed `count` founder plants; see [`crate::bodies::spawn_founders`].
  pub fn spawn_founders(&mut self, count: usize) -> usize {
    dispatch!(self, sim => crate::bodies::spawn_founders(sim, count))
  }

  pub fn organism_count(&self) -> usize {
    dispatch!(self, sim => sim.organism_count())
  }

  /// Write the population out; see [`crate::popdump`].
  pub fn save_population(
    &self,
    path: &std::path::Path,
  ) -> Result<(), crate::popdump::PopDumpError> {
    dispatch!(self, sim => crate::popdump::write(
      path,
      sim.population(),
      sim.bodies(),
      sim.step_count(),
      sim.seed(),
    ))
  }

  /// Install a saved population and re-grow its bodies; see
  /// [`crate::popdump::restore`].
  pub fn load_population(
    &mut self,
    snapshot: &crate::popdump::PopSnapshot,
  ) -> Result<usize, crate::popdump::PopDumpError> {
    dispatch!(self, sim => crate::popdump::restore(sim, snapshot))
  }

  /// Write one row of the metrics time series; see [`crate::metrics`].
  pub fn sample_metrics(&self, sampler: &mut crate::metrics::Sampler) -> std::io::Result<()> {
    dispatch!(self, sim => sampler.sample(sim))
  }

  pub fn counters(&self) -> crate::sim::Counters {
    dispatch!(self, sim => sim.counters())
  }

  pub fn population(&self) -> &crate::genome::Population {
    dispatch!(self, sim => sim.population())
  }

  pub fn bodies(&self) -> &crate::bodies::BodyState {
    dispatch!(self, sim => sim.bodies())
  }

  /// The brain's per-limb outputs; see [`Sim::brain_outputs`].
  pub fn brain_outputs(&self) -> Vec<f32> {
    dispatch!(self, sim => sim.brain_outputs())
  }

  /// The persistent latent state; see [`Sim::brain_latents`].
  pub fn brain_latents(&self) -> Vec<f32> {
    dispatch!(self, sim => sim.brain_latents())
  }

  pub fn brain(&self) -> &crate::brain::BrainState {
    dispatch!(self, sim => sim.brain())
  }

  /// The brain's sensor buffer, read back to the host.
  pub fn brain_sensors(&self) -> Vec<f32> {
    dispatch!(self, sim => sim.brain().read_sensors(sim.client()))
  }

  /// The limb geometry the last constraint pass published; see
  /// [`Sim::read_limb_geometry`].
  pub fn read_limb_geometry(&self) -> crate::bodies::LimbGeometryHost {
    dispatch!(self, sim => sim.read_limb_geometry())
  }

  pub fn sync(&self) {
    dispatch!(self, sim => sim.sync())
  }

  pub fn enable_timing(&mut self) {
    dispatch!(self, sim => sim.enable_timing())
  }

  pub fn read_particles(&self) -> SphHost {
    dispatch!(self, sim => sim.read_particles())
  }

  pub fn timings(&self) -> &KernelTimings {
    dispatch!(self, sim => sim.timings())
  }

  pub fn step_count(&self) -> u32 {
    dispatch!(self, sim => sim.step_count())
  }

  pub fn seed(&self) -> u64 {
    dispatch!(self, sim => sim.seed())
  }

  pub fn params(&self) -> &SimParams {
    dispatch!(self, sim => sim.params())
  }

  pub fn set_params(&mut self, params: SimParams) {
    dispatch!(self, sim => sim.set_params(params))
  }

  pub fn geometry(&self) -> &WorldGeometry {
    dispatch!(self, sim => sim.geometry())
  }

  pub fn soil(&self) -> &SoilGrid {
    dispatch!(self, sim => sim.soil())
  }
}

/// Whether this box can run a given backend.
///
/// This throws the initialized backend away, so it is for tests and for
/// answering the question on its own; the startup path uses
/// [`AnySim::new_auto`], which keeps what it starts.
pub fn available(kind: RuntimeKind) -> bool {
  match kind {
    RuntimeKind::Cpu => true,
    RuntimeKind::Cuda => probe(|| CudaRuntime::client(&CudaDevice::new(0))).is_some(),
    RuntimeKind::Wgpu => {
      probe(|| crate::wgpu_backend::headless_setup(wgpu::Backends::PRIMARY, None).is_ok())
        .unwrap_or(false)
    }
  }
}

/// Run something that may panic because a backend is not on this box.
///
/// CubeCL's `Runtime::client` is infallible and panics when the backend is not
/// there, so probing means catching that panic. `None` is that panic; anything
/// the closure returns comes back intact, which is what lets the startup path
/// keep the client it just built.
fn probe<T>(f: impl FnOnce() -> T) -> Option<T> {
  let previous = std::panic::take_hook();
  std::panic::set_hook(Box::new(|_| {}));
  let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).ok();
  std::panic::set_hook(previous);
  out
}
