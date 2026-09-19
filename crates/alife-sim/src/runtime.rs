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
/// CubeCL's `Runtime::client` is infallible and panics when the backend is not
/// there, so probing means catching that panic; nothing is kept from a failed
/// probe.
pub fn available(kind: RuntimeKind) -> bool {
  match kind {
    RuntimeKind::Cpu => true,
    RuntimeKind::Cuda => probe(|| {
      let _ = CudaRuntime::client(&CudaDevice::new(0));
    }),
    RuntimeKind::Wgpu => probe(|| {
      let _ = crate::wgpu_backend::headless_setup(wgpu::Backends::PRIMARY, None).unwrap();
    }),
  }
}

/// CUDA if this box can run it, else wgpu, else the CPU runtime.
pub fn default_runtime() -> RuntimeKind {
  for kind in [RuntimeKind::Cuda, RuntimeKind::Wgpu] {
    if available(kind) {
      return kind;
    }
  }
  RuntimeKind::Cpu
}

fn probe(f: impl FnOnce()) -> bool {
  let previous = std::panic::take_hook();
  std::panic::set_hook(Box::new(|_| {}));
  let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).is_ok();
  std::panic::set_hook(previous);
  ok
}
