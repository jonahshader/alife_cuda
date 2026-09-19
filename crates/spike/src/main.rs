//! Throwaway CubeCL feasibility spike. See `README.md` for the questions it
//! answers and the exact commands.
//!
//! Nothing in here opens a window.

mod config;
mod input;
mod kernels;
mod reference;
mod wgpu_setup;

use std::time::Instant;

use cubecl::prelude::*;

use config::*;
use input::Input;

const SEED: u64 = 0xA11FE;
const TIMED_LAUNCHES: usize = 100;

fn main() {
  let args: Vec<String> = std::env::args().skip(1).collect();
  let what = args.first().map(String::as_str).unwrap_or("all");

  println!(
    "config: {NUM_PARTICLES} particles, grid {GRID_W}x{GRID_H}, cell {CELL_SIZE}, \
         radius {SMOOTHING_RADIUS}, max {MAX_PER_CELL} per cell"
  );

  let cfg = Cfg::default();
  let input = Input::generate(SEED);
  let t = Instant::now();
  let want = reference::run(&cfg, &input);
  println!(
    "reference: {:.1} ms (single-threaded Rust), density[0..3] = {:?}",
    t.elapsed().as_secs_f64() * 1e3,
    &want.density[0..3]
  );

  match what {
    "adapters" => list_adapters(),
    "cpu" => run_cpu(&cfg, &input, &want),
    "cuda" => run_cuda(&cfg, &input, &want),
    "wgpu" => run_wgpu(&cfg, &input, &want, None),
    "llvmpipe" => run_wgpu(&cfg, &input, &want, Some("llvmpipe")),
    "wgpu-cubecl-device" => run_wgpu_cubecl_device(&cfg, &input, &want),
    "wgpu-cpu-device" => run_wgpu_cpu_device(&cfg, &input, &want),
    "share" => run_share(&cfg, &input, &want),
    "gdb-probe" => gdb_probe(&cfg, &input),
    "all" => {
      list_adapters();
      run_cpu(&cfg, &input, &want);
      run_cuda(&cfg, &input, &want);
      run_wgpu(&cfg, &input, &want, None);
      run_wgpu_cubecl_device(&cfg, &input, &want);
      run_wgpu_cpu_device(&cfg, &input, &want);
      run_wgpu(&cfg, &input, &want, Some("llvmpipe"));
      run_share(&cfg, &input, &want);
    }
    other => {
      eprintln!(
        "unknown target {other:?}; expected one of: \
                 adapters cpu cuda wgpu llvmpipe wgpu-cubecl-device share all"
      );
      std::process::exit(2);
    }
  }
}

fn list_adapters() {
  println!("\n== wgpu adapters (Vulkan) ==");
  for info in wgpu_setup::list_adapters(wgpu::Backends::VULKAN) {
    println!("  {:?} {} ({:?})", info.device_type, info.name, info.driver);
  }
}

/// Upload, warm up (which is what pays the JIT), time one launch and
/// `TIMED_LAUNCHES` launches, then read back and compare.
fn run<R: Runtime>(
  label: &str,
  client: ComputeClient<R>,
  cfg: &Cfg,
  input: &Input,
  want: &reference::Output,
) {
  println!("\n== {label} ==");

  let bufs = kernels::upload(&client, input);

  let t = Instant::now();
  kernels::launch_all(&client, &bufs, *cfg);
  sync(&client);
  println!(
    "  first launch (includes JIT compile): {:.1} ms",
    t.elapsed().as_secs_f64() * 1e3
  );

  let t = Instant::now();
  kernels::launch_all(&client, &bufs, *cfg);
  sync(&client);
  let one = t.elapsed().as_secs_f64() * 1e3;

  let t = Instant::now();
  for _ in 0..TIMED_LAUNCHES {
    kernels::launch_all(&client, &bufs, *cfg);
  }
  sync(&client);
  let many = t.elapsed().as_secs_f64() * 1e3;

  println!("  1 launch:    {one:.3} ms");
  println!(
    "  {TIMED_LAUNCHES} launches: {many:.3} ms ({:.3} ms each)",
    many / TIMED_LAUNCHES as f64
  );

  let got = kernels::download(&client, &bufs);
  report(cfg, want, &got);
}

fn sync<R: Runtime>(client: &ComputeClient<R>) {
  pollster::block_on(client.sync()).expect("sync");
}

fn report(cfg: &Cfg, want: &reference::Output, got: &reference::Output) {
  let diff = reference::compare(cfg, want, got);
  println!(
    "  vs reference: counts {} | grid sets {} | max|density| {:.3e} | max|near_density| {:.3e} -> {}",
    if diff.counts_equal { "equal" } else { "DIFFER" },
    if diff.grid_sets_equal {
      "equal"
    } else {
      "DIFFER"
    },
    diff.max_abs_density,
    diff.max_abs_near_density,
    if diff.ok() { "PASS" } else { "FAIL" },
  );
}

fn run_cpu(cfg: &Cfg, input: &Input, want: &reference::Output) {
  use cubecl_cpu::{CpuDevice, CpuRuntime};
  let client = CpuRuntime::client(&CpuDevice);
  run(
    "cpu runtime (cubecl-cpu / LLVM JIT)",
    client,
    cfg,
    input,
    want,
  );
}

fn run_cuda(cfg: &Cfg, input: &Input, want: &reference::Output) {
  use cubecl_cuda::{CudaDevice, CudaRuntime};
  let client = CudaRuntime::client(&CudaDevice::new(0));
  run("cuda runtime", client, cfg, input, want);
}

/// The sharing path: we build the `wgpu::Device` and hand it to CubeCL.
fn run_wgpu(cfg: &Cfg, input: &Input, want: &reference::Output, adapter: Option<&str>) {
  let setup = wgpu_setup::headless_setup(wgpu::Backends::VULKAN, adapter);
  let info = setup.adapter.get_info();
  let (_device, client) = wgpu_setup::client_on(&setup);
  run(
    &format!(
      "wgpu runtime on shared device — {} ({:?})",
      info.name, info.backend
    ),
    client,
    cfg,
    input,
    want,
  );
}

/// The ordinary path: CubeCL creates the device itself from a `WgpuDevice`.
fn run_wgpu_cubecl_device(cfg: &Cfg, input: &Input, want: &reference::Output) {
  use cubecl_wgpu::{WgpuDevice, WgpuRuntime};
  let device = WgpuDevice::DiscreteGpu(0);
  let client: ComputeClient<WgpuRuntime> = WgpuRuntime::client(&device);
  run(
    "wgpu runtime on WgpuDevice::DiscreteGpu(0)",
    client,
    cfg,
    input,
    want,
  );
}

/// CubeCL's own adapter selection: `WgpuDevice::Cpu` is the software adapter,
/// which on this box is llvmpipe.
fn run_wgpu_cpu_device(cfg: &Cfg, input: &Input, want: &reference::Output) {
  use cubecl_wgpu::{WgpuDevice, WgpuRuntime};
  let device = WgpuDevice::Cpu;
  let client: ComputeClient<WgpuRuntime> = WgpuRuntime::client(&device);
  run("wgpu runtime on WgpuDevice::Cpu", client, cfg, input, want);
}

fn run_share(cfg: &Cfg, input: &Input, want: &reference::Output) {
  println!("\n== device and buffer sharing with a rendering pipeline ==");
  let setup = wgpu_setup::headless_setup(wgpu::Backends::VULKAN, None);
  println!("  adapter: {}", setup.adapter.get_info().name);

  let (_device, client) = wgpu_setup::client_on(&setup);
  let bufs = kernels::upload(&client, input);
  kernels::launch_all(&client, &bufs, *cfg);
  sync(&client);

  let shared = wgpu_setup::shared_buffer(&client, &bufs.density);
  println!(
    "  CubeCL density handle -> wgpu::Buffer {{ size {} B, offset {} }} (usages {:?})",
    shared.size,
    shared.offset,
    shared.buffer.usage()
  );

  let n = 4096.min(want.density.len());

  let raw = wgpu_setup::read_back_with_wgpu(&setup, &shared, n);
  let d = max_abs(&want.density[..n], &raw);
  println!("  wgpu copy_buffer_to_buffer + map_async: max|diff| = {d:.3e}");

  let doubled = wgpu_setup::compute_pass_on_shared(&setup, &shared, n);
  let want2: Vec<f32> = want.density[..n].iter().map(|v| v * 2.0).collect();
  let d2 = max_abs(&want2, &doubled);
  println!("  our own compute pass reading it as storage: max|diff| = {d2:.3e}");

  let idx = 7u32;
  let texel = wgpu_setup::render_pass_on_shared(&setup, &shared, idx);
  println!(
    "  our own offscreen render pass, fragment reads src[{idx}]: {texel} (want {})",
    want.density[idx as usize]
  );
}

/// A breakpoint target for the debuggability question: by the time this is
/// called the CPU runtime has JIT-compiled and run every kernel, so whatever
/// `info functions` / `maintenance info jit` shows at this point is what a
/// debugger gets to see of a `#[cube]` body.
#[inline(never)]
#[unsafe(no_mangle)]
pub extern "C" fn spike_after_jit() {
  std::hint::black_box(());
}

fn gdb_probe(cfg: &Cfg, input: &Input) {
  use cubecl_cpu::{CpuDevice, CpuRuntime};
  let client = CpuRuntime::client(&CpuDevice);
  let bufs = kernels::upload(&client, input);
  kernels::launch_all(&client, &bufs, *cfg);
  sync(&client);
  spike_after_jit();
  println!("gdb-probe: kernels JIT-compiled and executed");
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
  a.iter()
    .zip(b.iter())
    .map(|(x, y)| (x - y).abs())
    .fold(0.0f32, f32::max)
}
