//! One tick of the brain, as launches.
//!
//! The spec's order (`docs/organism.md`, *Brain* and *Implementation
//! layout*): tokens; input cross-attention, latents attending to tokens; a
//! self-attention and an MLP over the latents; the gated update that makes
//! the latents persistent; output cross-attention, tokens attending to the
//! updated latents; and a sprout head and an actuator head per limb. Single
//! head throughout, fp32 accumulate, softmax with the row maximum subtracted.
//!
//! **Six kernels do the whole pass.** [`gemv`] is every projection — one
//! unit per output element, `k` multiply-adds each, with a bias, an
//! activation and a residual term switched on at comptime and the weight
//! slice's offset passed as a runtime argument, so one compiled kernel serves
//! every launch that shares a shape. [`attn_scores`] and [`attn_attend`] are
//! an attention block, and [`gate_update`], [`latent_norm`] and
//! [`write_heads`] are the steps with no other shape.
//!
//! [`latent_norm`] is the one step the spec does not list, and the pass
//! diverges without it; its own comment has the measurement.
//!
//! **Why the residual is always read.** `gemv` adds a residual term
//! unconditionally and the launches with no residual in the spec point it at
//! a buffer of zeros. That is one extra load against the `2k` a unit already
//! does — under 2% — and it halves the number of compiled variants, because
//! `in_q` and `in_o` then differ only in their arguments.

use cubecl::prelude::*;
use cubecl_runtime::server::Handle;

use super::{BrainCfg, BrainDevice, HEAD_DIM, Weights, sigmoid, silu};
use crate::genome::Population;
use crate::genome::shape::{BrainShape, IDENTITY_DIM, SENSOR_DIM};
use crate::kernels::{CUBE_DIM, cube_count, whole};

/// Launches one tick costs, [`launch`] in order. Measured in `docs/perf.md`.
pub const LAUNCHES: usize = 22;

// --- The projection kernel ---

/// The shape of one [`gemv`] launch. `jobs` consecutive weight slices of the
/// same shape run in one launch, which is what makes a block's keys and
/// values — adjacent in the slice table, same shape — a single dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GemvCfg {
  pub max_organisms: u32,
  /// Rows per organism in the source and in each job's output.
  pub rows: u32,
  /// Input width: the weight slice's row count.
  pub k: u32,
  /// Output width: the weight slice's column count.
  pub n: u32,
  pub jobs: u32,
  pub param_count: u32,
  pub bias: bool,
  pub silu: bool,
}

impl GemvCfg {
  fn src_len(&self) -> usize {
    (self.max_organisms * self.rows * self.k) as usize
  }

  fn out_len(&self) -> usize {
    (self.jobs * self.max_organisms * self.rows * self.n) as usize
  }

  fn residual_len(&self) -> usize {
    (self.max_organisms * self.rows * self.n) as usize
  }
}

/// `out[job][o][r][c] = bias[c] + sum_j src[o][r][j] * w[job][j][c]`, then the
/// activation, then the residual. One unit per output element.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn gemv<W: Float>(
  brain: &[W],
  src: &[f32],
  residual: &[f32],
  out: &mut [f32],
  alive: &[u32],
  w_off: u32,
  b_off: u32,
  #[comptime] cfg: GemvCfg,
) {
  let t = ABSOLUTE_POS as u32;
  if t >= comptime!(cfg.jobs * cfg.max_organisms * cfg.rows * cfg.n) {
    terminate!();
  }
  let n = comptime!(cfg.n);
  let k = comptime!(cfg.k);
  let rows = comptime!(cfg.rows);
  let c = t % n;
  let r = (t / n) % rows;
  let o = (t / comptime!(cfg.n * cfg.rows)) % comptime!(cfg.max_organisms);
  let job = t / comptime!(cfg.n * cfg.rows * cfg.max_organisms);

  if alive[o as usize] == 0u32 {
    terminate!();
  }

  let row = o * comptime!(cfg.param_count);
  let wb = row + w_off + job * comptime!(cfg.k * cfg.n);
  let sb = (o * rows + r) * k;

  let mut acc = 0.0f32;
  if comptime!(cfg.bias) {
    // A bias belongs to one weight slice, so a multi-job launch has none;
    // `launch_gemv` asserts that rather than guessing a stride.
    acc = f32::cast_from(brain[(row + b_off + c) as usize]);
  }
  for j in 0..k {
    acc += src[(sb + j) as usize] * f32::cast_from(brain[(wb + j * n + c) as usize]);
  }
  if comptime!(cfg.silu) {
    acc = silu(acc);
  }
  out[t as usize] = acc + residual[((o * rows + r) * n + c) as usize];
}

// --- The attention kernels ---

/// The shape of one attention block. `queries` and `keys` are row counts per
/// organism; `d` is the head width, which is `d_latent` everywhere.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AttnCfg {
  pub max_organisms: u32,
  pub max_limbs: u32,
  pub queries: u32,
  pub keys: u32,
  pub d: u32,
  /// Whether a key is a limb, so an absent limb is masked out of the softmax.
  /// True for the input cross-attention only: the other two blocks' keys are
  /// latents, which are all present.
  pub mask_keys: bool,
}

/// `scores[o][query][key] = scale * dot(q[query], k[key])`. One unit per
/// score.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn attn_scores(
  q: &[f32],
  k: &[f32],
  scores: &mut [f32],
  alive: &[u32],
  q_base: u32,
  k_base: u32,
  scale: f32,
  #[comptime] cfg: AttnCfg,
) {
  let t = ABSOLUTE_POS as u32;
  if t >= comptime!(cfg.max_organisms * cfg.queries * cfg.keys) {
    terminate!();
  }
  let keys = comptime!(cfg.keys);
  let d = comptime!(cfg.d);
  let key = t % keys;
  let query = (t / keys) % comptime!(cfg.queries);
  let o = t / comptime!(cfg.queries * cfg.keys);
  if alive[o as usize] == 0u32 {
    terminate!();
  }

  let qb = q_base + (o * comptime!(cfg.queries) + query) * d;
  let kb = k_base + (o * keys + key) * d;
  let mut acc = 0.0f32;
  for e in 0..d {
    acc += q[(qb + e) as usize] * k[(kb + e) as usize];
  }
  scores[t as usize] = acc * scale;
}

/// `out[o][query][c] = sum_key softmax(scores[query])[key] * v[key][c]`. One
/// unit per output component.
///
/// The softmax is recomputed by every unit of a query's row rather than
/// materialised by a launch of its own: `keys` exponentials and `keys` loads
/// from a row that is at most 16 floats wide, against a launch and a buffer.
/// The maximum is subtracted first, and both passes skip masked keys, so a
/// masked key contributes nothing rather than `exp(-inf)`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn attn_attend(
  scores: &[f32],
  v: &[f32],
  out: &mut [f32],
  alive: &[u32],
  part_type: &[u32],
  v_base: u32,
  #[comptime] cfg: AttnCfg,
) {
  let t = ABSOLUTE_POS as u32;
  if t >= comptime!(cfg.max_organisms * cfg.queries * cfg.d) {
    terminate!();
  }
  let d = comptime!(cfg.d);
  let keys = comptime!(cfg.keys);
  let c = t % d;
  let query = (t / d) % comptime!(cfg.queries);
  let o = t / comptime!(cfg.queries * cfg.d);
  if alive[o as usize] == 0u32 {
    terminate!();
  }

  let srow = (o * comptime!(cfg.queries) + query) * keys;
  let limb_base = o * comptime!(cfg.max_limbs);

  let mut best = 0.0f32;
  let mut present_count = 0u32;
  for key in 0..keys {
    let mut present = true;
    if comptime!(cfg.mask_keys) {
      present = part_type[(limb_base + key) as usize] != 0u32;
    }
    if present {
      let s = scores[(srow + key) as usize];
      if present_count == 0u32 || s > best {
        best = s;
      }
      present_count += 1u32;
    }
  }

  let mut sum = 0.0f32;
  let mut acc = 0.0f32;
  for key in 0..keys {
    let mut present = true;
    if comptime!(cfg.mask_keys) {
      present = part_type[(limb_base + key) as usize] != 0u32;
    }
    if present {
      let e = f32::exp(scores[(srow + key) as usize] - best);
      sum += e;
      acc += e * v[(v_base + (o * keys + key) * d + c) as usize];
    }
  }

  // An organism whose every limb record is absent has nothing to attend to.
  // It cannot happen to a live plant — its root is present — but a corrupt
  // genome would otherwise divide by zero here.
  let mut result = 0.0f32;
  if present_count > 0u32 {
    result = acc / sum;
  }
  out[t as usize] = result;
}

// --- The gate and the heads ---

/// `gated = g * trunk + (1 - g) * latent`, with
/// `g = sigmoid(gate_w · trunk + gate_b)`. One unit per latent component.
///
/// Writes a scratch buffer rather than the persistent latents, because
/// [`latent_norm`] has to read a whole latent vector after the blend and a
/// unit cannot read what its neighbours are still writing.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn gate_update<W: Float>(
  brain: &[W],
  trunk: &[f32],
  latents: &[f32],
  gated: &mut [f32],
  alive: &[u32],
  gate_w_off: u32,
  gate_b_off: u32,
  #[comptime] cfg: BrainCfg,
) {
  let t = ABSOLUTE_POS as u32;
  if t >= comptime!(cfg.max_organisms * cfg.n_latents * cfg.d_latent) {
    terminate!();
  }
  let d = comptime!(cfg.d_latent);
  let c = t % d;
  let o = t / comptime!(cfg.n_latents * cfg.d_latent);
  if alive[o as usize] == 0u32 {
    terminate!();
  }

  let row = o * comptime!(cfg.param_count);
  // `t - c` is the start of this latent vector: the layout is
  // `[organism][latent][component]`.
  let hb = t - c;
  let mut acc = f32::cast_from(brain[(row + gate_b_off + c) as usize]);
  for e in 0..d {
    let w = brain[(row + gate_w_off + e * d + c) as usize];
    acc += trunk[(hb + e) as usize] * f32::cast_from(w);
  }
  let g = sigmoid(acc);
  gated[t as usize] = g * trunk[t as usize] + (1.0f32 - g) * latents[t as usize];
}

/// Guard against a zero-length latent vector, in the units the mean square is
/// measured in.
pub const LATENT_NORM_EPS: f32 = 1e-6;

/// Scale each latent vector to unit RMS on its way into the persistent state.
/// One unit per latent component.
///
/// **This is not in the spec's tick, and it is not optional.** One tick makes
/// three residual adds onto the latent stream — the input cross-attention,
/// the self-attention and the MLP — and none of them is normalised, so with
/// weights initialised at `1/sqrt(fan_in)` each one multiplies the latent
/// norm by about `sqrt(2)`. Measured on the default world at `--founders
/// 256`: the largest latent grows by a factor of ~2.6 per step, reaches 4e18
/// by step 40 and overflows to infinity by step 45, taking every head with
/// it. A Perceiver-IO normalises each block's input; normalising the
/// recurrent state instead bounds the same thing in one launch and, unlike a
/// learned LayerNorm, adds no parameters, so the genome's slice table is
/// untouched. `docs/organism.md` records the divergence.
#[cube(launch)]
pub fn latent_norm(gated: &[f32], latents: &mut [f32], alive: &[u32], #[comptime] cfg: BrainCfg) {
  let t = ABSOLUTE_POS as u32;
  if t >= comptime!(cfg.max_organisms * cfg.n_latents * cfg.d_latent) {
    terminate!();
  }
  let d = comptime!(cfg.d_latent);
  let c = t % d;
  let o = t / comptime!(cfg.n_latents * cfg.d_latent);
  if alive[o as usize] == 0u32 {
    terminate!();
  }

  let base = t - c;
  let mut sum = 0.0f32;
  for e in 0..d {
    let v = gated[(base + e) as usize];
    sum += v * v;
  }
  let inv = 1.0f32 / f32::sqrt(sum / d as f32 + LATENT_NORM_EPS);
  latents[t as usize] = gated[t as usize] * inv;
}

/// The two heads, one unit per output. The sprout logits come first, then the
/// two actuator outputs, so a limb's outputs are `HEAD_DIM` contiguous
/// floats.
///
/// An absent limb and a free organism slot are written as zeros: whatever
/// reads a sprout logit must not see the last organism that held the slot.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn write_heads<W: Float>(
  brain: &[W],
  z: &[f32],
  part_type: &[u32],
  alive: &[u32],
  out: &mut [f32],
  sprout_w_off: u32,
  sprout_b_off: u32,
  act_w_off: u32,
  act_b_off: u32,
  #[comptime] cfg: BrainCfg,
) {
  let t = ABSOLUTE_POS as u32;
  if t >= comptime!(cfg.max_organisms * cfg.max_limbs * cfg.head_dim) {
    terminate!();
  }
  let hd = comptime!(cfg.head_dim);
  let j = t % hd;
  let li = t / hd;
  let o = li / comptime!(cfg.max_limbs);

  out[t as usize] = 0.0f32;
  if alive[o as usize] == 0u32 || part_type[li as usize] == 0u32 {
    terminate!();
  }

  let row = o * comptime!(cfg.param_count);
  let dt = comptime!(cfg.d_token);
  let nt = comptime!(cfg.n_types);
  let zb = li * dt;

  let mut wb = row + sprout_w_off;
  let mut bb = row + sprout_b_off;
  let mut cols = nt.runtime();
  let mut col = j;
  if j >= nt {
    wb = row + act_w_off;
    bb = row + act_b_off;
    cols = 2u32;
    col = j - nt;
  }

  let mut acc = f32::cast_from(brain[(bb + col) as usize]);
  for c in 0..dt {
    acc += z[(zb + c) as usize] * f32::cast_from(brain[(wb + c * cols + col) as usize]);
  }
  out[t as usize] = acc;
}

// --- Launchers ---

#[allow(clippy::too_many_arguments)]
fn launch_gemv<R: Runtime>(
  client: &ComputeClient<R>,
  weights: &Weights,
  src: &Handle,
  residual: &Handle,
  out: &Handle,
  alive: &Handle,
  w_off: u32,
  b_off: u32,
  cfg: GemvCfg,
) {
  debug_assert!(
    !cfg.bias || cfg.jobs == 1,
    "a bias belongs to one slice, so it cannot be shared by a multi-job launch"
  );
  let brain_len = (cfg.max_organisms * cfg.param_count) as usize;
  let count = cube_count(cfg.out_len());
  let dim = CubeDim::new_1d(CUBE_DIM);

  macro_rules! go {
    ($elem:ty, $handle:expr) => {
      gemv::launch::<$elem, R>(
        client,
        count,
        dim,
        whole($handle, brain_len),
        whole(src, cfg.src_len()),
        whole(residual, cfg.residual_len()),
        whole(out, cfg.out_len()),
        whole(alive, cfg.max_organisms as usize),
        w_off,
        b_off,
        cfg,
      )
    };
  }
  match weights {
    Weights::F32(handle) => go!(f32, handle),
    Weights::F16(handle) => go!(half::f16, handle),
  }
}

#[allow(clippy::too_many_arguments)]
fn launch_scores<R: Runtime>(
  client: &ComputeClient<R>,
  q: &Handle,
  q_len: usize,
  k: &Handle,
  k_len: usize,
  scores: &Handle,
  alive: &Handle,
  q_base: u32,
  k_base: u32,
  cfg: AttnCfg,
) {
  let n = (cfg.max_organisms * cfg.queries * cfg.keys) as usize;
  attn_scores::launch::<R>(
    client,
    cube_count(n),
    CubeDim::new_1d(CUBE_DIM),
    whole(q, q_len),
    whole(k, k_len),
    whole(scores, n),
    whole(alive, cfg.max_organisms as usize),
    q_base,
    k_base,
    1.0 / (cfg.d as f32).sqrt(),
    cfg,
  );
}

#[allow(clippy::too_many_arguments)]
fn launch_attend<R: Runtime>(
  client: &ComputeClient<R>,
  scores: &Handle,
  v: &Handle,
  v_len: usize,
  out: &Handle,
  alive: &Handle,
  part_type: &Handle,
  v_base: u32,
  cfg: AttnCfg,
) {
  let n = (cfg.max_organisms * cfg.queries * cfg.d) as usize;
  attn_attend::launch::<R>(
    client,
    cube_count(n),
    CubeDim::new_1d(CUBE_DIM),
    whole(
      scores,
      (cfg.max_organisms * cfg.queries * cfg.keys) as usize,
    ),
    whole(v, v_len),
    whole(out, n),
    whole(alive, cfg.max_organisms as usize),
    whole(part_type, (cfg.max_organisms * cfg.max_limbs) as usize),
    v_base,
    cfg,
  );
}

/// Every launch of one tick, in order, each handed to `timed` under its own
/// name — the same shape [`crate::kernels::grid::build`] uses.
///
/// The caller runs this only when something is alive: with no organism the
/// whole pass is skipped, so a run without founders costs exactly what it did
/// before brains existed.
pub fn launch<R: Runtime>(
  client: &ComputeClient<R>,
  weights: &Weights,
  brain: &BrainDevice,
  pop: &Population,
  shape: &BrainShape,
  cfg: BrainCfg,
  timed: &mut dyn FnMut(&'static str, &mut (dyn FnMut() + Send)),
) {
  let alive = &pop.device.organisms.alive;
  let part_type = &pop.device.limbs.part_type;
  let latents = &pop.device.latents;
  let zeros = &brain.zeros;

  let (o, l, m) = (cfg.max_organisms, cfg.max_limbs, cfg.n_latents);
  let (dt, dl, th) = (cfg.d_token, cfg.d_latent, cfg.trunk_hidden);
  let latent_len = cfg.latent_len();
  let limb_latent_len = cfg.limb_latent_len();

  // `latent` is every projection that reads and writes latent rows; `token`
  // is every one that reads token rows.
  let latent = |jobs: u32| GemvCfg {
    max_organisms: o,
    rows: m,
    k: dl,
    n: dl,
    jobs,
    param_count: cfg.param_count,
    bias: false,
    silu: false,
  };
  let from_token = |jobs: u32| GemvCfg {
    rows: l,
    k: dt,
    ..latent(jobs)
  };
  let attn = |queries: u32, keys: u32, mask_keys: bool| AttnCfg {
    max_organisms: o,
    max_limbs: l,
    queries,
    keys,
    d: dl,
    mask_keys,
  };

  timed("brain_in_q", &mut || {
    launch_gemv(
      client,
      weights,
      latents,
      zeros,
      &brain.in_q,
      alive,
      shape.in_q().start as u32,
      0,
      latent(1),
    );
  });
  // `in_k` and `in_v` are adjacent in the slice table and the same shape, so
  // one launch writes both: keys first, values one `limb_latent_len` on.
  timed("brain_in_kv", &mut || {
    launch_gemv(
      client,
      weights,
      &brain.tokens,
      zeros,
      &brain.in_kv,
      alive,
      shape.in_k().start as u32,
      0,
      from_token(2),
    );
  });
  timed("brain_in_scores", &mut || {
    launch_scores(
      client,
      &brain.in_q,
      latent_len,
      &brain.in_kv,
      2 * limb_latent_len,
      &brain.in_scores,
      alive,
      0,
      0,
      attn(m, l, true),
    );
  });
  timed("brain_in_attend", &mut || {
    launch_attend(
      client,
      &brain.in_scores,
      &brain.in_kv,
      2 * limb_latent_len,
      &brain.in_att,
      alive,
      part_type,
      limb_latent_len as u32,
      attn(m, l, true),
    );
  });
  timed("brain_in_out", &mut || {
    launch_gemv(
      client,
      weights,
      &brain.in_att,
      latents,
      &brain.h1,
      alive,
      shape.in_o().start as u32,
      0,
      latent(1),
    );
  });

  timed("brain_self_q", &mut || {
    launch_gemv(
      client,
      weights,
      &brain.h1,
      zeros,
      &brain.self_q,
      alive,
      shape.self_q().start as u32,
      0,
      latent(1),
    );
  });
  timed("brain_self_kv", &mut || {
    launch_gemv(
      client,
      weights,
      &brain.h1,
      zeros,
      &brain.self_kv,
      alive,
      shape.self_k().start as u32,
      0,
      latent(2),
    );
  });
  timed("brain_self_scores", &mut || {
    launch_scores(
      client,
      &brain.self_q,
      latent_len,
      &brain.self_kv,
      2 * latent_len,
      &brain.self_scores,
      alive,
      0,
      0,
      attn(m, m, false),
    );
  });
  timed("brain_self_attend", &mut || {
    launch_attend(
      client,
      &brain.self_scores,
      &brain.self_kv,
      2 * latent_len,
      &brain.self_att,
      alive,
      part_type,
      latent_len as u32,
      attn(m, m, false),
    );
  });
  timed("brain_self_out", &mut || {
    launch_gemv(
      client,
      weights,
      &brain.self_att,
      &brain.h1,
      &brain.h2,
      alive,
      shape.self_o().start as u32,
      0,
      latent(1),
    );
  });

  timed("brain_mlp1", &mut || {
    launch_gemv(
      client,
      weights,
      &brain.h2,
      zeros,
      &brain.hidden,
      alive,
      shape.mlp_w1().start as u32,
      shape.mlp_b1().start as u32,
      GemvCfg {
        n: th,
        bias: true,
        silu: true,
        ..latent(1)
      },
    );
  });
  timed("brain_mlp2", &mut || {
    launch_gemv(
      client,
      weights,
      &brain.hidden,
      &brain.h2,
      &brain.trunk,
      alive,
      shape.mlp_w2().start as u32,
      shape.mlp_b2().start as u32,
      GemvCfg {
        k: th,
        bias: true,
        ..latent(1)
      },
    );
  });
  timed("brain_gate", &mut || {
    launch_gate(client, weights, brain, pop, shape, cfg);
  });
  timed("brain_latent_norm", &mut || {
    latent_norm::launch::<R>(
      client,
      cube_count(latent_len),
      CubeDim::new_1d(CUBE_DIM),
      whole(&brain.gated, latent_len),
      whole(latents, latent_len),
      whole(alive, o as usize),
      cfg,
    );
  });

  timed("brain_out_q", &mut || {
    launch_gemv(
      client,
      weights,
      &brain.tokens,
      zeros,
      &brain.out_q,
      alive,
      shape.out_q().start as u32,
      0,
      from_token(1),
    );
  });
  timed("brain_out_kv", &mut || {
    launch_gemv(
      client,
      weights,
      latents,
      zeros,
      &brain.out_kv,
      alive,
      shape.out_k().start as u32,
      0,
      latent(2),
    );
  });
  timed("brain_out_scores", &mut || {
    launch_scores(
      client,
      &brain.out_q,
      limb_latent_len,
      &brain.out_kv,
      2 * latent_len,
      &brain.out_scores,
      alive,
      0,
      0,
      attn(l, m, false),
    );
  });
  timed("brain_out_attend", &mut || {
    launch_attend(
      client,
      &brain.out_scores,
      &brain.out_kv,
      2 * latent_len,
      &brain.out_att,
      alive,
      part_type,
      latent_len as u32,
      attn(l, m, false),
    );
  });
  timed("brain_out_out", &mut || {
    launch_gemv(
      client,
      weights,
      &brain.out_att,
      zeros,
      &brain.z,
      alive,
      shape.out_o().start as u32,
      0,
      GemvCfg {
        rows: l,
        k: dl,
        n: dt,
        ..latent(1)
      },
    );
  });
  timed("brain_heads", &mut || {
    launch_heads(client, weights, brain, pop, shape, cfg);
  });
}

fn launch_gate<R: Runtime>(
  client: &ComputeClient<R>,
  weights: &Weights,
  brain: &BrainDevice,
  pop: &Population,
  shape: &BrainShape,
  cfg: BrainCfg,
) {
  let brain_len = (cfg.max_organisms * cfg.param_count) as usize;
  let n = cfg.latent_len();
  let count = cube_count(n);
  let dim = CubeDim::new_1d(CUBE_DIM);
  macro_rules! go {
    ($elem:ty, $handle:expr) => {
      gate_update::launch::<$elem, R>(
        client,
        count,
        dim,
        whole($handle, brain_len),
        whole(&brain.trunk, n),
        whole(&pop.device.latents, n),
        whole(&brain.gated, n),
        whole(&pop.device.organisms.alive, cfg.max_organisms as usize),
        shape.gate_w().start as u32,
        shape.gate_b().start as u32,
        cfg,
      )
    };
  }
  match weights {
    Weights::F32(handle) => go!(f32, handle),
    Weights::F16(handle) => go!(half::f16, handle),
  }
}

fn launch_heads<R: Runtime>(
  client: &ComputeClient<R>,
  weights: &Weights,
  brain: &BrainDevice,
  pop: &Population,
  shape: &BrainShape,
  cfg: BrainCfg,
) {
  let brain_len = (cfg.max_organisms * cfg.param_count) as usize;
  let count = cube_count(cfg.heads_len());
  let dim = CubeDim::new_1d(CUBE_DIM);
  macro_rules! go {
    ($elem:ty, $handle:expr) => {
      write_heads::launch::<$elem, R>(
        client,
        count,
        dim,
        whole($handle, brain_len),
        whole(&brain.z, cfg.tokens_len()),
        whole(&pop.device.limbs.part_type, cfg.limbs()),
        whole(&pop.device.organisms.alive, cfg.max_organisms as usize),
        whole(&brain.heads, cfg.heads_len()),
        shape.head_sprout().start as u32,
        shape.sprout_b().start as u32,
        shape.head_actuator().start as u32,
        shape.actuator_b().start as u32,
        cfg,
      )
    };
  }
  match weights {
    Weights::F32(handle) => go!(f32, handle),
    Weights::F16(handle) => go!(half::f16, handle),
  }
}

// --- The plain-Rust reference ---

/// Everything one organism's tick reads about its body. Slices are that
/// organism's, `max_limbs` long.
pub struct OrganismInput<'a> {
  pub part_type: &'a [u8],
  pub child_slot: &'a [u8],
  pub identity: &'a [[f32; IDENTITY_DIM]],
  pub rel_pos: &'a [glam::Vec2],
  pub angle: &'a [f32],
  pub depth: &'a [u32],
  /// `max_limbs * SENSOR_DIM`, as [`super::sense`] writes it.
  pub sensors: &'a [f32],
}

/// What one organism's tick produced.
pub struct ForwardOutput {
  /// `max_limbs * d_token`.
  pub tokens: Vec<f32>,
  /// `max_limbs * HEAD_DIM`: the sprout logits, then the actuator outputs.
  pub heads: Vec<f32>,
}

/// `out[r][c] = bias[c] + sum_j src[r][k] * w[j][c]`, accumulated in the
/// kernel's order.
fn matvec(
  src: &[f32],
  rows: usize,
  k: usize,
  n: usize,
  w: &[f32],
  bias: Option<&[f32]>,
) -> Vec<f32> {
  let mut out = vec![0.0f32; rows * n];
  for r in 0..rows {
    for c in 0..n {
      let mut acc = bias.map_or(0.0, |b| b[c]);
      for j in 0..k {
        acc += src[r * k + j] * w[j * n + c];
      }
      out[r * n + c] = acc;
    }
  }
  out
}

fn scores_ref(q: &[f32], k: &[f32], queries: usize, keys: usize, d: usize) -> Vec<f32> {
  let scale = 1.0 / (d as f32).sqrt();
  let mut out = vec![0.0f32; queries * keys];
  for query in 0..queries {
    for key in 0..keys {
      let mut acc = 0.0f32;
      for e in 0..d {
        acc += q[query * d + e] * k[key * d + e];
      }
      out[query * keys + key] = acc * scale;
    }
  }
  out
}

fn attend_ref(
  scores: &[f32],
  v: &[f32],
  queries: usize,
  keys: usize,
  d: usize,
  present: &[bool],
) -> Vec<f32> {
  let mut out = vec![0.0f32; queries * d];
  for query in 0..queries {
    let row = &scores[query * keys..(query + 1) * keys];
    let mut best = 0.0f32;
    let mut count = 0usize;
    for (key, present) in present.iter().enumerate().take(keys) {
      if *present {
        if count == 0 || row[key] > best {
          best = row[key];
        }
        count += 1;
      }
    }
    if count == 0 {
      continue;
    }
    for c in 0..d {
      let mut sum = 0.0f32;
      let mut acc = 0.0f32;
      for (key, present) in present.iter().enumerate().take(keys) {
        if *present {
          let e = (row[key] - best).exp();
          sum += e;
          acc += e * v[key * d + c];
        }
      }
      out[query * d + c] = acc / sum;
    }
  }
  out
}

/// Plain-Rust twin of the whole tick, over one organism's brain row.
///
/// `latents` is that organism's persistent latent state and is updated in
/// place, exactly as [`gate_update`] updates the device copy.
pub fn forward_ref(
  shape: &BrainShape,
  max_limbs: usize,
  brain: &[f32],
  latents: &mut [f32],
  input: &OrganismInput<'_>,
) -> ForwardOutput {
  assert_eq!(brain.len(), shape.param_count());
  assert_eq!(latents.len(), shape.latent_state_len());
  let (dt, dl, m, th) = (
    shape.d_token,
    shape.d_latent,
    shape.n_latents,
    shape.trunk_hidden,
  );
  let l = max_limbs;
  let w = |r: std::ops::Range<usize>| &brain[r];

  // Tokens. An absent limb is a zero token and is masked out below.
  let mut tokens = vec![0.0f32; l * dt];
  let present: Vec<bool> = (0..l).map(|i| input.part_type[i] != 0).collect();
  for limb in 0..l {
    if !present[limb] {
      continue;
    }
    let features = super::tokens::token_features_ref(
      input.rel_pos[limb],
      input.angle[limb],
      input.depth[limb],
      input.child_slot[limb],
      &input.identity[limb],
      &input.sensors[limb * SENSOR_DIM..],
    );
    let token = super::tokens::token_ref(shape, brain, input.part_type[limb], &features);
    tokens[limb * dt..(limb + 1) * dt].copy_from_slice(&token);
  }

  // Input cross-attention: latents ask, tokens answer.
  let in_q = matvec(latents, m, dl, dl, w(shape.in_q()), None);
  let in_k = matvec(&tokens, l, dt, dl, w(shape.in_k()), None);
  let in_v = matvec(&tokens, l, dt, dl, w(shape.in_v()), None);
  let s = scores_ref(&in_q, &in_k, m, l, dl);
  let att = attend_ref(&s, &in_v, m, l, dl, &present);
  let mut h1 = matvec(&att, m, dl, dl, w(shape.in_o()), None);
  for (h, latent) in h1.iter_mut().zip(latents.iter()) {
    *h += *latent;
  }

  // Latent self-attention.
  let all_present = vec![true; m];
  let self_q = matvec(&h1, m, dl, dl, w(shape.self_q()), None);
  let self_k = matvec(&h1, m, dl, dl, w(shape.self_k()), None);
  let self_v = matvec(&h1, m, dl, dl, w(shape.self_v()), None);
  let s = scores_ref(&self_q, &self_k, m, m, dl);
  let att = attend_ref(&s, &self_v, m, m, dl, &all_present);
  let mut h2 = matvec(&att, m, dl, dl, w(shape.self_o()), None);
  for (h, prev) in h2.iter_mut().zip(h1.iter()) {
    *h += *prev;
  }

  // The trunk.
  let mut hidden = matvec(&h2, m, dl, th, w(shape.mlp_w1()), Some(w(shape.mlp_b1())));
  for v in hidden.iter_mut() {
    *v = super::silu_ref(*v);
  }
  let mut trunk = matvec(
    &hidden,
    m,
    th,
    dl,
    w(shape.mlp_w2()),
    Some(w(shape.mlp_b2())),
  );
  for (v, prev) in trunk.iter_mut().zip(h2.iter()) {
    *v += *prev;
  }

  // The gated update of the persistent latents, then the RMS normalisation
  // that keeps the recurrence from compounding (see [`latent_norm`]).
  let gate = matvec(
    &trunk,
    m,
    dl,
    dl,
    w(shape.gate_w()),
    Some(w(shape.gate_b())),
  );
  let mut gated = vec![0.0f32; m * dl];
  for i in 0..m * dl {
    let g = super::sigmoid_ref(gate[i]);
    gated[i] = g * trunk[i] + (1.0 - g) * latents[i];
  }
  for row in 0..m {
    let mut sum = 0.0f32;
    for c in 0..dl {
      let v = gated[row * dl + c];
      sum += v * v;
    }
    let inv = 1.0 / (sum / dl as f32 + LATENT_NORM_EPS).sqrt();
    for c in 0..dl {
      latents[row * dl + c] = gated[row * dl + c] * inv;
    }
  }

  // Output cross-attention: tokens ask, the updated latents answer.
  let out_q = matvec(&tokens, l, dt, dl, w(shape.out_q()), None);
  let out_k = matvec(latents, m, dl, dl, w(shape.out_k()), None);
  let out_v = matvec(latents, m, dl, dl, w(shape.out_v()), None);
  let s = scores_ref(&out_q, &out_k, l, m, dl);
  let att = attend_ref(&s, &out_v, l, m, dl, &all_present);
  let z = matvec(&att, l, dl, dt, w(shape.out_o()), None);

  // The heads.
  let sprout = matvec(
    &z,
    l,
    dt,
    shape.sprout_b().len(),
    w(shape.head_sprout()),
    Some(w(shape.sprout_b())),
  );
  let actuator = matvec(
    &z,
    l,
    dt,
    2,
    w(shape.head_actuator()),
    Some(w(shape.actuator_b())),
  );
  let n_types = shape.sprout_b().len();
  let mut heads = vec![0.0f32; l * HEAD_DIM];
  for limb in 0..l {
    if !present[limb] {
      continue;
    }
    for j in 0..n_types {
      heads[limb * HEAD_DIM + j] = sprout[limb * n_types + j];
    }
    for j in 0..2 {
      heads[limb * HEAD_DIM + n_types + j] = actuator[limb * 2 + j];
    }
  }

  ForwardOutput { tokens, heads }
}

/// [`forward_ref`] over a whole population, so a test can compare the device
/// pass against it in one call.
///
/// `latents` is the whole `[max_organisms x n_latents x d_latent]` state and
/// is updated in place, as the device pass updates its own. It is passed
/// separately rather than taken from `pop` so a caller can start from the
/// population's copy without having to own a mutable one.
pub fn forward_population_ref(
  pop: &Population,
  geometry: &crate::bodies::LimbGeometryHost,
  sensors: &[f32],
  latents: &mut [f32],
) -> Vec<f32> {
  let shape = pop.shape;
  let ml = pop.max_limbs;
  let state = shape.latent_state_len();
  assert_eq!(latents.len(), pop.max_organisms * state);
  let mut heads = vec![0.0f32; pop.max_organisms * ml * HEAD_DIM];

  for o in 0..pop.max_organisms {
    if pop.organisms.alive[o] == 0 {
      continue;
    }
    let limbs = pop.limb_range(o);
    let part_type: Vec<u8> = pop.limbs.part_type[limbs.clone()]
      .iter()
      .map(|t| *t as u8)
      .collect();
    let input = OrganismInput {
      part_type: &part_type,
      child_slot: &pop.limbs.child_slot[limbs.clone()],
      identity: &pop.limbs.identity[limbs.clone()],
      rel_pos: &geometry.rel_pos[limbs.clone()],
      angle: &geometry.angle[limbs.clone()],
      depth: &geometry.depth[limbs.clone()],
      sensors: &sensors[limbs.start * SENSOR_DIM..limbs.end * SENSOR_DIM],
    };
    let out = forward_ref(
      &shape,
      ml,
      pop.brain_row(o),
      &mut latents[o * state..(o + 1) * state],
      &input,
    );
    heads[o * ml * HEAD_DIM..(o + 1) * ml * HEAD_DIM].copy_from_slice(&out.heads);
  }
  heads
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::kernels::test_support::{OrganismHarness, assert_close};

  /// Kernel against reference, on the CPU runtime, over the fixture plant.
  /// The cross-runtime version of this is `tests/brain.rs`.
  #[test]
  fn matches_reference() {
    let h = OrganismHarness::new();
    h.run_brain();

    let sensors = h.brain.read_sensors(&h.client);
    let mut latents = h.pop.latents.clone();
    let expected = forward_population_ref(&h.pop, &h.bodies.geometry, &sensors, &mut latents);

    let actual_heads = h.brain.read_heads(&h.client);
    let actual_latents =
      crate::genome::population::read_f32(&h.client, &h.pop.device.latents, h.pop.latents.len());

    assert_close(&actual_latents, &latents, 1e-5, "latents");
    assert_close(&actual_heads, &expected, 1e-5, "heads");
  }

  /// The latents persist: a second tick starts from what the first one left,
  /// so two ticks on the device track two ticks of the reference.
  #[test]
  fn the_latents_carry_across_ticks() {
    let h = OrganismHarness::new();
    let mut latents = h.pop.latents.clone();
    let before = latents.clone();

    h.run_brain();
    let sensors = h.brain.read_sensors(&h.client);
    forward_population_ref(&h.pop, &h.bodies.geometry, &sensors, &mut latents);
    h.run_brain();
    let expected = forward_population_ref(&h.pop, &h.bodies.geometry, &sensors, &mut latents);

    let actual_latents =
      crate::genome::population::read_f32(&h.client, &h.pop.device.latents, h.pop.latents.len());
    assert_ne!(
      &actual_latents[..h.pop.shape.latent_state_len()],
      &before[..h.pop.shape.latent_state_len()],
      "the tick left the latents untouched"
    );
    assert_close(&actual_latents, &latents, 1e-5, "latents after two ticks");
    assert_close(&h.brain.read_heads(&h.client), &expected, 1e-5, "heads");
  }

  /// A free organism slot and an absent limb produce no head at all: the
  /// life-cycle chunk reads sprout logits straight out of this buffer.
  #[test]
  fn dead_slots_and_absent_limbs_have_zero_heads() {
    let h = OrganismHarness::new();
    h.run_brain();
    let heads = h.brain.read_heads(&h.client);
    let ml = h.pop.max_limbs;

    for li in 0..h.brain.cfg.limbs() {
      let present = h.pop.organisms.alive[li / ml] == 1 && h.pop.limbs.part_type[li].is_present();
      let row = &heads[li * HEAD_DIM..(li + 1) * HEAD_DIM];
      if !present {
        assert!(row.iter().all(|v| *v == 0.0), "limb {li} was written");
      }
      assert!(row.iter().all(|v| v.is_finite()), "limb {li} is not finite");
    }
    // Something was written, or the test above is vacuous.
    assert!(heads.iter().any(|v| *v != 0.0));
  }
}
