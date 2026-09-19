//! The brain: one batched Perceiver-IO tick per step, over every alive
//! organism.
//!
//! Built from `docs/organism.md`'s *Brain* section and the `BrainShape` slice
//! table in *Implementation layout*. [`crate::genome::shape`] owns that table;
//! nothing here writes an offset down, it asks the shape for the slice.
//!
//! One tick, in the spec's order: sensors from the world ([`sense`]), a token
//! per limb ([`tokens`]), then input cross-attention, latent self-attention,
//! the MLP trunk, the gated latent update, output cross-attention and the
//! per-limb heads ([`forward`]). The latents persist across ticks; everything
//! else is scratch.
//!
//! **Why this many launches.** Kernels stay barrier-free (`docs/organism.md`,
//! decisions, 2026-09-19), so every point where the pass has to reduce across
//! units — an attention row's softmax denominator, a projection that needs a
//! whole vector rather than one of its components — is a launch boundary
//! instead of a `sync_cube()`. [`forward::LAUNCHES`] is the count and
//! `docs/perf.md` has what each one costs. The one reduction that is *not* a
//! launch boundary is the attention softmax: [`forward::attn_attend`]
//! recomputes it per output component rather than materialising the
//! probabilities, which is `keys` extra exponentials per unit and saves three
//! launches.
//!
//! Nothing here applies a head. The life-cycle chunk reads the sprout logits
//! and the actuator targets out of [`BrainDevice::heads`].

use cubecl::prelude::*;
use cubecl_runtime::server::Handle;

use crate::SimParams;
use crate::genome::shape::{BrainShape, FEATURE_DIM, IDENTITY_DIM, N_TYPES, SENSOR_DIM};

/// Which device copy of the brain tensor the kernels read. It belongs to the
/// population that owns the tensor; every kernel here is generic over it.
pub use crate::genome::population::Weights;

pub mod sense;
pub mod tokens;

/// Outputs per limb: the sprout logits over child types, then the two
/// actuator outputs (target angle and one reserved).
pub const HEAD_DIM: usize = N_TYPES + 2;

/// The shapes the brain kernels bake in. Comptime configuration, as
/// [`crate::bodies::BodyCfg`] is for the body passes.
///
/// Brain *slice offsets* deliberately stay out of it: they are runtime `u32`
/// arguments, so one compiled kernel serves every launch that shares a shape
/// — the six `d_latent x d_latent` projections, say — instead of one compiled
/// kernel per slice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BrainCfg {
  pub max_organisms: u32,
  pub max_limbs: u32,
  pub d_token: u32,
  pub d_latent: u32,
  pub n_latents: u32,
  pub trunk_hidden: u32,
  pub n_types: u32,
  pub feature_dim: u32,
  pub sensor_dim: u32,
  pub identity_dim: u32,
  pub head_dim: u32,
  pub param_count: u32,
}

impl BrainCfg {
  pub fn new(shape: &BrainShape, params: &SimParams) -> Self {
    Self {
      max_organisms: params.max_organisms.max(1) as u32,
      max_limbs: params.max_limbs.max(1) as u32,
      d_token: shape.d_token as u32,
      d_latent: shape.d_latent as u32,
      n_latents: shape.n_latents as u32,
      trunk_hidden: shape.trunk_hidden as u32,
      n_types: N_TYPES as u32,
      feature_dim: FEATURE_DIM as u32,
      sensor_dim: SENSOR_DIM as u32,
      identity_dim: IDENTITY_DIM as u32,
      head_dim: HEAD_DIM as u32,
      param_count: shape.param_count() as u32,
    }
  }

  fn organisms(&self) -> usize {
    self.max_organisms as usize
  }

  /// Entries in a per-`(organism, limb)` buffer.
  pub fn limbs(&self) -> usize {
    (self.max_organisms * self.max_limbs) as usize
  }

  pub fn sensors_len(&self) -> usize {
    self.limbs() * self.sensor_dim as usize
  }

  pub fn tokens_len(&self) -> usize {
    self.limbs() * self.d_token as usize
  }

  /// `[max_organisms x n_latents x d_latent]`, the shape of every latent-side
  /// intermediate.
  pub fn latent_len(&self) -> usize {
    self.organisms() * (self.n_latents * self.d_latent) as usize
  }

  /// `[max_organisms x max_limbs x d_latent]`, the shape of every token-side
  /// intermediate.
  pub fn limb_latent_len(&self) -> usize {
    self.limbs() * self.d_latent as usize
  }

  pub fn hidden_len(&self) -> usize {
    self.organisms() * (self.n_latents * self.trunk_hidden) as usize
  }

  pub fn heads_len(&self) -> usize {
    self.limbs() * self.head_dim as usize
  }

  /// Long enough to stand in for any projection's residual term: the widest
  /// `[max_organisms x rows x n]` any launch asks for.
  pub fn residual_pad_len(&self) -> usize {
    self
      .latent_len()
      .max(self.limb_latent_len())
      .max(self.hidden_len())
      .max(self.tokens_len())
  }
}

/// Scratch for one tick, plus the outputs the life-cycle chunk will read.
///
/// Every buffer is allocated zeroed and only ever written for an alive
/// organism, so a free slot reads as zeros rather than as whatever the
/// allocator handed back.
#[derive(Debug, Clone)]
pub struct BrainDevice {
  /// `[max_organisms x max_limbs x SENSOR_DIM]`.
  pub sensors: Handle,
  /// `[max_organisms x max_limbs x d_token]`.
  pub tokens: Handle,
  /// Input cross-attention: queries from the latents, keys and values from
  /// the tokens. `in_kv` holds both jobs, keys first.
  pub in_q: Handle,
  pub in_kv: Handle,
  /// `[max_organisms x n_latents x max_limbs]`.
  pub in_scores: Handle,
  pub in_att: Handle,
  /// The latents after the input cross-attention's residual add.
  pub h1: Handle,
  /// Latent self-attention. Queries stay in a buffer of their own: a scores
  /// launch binds its queries and its keys as two arguments, so they cannot
  /// be two jobs of one buffer.
  pub self_q: Handle,
  pub self_kv: Handle,
  /// `[max_organisms x n_latents x n_latents]`.
  pub self_scores: Handle,
  pub self_att: Handle,
  /// The latents after the self-attention's residual add.
  pub h2: Handle,
  /// `[max_organisms x n_latents x trunk_hidden]`.
  pub hidden: Handle,
  /// The trunk's output: the candidate the gate blends into the latents.
  pub trunk: Handle,
  /// Output cross-attention: queries from the tokens, keys and values from
  /// the updated latents. `out_kv` holds both jobs, keys first.
  pub out_q: Handle,
  pub out_kv: Handle,
  /// `[max_organisms x max_limbs x n_latents]`.
  pub out_scores: Handle,
  pub out_att: Handle,
  /// The attended token, projected back to `d_token`: what the heads read.
  pub z: Handle,
  /// `[max_organisms x max_limbs x HEAD_DIM]`: the sprout logits, then the
  /// two actuator outputs.
  pub heads: Handle,
  /// Zeros, long enough for any projection's residual term. Read where the
  /// spec has no residual, so [`forward::gemv`] can add one unconditionally
  /// (`forward`, *Why the residual is always read*). Never written.
  pub zeros: Handle,
}

pub struct BrainState {
  pub cfg: BrainCfg,
  pub device: BrainDevice,
}

impl BrainState {
  pub fn new<R: Runtime>(client: &ComputeClient<R>, cfg: BrainCfg) -> Self {
    let zeros = |n: usize| client.create_from_slice(bytemuck::cast_slice(&vec![0.0f32; n]));
    let latent = cfg.latent_len();
    let device = BrainDevice {
      sensors: zeros(cfg.sensors_len()),
      tokens: zeros(cfg.tokens_len()),
      in_q: zeros(latent),
      in_kv: zeros(2 * cfg.limb_latent_len()),
      in_scores: zeros(cfg.organisms() * (cfg.n_latents * cfg.max_limbs) as usize),
      in_att: zeros(latent),
      h1: zeros(latent),
      self_q: zeros(latent),
      self_kv: zeros(2 * latent),
      self_scores: zeros(cfg.organisms() * (cfg.n_latents * cfg.n_latents) as usize),
      self_att: zeros(latent),
      h2: zeros(latent),
      hidden: zeros(cfg.hidden_len()),
      trunk: zeros(latent),
      out_q: zeros(cfg.limb_latent_len()),
      out_kv: zeros(2 * latent),
      out_scores: zeros(cfg.organisms() * (cfg.max_limbs * cfg.n_latents) as usize),
      out_att: zeros(cfg.limb_latent_len()),
      z: zeros(cfg.tokens_len()),
      heads: zeros(cfg.heads_len()),
      zeros: zeros(cfg.residual_pad_len()),
    };
    Self { cfg, device }
  }

  /// The per-limb outputs, host side: `HEAD_DIM` floats per
  /// `(organism, limb)`.
  pub fn read_heads<R: Runtime>(&self, client: &ComputeClient<R>) -> Vec<f32> {
    crate::genome::population::read_f32(client, &self.device.heads, self.cfg.heads_len())
  }

  /// The sensor buffer, host side.
  pub fn read_sensors<R: Runtime>(&self, client: &ComputeClient<R>) -> Vec<f32> {
    crate::genome::population::read_f32(client, &self.device.sensors, self.cfg.sensors_len())
  }

  /// The token buffer, host side.
  pub fn read_tokens<R: Runtime>(&self, client: &ComputeClient<R>) -> Vec<f32> {
    crate::genome::population::read_f32(client, &self.device.tokens, self.cfg.tokens_len())
  }
}

// --- Activations, shared by the kernels and their references ---

/// Guarded against overflow rather than accurate at the tails: a large
/// negative `x` makes `exp(-x)` infinite and the quotient zero, which is the
/// limit, where `exp(x) / (1 + exp(x))` would be `inf / inf`.
#[cube]
pub fn sigmoid(x: f32) -> f32 {
  1.0f32 / (1.0f32 + f32::exp(-x))
}

/// SiLU, `x * sigmoid(x)`: the trunk's activation.
///
/// GELU was the other option the spec left open. SiLU is the same shape to
/// within a few percent and costs one `exp` where an exact GELU costs an
/// `erf` — and the gate already needs a sigmoid, so the backends only have to
/// agree about one transcendental in the whole pass.
#[cube]
pub fn silu(x: f32) -> f32 {
  x * sigmoid(x)
}

pub fn sigmoid_ref(x: f32) -> f32 {
  1.0 / (1.0 + (-x).exp())
}

pub fn silu_ref(x: f32) -> f32 {
  x * sigmoid_ref(x)
}
