//! The fixed shape of one organism's brain tensor.
//!
//! Every organism in the population carries the same number of brain
//! parameters regardless of its body, so the population is one dense
//! `[max_organisms x param_count()]` tensor and a species mean or a lineage
//! delta is a plain tensor operation (`docs/organism.md`, *Genome*).
//!
//! This module owns the layout of a single row: the ordered table of named
//! slices in `docs/organism.md`'s *Implementation layout*. The genome chunk
//! initializes and mutates that row and the brain chunk reads it, so the slice
//! accessors here are the one place the offsets are written down.

use core::ops::Range;

use crate::SimParams;

/// Part-type codes a genome can use, and with it the height of the brain's
/// embedding table. 0 is "absent"; 1–4 are root, stem, leaf and seed; 5–7 are
/// reserved for the later types (actuated limb, digger, mouth).
pub const N_TYPES: usize = 8;

/// Floats in a limb record's identity vector.
pub const IDENTITY_DIM: usize = 4;

/// Live sensor readings per limb: light, water, soil solid fraction, contact,
/// energy.
pub const SENSOR_DIM: usize = 5;

/// A token's raw feature vector before the projection: spatial position
/// relative to the root (2), rotation against the parent as cos/sin (2),
/// depth (1), child slot (1), identity (4), sensors (5).
pub const FEATURE_DIM: usize = 2 + 2 + 1 + 1 + IDENTITY_DIM + SENSOR_DIM;

/// How a slice's initial values are drawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitKind {
  /// Gaussian with sigma `1 / sqrt(fan_in)`, where fan-in is the slice's row
  /// count — every weight slice is stored `[input dim x output dim]`.
  Weight,
  /// Zero. A bias is stored as a `1 x n` slice, so its row count is not a
  /// meaningful fan-in.
  Bias,
  /// Gaussian with sigma 1: the persistent latents start as unit-scale state,
  /// not as a projection of anything.
  Unit,
}

/// One named slice of the flat parameter row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SliceSpec {
  pub name: &'static str,
  pub rows: usize,
  pub cols: usize,
  pub init: InitKind,
  pub range: Range<usize>,
}

impl SliceSpec {
  /// Standard deviation of this slice's initial values.
  pub fn sigma(&self) -> f32 {
    match self.init {
      InitKind::Weight => 1.0 / (self.rows as f32).sqrt(),
      InitKind::Bias => 0.0,
      InitKind::Unit => 1.0,
    }
  }
}

/// The dimensions every organism's brain shares.
///
/// `Eq`/`Hash` because it is comptime kernel configuration on the brain side,
/// the same way [`crate::world::Cfg`] is on the fluid side.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BrainShape {
  pub d_token: usize,
  pub d_latent: usize,
  pub n_latents: usize,
  pub trunk_hidden: usize,
}

impl BrainShape {
  pub fn from_params(params: &SimParams) -> Self {
    debug_assert!(params.validate().is_ok(), "brain dims must be positive");
    Self {
      d_token: params.brain_d_token.max(1) as usize,
      d_latent: params.brain_d_latent.max(1) as usize,
      n_latents: params.brain_n_latents.max(1) as usize,
      trunk_hidden: params.brain_trunk_hidden.max(1) as usize,
    }
  }

  /// Floats of persistent latent state per organism. State, not genome — it
  /// lives in its own buffer and is seeded from [`Self::latent_init`].
  pub fn latent_state_len(&self) -> usize {
    self.n_latents * self.d_latent
  }
}

impl Default for BrainShape {
  fn default() -> Self {
    Self::from_params(&SimParams::default())
  }
}

/// The slice table from `docs/organism.md`, in its order.
///
/// Each line gives the slice's shape as `(rows, cols)` and how it is
/// initialized; the macro chains them so that each slice starts where the
/// previous one ends, and `param_count()` is the last slice's end. Add a slice
/// by adding a line — never by writing an offset down anywhere else.
macro_rules! brain_slices {
    ( $( $(#[$meta:meta])* $name:ident = |$s:ident| ($rows:expr, $cols:expr), $init:ident ; )* ) => {
        brain_slices!(@chain first, $( $(#[$meta])* $name = |$s| ($rows, $cols), $init ; )*);

        impl BrainShape {
            /// Every slice, in table order, with its shape and range.
            pub fn slices(&self) -> Vec<SliceSpec> {
                vec![ $( {
                    #[allow(unused_variables)]
                    let $s = self;
                    SliceSpec {
                        name: stringify!($name),
                        rows: $rows,
                        cols: $cols,
                        init: InitKind::$init,
                        range: self.$name(),
                    }
                } ),* ]
            }
        }
    };

    (@chain $prev:tt, $(#[$meta:meta])* $name:ident = |$s:ident| ($rows:expr, $cols:expr), $init:ident ; $($rest:tt)*) => {
        impl BrainShape {
            $(#[$meta])*
            pub fn $name(&self) -> Range<usize> {
                let start = brain_slices!(@start self, $prev);
                #[allow(unused_variables)]
                let $s = self;
                start..start + ($rows) * ($cols)
            }
        }
        brain_slices!(@chain ($name), $($rest)*);
    };

    (@chain $prev:tt,) => {
        impl BrainShape {
            /// Floats in one organism's brain row. 18,890 at the defaults.
            pub fn param_count(&self) -> usize {
                brain_slices!(@start self, $prev)
            }
        }
    };

    (@start $self:ident, first) => { 0usize };
    (@start $self:ident, ($prev:ident)) => { $self.$prev().end };
}

brain_slices! {
    /// Per-part-type embedding, added to every token of that type.
    type_embed = |s| (N_TYPES, s.d_token), Weight;
    /// Projection of a token's raw features into token space.
    tok_proj = |s| (FEATURE_DIM, s.d_token), Weight;
    tok_bias = |s| (1, s.d_token), Bias;
    /// Input cross-attention, latents attending to tokens.
    in_q = |s| (s.d_latent, s.d_latent), Weight;
    in_k = |s| (s.d_token, s.d_latent), Weight;
    in_v = |s| (s.d_token, s.d_latent), Weight;
    in_o = |s| (s.d_latent, s.d_latent), Weight;
    /// Latent self-attention.
    self_q = |s| (s.d_latent, s.d_latent), Weight;
    self_k = |s| (s.d_latent, s.d_latent), Weight;
    self_v = |s| (s.d_latent, s.d_latent), Weight;
    self_o = |s| (s.d_latent, s.d_latent), Weight;
    /// The fixed trunk over the latents.
    mlp_w1 = |s| (s.d_latent, s.trunk_hidden), Weight;
    mlp_b1 = |s| (1, s.trunk_hidden), Bias;
    mlp_w2 = |s| (s.trunk_hidden, s.d_latent), Weight;
    mlp_b2 = |s| (1, s.d_latent), Bias;
    /// Gate on the persistent latent update.
    gate_w = |s| (s.d_latent, s.d_latent), Weight;
    gate_b = |s| (1, s.d_latent), Bias;
    /// Output cross-attention, tokens attending to latents.
    out_q = |s| (s.d_token, s.d_latent), Weight;
    out_k = |s| (s.d_latent, s.d_latent), Weight;
    out_v = |s| (s.d_latent, s.d_latent), Weight;
    out_o = |s| (s.d_latent, s.d_token), Weight;
    /// Logits over the child type to sprout; index 0 is "none".
    head_sprout = |s| (s.d_token, N_TYPES), Weight;
    sprout_b = |s| (1, N_TYPES), Bias;
    /// Target joint angle, plus one reserved output.
    head_actuator = |s| (s.d_token, 2), Weight;
    actuator_b = |s| (1, 2), Bias;
    /// Initial value of the persistent latents; copied into latent state at
    /// birth, so it is genome rather than state.
    latent_init = |s| (s.n_latents, s.d_latent), Unit;
}

#[cfg(test)]
mod tests {
  use super::*;

  /// The table in `docs/organism.md`, written out independently of the macro
  /// above: `(name, count)` at the default dimensions. If the two disagree,
  /// one of them is what a reader of the doc would get.
  const EXPECTED: &[(&str, usize)] = &[
    ("type_embed", 8 * 32),
    ("tok_proj", 15 * 32),
    ("tok_bias", 32),
    ("in_q", 32 * 32),
    ("in_k", 32 * 32),
    ("in_v", 32 * 32),
    ("in_o", 32 * 32),
    ("self_q", 32 * 32),
    ("self_k", 32 * 32),
    ("self_v", 32 * 32),
    ("self_o", 32 * 32),
    ("mlp_w1", 32 * 64),
    ("mlp_b1", 64),
    ("mlp_w2", 64 * 32),
    ("mlp_b2", 32),
    ("gate_w", 32 * 32),
    ("gate_b", 32),
    ("out_q", 32 * 32),
    ("out_k", 32 * 32),
    ("out_v", 32 * 32),
    ("out_o", 32 * 32),
    ("head_sprout", 32 * 8),
    ("sprout_b", 8),
    ("head_actuator", 32 * 2),
    ("actuator_b", 2),
    ("latent_init", 8 * 32),
  ];

  #[test]
  fn feature_dim_is_fifteen() {
    assert_eq!(FEATURE_DIM, 15);
  }

  #[test]
  fn slices_match_the_spec_table() {
    let shape = BrainShape::default();
    let slices = shape.slices();
    assert_eq!(slices.len(), EXPECTED.len());
    for (slice, (name, count)) in slices.iter().zip(EXPECTED) {
      assert_eq!(slice.name, *name);
      assert_eq!(slice.rows * slice.cols, *count, "{name}");
      assert_eq!(slice.range.len(), *count, "{name}");
    }
  }

  #[test]
  fn param_count_at_the_defaults() {
    assert_eq!(BrainShape::default().param_count(), 18_890);
    assert_eq!(
      BrainShape::default().param_count(),
      EXPECTED.iter().map(|(_, n)| n).sum::<usize>()
    );
  }

  #[test]
  fn slices_are_contiguous_and_cover_the_row() {
    let shape = BrainShape::default();
    let mut next = 0;
    let mut covered = vec![false; shape.param_count()];
    for slice in shape.slices() {
      assert_eq!(slice.range.start, next, "{} is not contiguous", slice.name);
      for i in slice.range.clone() {
        assert!(!covered[i], "{} overlaps an earlier slice", slice.name);
        covered[i] = true;
      }
      next = slice.range.end;
    }
    assert_eq!(next, shape.param_count());
    assert!(covered.iter().all(|c| *c));
  }

  #[test]
  fn slices_stay_contiguous_at_other_dimensions() {
    let shape = BrainShape {
      d_token: 16,
      d_latent: 48,
      n_latents: 3,
      trunk_hidden: 7,
    };
    let mut next = 0;
    for slice in shape.slices() {
      assert_eq!(slice.range.start, next, "{}", slice.name);
      assert_eq!(slice.range.len(), slice.rows * slice.cols, "{}", slice.name);
      next = slice.range.end;
    }
    assert_eq!(next, shape.param_count());
  }

  #[test]
  fn weight_sigma_is_one_over_root_fan_in_and_biases_are_zero() {
    let shape = BrainShape::default();
    let by_name = |name: &str| {
      shape
        .slices()
        .into_iter()
        .find(|s| s.name == name)
        .expect("slice exists")
    };
    assert_eq!(by_name("in_k").sigma(), 1.0 / 32.0f32.sqrt());
    assert_eq!(by_name("tok_proj").sigma(), 1.0 / 15.0f32.sqrt());
    assert_eq!(by_name("tok_bias").sigma(), 0.0);
    assert_eq!(by_name("latent_init").sigma(), 1.0);
  }

  #[test]
  fn latent_state_is_the_latent_init_slice_size() {
    let shape = BrainShape::default();
    assert_eq!(shape.latent_state_len(), shape.latent_init().len());
  }
}
