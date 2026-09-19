//! One token per limb: the raw feature vector and its projection into token
//! space.
//!
//! `docs/organism.md` fixes the feature order — spatial (2), rotation as
//! cos/sin (2), depth (1), child slot (1), identity (`IDENTITY_DIM`), sensors
//! (`SENSOR_DIM`) — and the projection: `type_embed[part_type] + tok_proj ·
//! features + tok_bias`.
//!
//! The features are assembled **inline**, inside [`write_tokens`], rather than
//! materialised into a `[max_organisms x max_limbs x FEATURE_DIM]` buffer of
//! their own. One unit is one `(organism, limb, token component)`, so a
//! limb's `d_token` units each rebuild its 15 features; that is two cheap
//! trigonometric calls and a dozen loads per unit against a launch and a
//! buffer, and it keeps the feature order in exactly one place on the device
//! side. [`token_features_ref`] is the host twin, and the only place a reader
//! needs to look for the order.

use cubecl::prelude::*;

use super::{BrainCfg, Weights};
use crate::bodies::{BodyState, LimbGeometryHost};
use crate::genome::Population;
use crate::genome::shape::{BrainShape, FEATURE_DIM, IDENTITY_DIM, SENSOR_DIM};
use crate::kernels::{CUBE_DIM, cube_count, whole};

/// One unit per `(organism, limb, token component)`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn write_tokens<W: Float>(
  brain: &[W],
  part_type: &[u32],
  child_slot: &[u32],
  identity: &[f32],
  rel_pos: &[f32],
  angle: &[f32],
  depth: &[u32],
  sensors: &[f32],
  alive: &[u32],
  tokens: &mut [f32],
  type_embed_off: u32,
  tok_proj_off: u32,
  tok_bias_off: u32,
  #[comptime] cfg: BrainCfg,
) {
  let t = ABSOLUTE_POS as u32;
  if t >= comptime!(cfg.max_organisms * cfg.max_limbs * cfg.d_token) {
    terminate!();
  }
  let dt = comptime!(cfg.d_token);
  let c = t % dt;
  // The per-limb buffers are indexed `organism * max_limbs + limb`, which is
  // exactly this quotient.
  let li = t / dt;
  let o = li / comptime!(cfg.max_limbs);

  // An absent limb is a zero token, and is masked out of the input attention
  // (`super::forward::attn_attend`). A free organism slot is zeroed too, so
  // nothing downstream reads a dead organism's last tick.
  if alive[o as usize] == 0u32 || part_type[li as usize] == 0u32 {
    tokens[t as usize] = 0.0f32;
    terminate!();
  }

  let row = o * comptime!(cfg.param_count);
  let proj = row + tok_proj_off;
  let pt = part_type[li as usize];

  let mut acc = f32::cast_from(brain[(row + type_embed_off + pt * dt + c) as usize]);
  acc += f32::cast_from(brain[(row + tok_bias_off + c) as usize]);

  // Feature order, spelled out: it is the spec's, and `token_features_ref` is
  // the host twin of these same lines.
  acc += rel_pos[2 * li as usize] * f32::cast_from(brain[(proj + c) as usize]);
  acc += rel_pos[2 * li as usize + 1] * f32::cast_from(brain[(proj + dt + c) as usize]);

  let a = angle[li as usize];
  acc += f32::cos(a) * f32::cast_from(brain[(proj + 2u32 * dt + c) as usize]);
  acc += f32::sin(a) * f32::cast_from(brain[(proj + 3u32 * dt + c) as usize]);
  acc += depth[li as usize] as f32 * f32::cast_from(brain[(proj + 4u32 * dt + c) as usize]);
  acc += child_slot[li as usize] as f32 * f32::cast_from(brain[(proj + 5u32 * dt + c) as usize]);

  let id_dim = comptime!(cfg.identity_dim);
  for k in 0..id_dim {
    let w = brain[(proj + (6u32 + k) * dt + c) as usize];
    acc += identity[(li * id_dim + k) as usize] * f32::cast_from(w);
  }
  let sensor_base = comptime!(6u32 + cfg.identity_dim);
  for k in 0..comptime!(cfg.sensor_dim) {
    let w = brain[(proj + (sensor_base + k) * dt + c) as usize];
    acc += sensors[(li * comptime!(cfg.sensor_dim) + k) as usize] * f32::cast_from(w);
  }

  tokens[t as usize] = acc;
}

#[allow(clippy::too_many_arguments)]
pub fn launch<R: Runtime>(
  client: &ComputeClient<R>,
  weights: &Weights,
  bodies: &BodyState,
  pop: &Population,
  brain: &super::BrainDevice,
  shape: &BrainShape,
  cfg: BrainCfg,
) {
  let limbs = cfg.limbs();
  let count = cube_count(cfg.tokens_len());
  let dim = CubeDim::new_1d(CUBE_DIM);
  let type_embed = shape.type_embed().start as u32;
  let tok_proj = shape.tok_proj().start as u32;
  let tok_bias = shape.tok_bias().start as u32;

  macro_rules! go {
    ($elem:ty, $handle:expr) => {
      write_tokens::launch::<$elem, R>(
        client,
        count,
        dim,
        whole(
          $handle,
          cfg.max_organisms as usize * cfg.param_count as usize,
        ),
        whole(&pop.device.limbs.part_type, limbs),
        whole(&pop.device.limbs.child_slot, limbs),
        whole(&pop.device.limbs.identity, limbs * IDENTITY_DIM),
        whole(&bodies.device.geometry.rel_pos, limbs * 2),
        whole(&bodies.device.geometry.angle, limbs),
        whole(&bodies.device.geometry.depth, limbs),
        whole(&brain.sensors, cfg.sensors_len()),
        whole(&pop.device.organisms.alive, cfg.max_organisms as usize),
        whole(&brain.tokens, cfg.tokens_len()),
        type_embed,
        tok_proj,
        tok_bias,
        cfg,
      )
    };
  }
  match weights {
    Weights::F32(handle) => go!(f32, handle),
    Weights::F16(handle) => go!(half::f16, handle),
  }
}

/// What one limb contributes to its token before the projection: the
/// `FEATURE_DIM` vector, in the spec's order.
///
/// `sensors` is that limb's slice of the sensor buffer.
pub fn token_features_ref(
  rel_pos: glam::Vec2,
  angle: f32,
  depth: u32,
  child_slot: u8,
  identity: &[f32; IDENTITY_DIM],
  sensors: &[f32],
) -> [f32; FEATURE_DIM] {
  let mut out = [0.0f32; FEATURE_DIM];
  out[0] = rel_pos.x;
  out[1] = rel_pos.y;
  out[2] = angle.cos();
  out[3] = angle.sin();
  out[4] = depth as f32;
  out[5] = child_slot as f32;
  out[6..6 + IDENTITY_DIM].copy_from_slice(identity);
  out[6 + IDENTITY_DIM..6 + IDENTITY_DIM + SENSOR_DIM].copy_from_slice(&sensors[..SENSOR_DIM]);
  out
}

/// One limb's token: `type_embed[part_type] + tok_proj · features +
/// tok_bias`.
///
/// Accumulated in the kernel's order — embedding, bias, then feature by
/// feature — because a different association rounds differently.
pub fn token_ref(
  shape: &BrainShape,
  brain: &[f32],
  part_type: u8,
  features: &[f32; FEATURE_DIM],
) -> Vec<f32> {
  let dt = shape.d_token;
  let embed = shape.type_embed().start + part_type as usize * dt;
  let proj = shape.tok_proj().start;
  let bias = shape.tok_bias().start;
  (0..dt)
    .map(|c| {
      let mut acc = brain[embed + c] + brain[bias + c];
      for (f, value) in features.iter().enumerate() {
        acc += value * brain[proj + f * dt + c];
      }
      acc
    })
    .collect()
}

/// Plain-Rust twin of [`write_tokens`], over the whole population.
///
/// `geometry` is what [`crate::kernels::limb_geometry`] published this step,
/// read back to the host.
pub fn write_tokens_ref(
  pop: &Population,
  geometry: &LimbGeometryHost,
  sensors: &[f32],
  shape: &BrainShape,
) -> Vec<f32> {
  let ml = pop.max_limbs;
  let dt = shape.d_token;
  let mut out = vec![0.0f32; pop.max_organisms * ml * dt];

  for o in 0..pop.max_organisms {
    if pop.organisms.alive[o] == 0 {
      continue;
    }
    for l in 0..ml {
      let li = o * ml + l;
      let part_type = pop.limbs.part_type[li];
      if !part_type.is_present() {
        continue;
      }
      let features = token_features_ref(
        geometry.rel_pos[li],
        geometry.angle[li],
        geometry.depth[li],
        pop.limbs.child_slot[li],
        &pop.limbs.identity[li],
        &sensors[li * SENSOR_DIM..],
      );
      let token = token_ref(shape, pop.brain_row(o), part_type as u8, &features);
      out[li * dt..(li + 1) * dt].copy_from_slice(&token);
    }
  }
  out
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::kernels::test_support::{OrganismHarness, assert_close};

  #[test]
  fn features_are_in_the_spec_order() {
    let f = token_features_ref(
      glam::Vec2::new(0.25, -0.5),
      0.0,
      3,
      2,
      &[1.0, 2.0, 3.0, 4.0],
      &[10.0, 11.0, 12.0, 13.0, 14.0],
    );
    assert_eq!(f.len(), FEATURE_DIM);
    assert_eq!(&f[..6], &[0.25, -0.5, 1.0, 0.0, 3.0, 2.0]);
    assert_eq!(&f[6..10], &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(&f[10..], &[10.0, 11.0, 12.0, 13.0, 14.0]);
  }

  #[test]
  fn matches_reference() {
    let h = OrganismHarness::new();
    h.run_sense();
    h.run_tokens();

    let actual = h.brain.read_tokens(&h.client);
    let sensors = h.brain.read_sensors(&h.client);
    let expected = write_tokens_ref(&h.pop, &h.bodies.geometry, &sensors, &h.pop.shape);

    assert_eq!(actual.len(), expected.len());
    assert_close(&actual, &expected, 1e-5, "tokens");
  }

  /// An absent limb is a zero token — the input attention masks it out, and a
  /// non-zero one would still leak through the output attention's queries.
  #[test]
  fn absent_limbs_get_a_zero_token() {
    let h = OrganismHarness::new();
    h.run_sense();
    h.run_tokens();
    let tokens = h.brain.read_tokens(&h.client);
    let dt = h.pop.shape.d_token;

    for li in 0..h.brain.cfg.limbs() {
      let present =
        h.pop.limbs.part_type[li].is_present() && h.pop.organisms.alive[li / h.pop.max_limbs] == 1;
      let all_zero = tokens[li * dt..(li + 1) * dt].iter().all(|v| *v == 0.0);
      assert_eq!(!all_zero, present, "limb {li}");
    }
  }
}
