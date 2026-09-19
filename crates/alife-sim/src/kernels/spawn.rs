//! The two kernels a body spawn needs: which particle slots are claimable,
//! and writing a limb's particles into the ones that were claimed.
//!
//! Slot allocation is a prefix scan over an occupancy flag, never an atomic
//! claim (`docs/organism.md`, *Implementation layout*): the ids have to come
//! out ascending and in the same order every run, or the run stops being
//! reproducible. [`crate::genome::slots`] owns the scan; this only builds the
//! flag buffer it scans.

use cubecl::prelude::*;
use cubecl_runtime::server::Handle;

use super::{BodyArgs, Cfg, KIND_BODY, KIND_FREE, body_args, cube_count, whole};
use crate::particles::SphDevice;

/// 1 where a particle slot is taken, 0 where it is claimable. A `Free` slot
/// is the only claimable one.
#[cube(launch)]
pub fn mark_body_occupancy(state: &[u32], flags: &mut [u32], #[comptime] cfg: Cfg) {
  let i = ABSOLUTE_POS;
  if i >= cfg.num_particles as usize {
    terminate!();
  }
  let mut occupied = 1u32;
  if state[i] == KIND_FREE {
    occupied = 0u32;
  }
  flags[i] = occupied;
}

/// Turn claimed slots into body particles. One unit per entry of the list.
///
/// Everything the fluid kernels read is set here, not only the position: a
/// slot that has been free since startup carries whatever `client.empty` gave
/// it in `density`, and `calculate_accel` divides by that.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn place_body_particles(
  sph: &mut BodyArgs,
  ids: &[u32],
  positions: &[f32],
  limb: &[u32],
  index_in_limb: &[u32],
  part_type: &[u32],
  organism: u32,
  count: u32,
) {
  let k = ABSOLUTE_POS;
  if k >= count as usize {
    terminate!();
  }
  let id = ids[k] as usize;
  sph.pos[2 * id] = positions[2 * k];
  sph.pos[2 * id + 1] = positions[2 * k + 1];
  sph.ppos[2 * id] = positions[2 * k];
  sph.ppos[2 * id + 1] = positions[2 * k + 1];
  sph.vel[2 * id] = 0.0f32;
  sph.vel[2 * id + 1] = 0.0f32;
  sph.mass[id] = 1.0f32;
  sph.density[id] = 0.0f32;
  sph.near_density[id] = 0.0f32;
  sph.evap_prob[id] = 0.0f32;
  sph.state[id] = KIND_BODY;
  sph.organism[id] = organism;
  sph.limb[id] = limb[k];
  sph.index_in_limb[id] = index_in_limb[k];
  sph.part_type[id] = part_type[k];
}

/// Fill a per-particle occupancy flag buffer from the particle kinds.
/// Over every slot, not the live prefix: a claim is looking for the free
/// slots above the high-water mark.
pub fn launch_mark_occupancy<R: Runtime>(
  client: &ComputeClient<R>,
  sph: &SphDevice,
  flags: &Handle,
  cfg: Cfg,
) {
  let n = cfg.num_particles as usize;
  mark_body_occupancy::launch::<R>(
    client,
    cube_count(n),
    CubeDim::new_1d(super::CUBE_DIM),
    whole(&sph.state, n),
    whole(flags, n),
    cfg,
  );
}

/// One limb's worth of particles to write.
pub struct Placement {
  pub ids: Vec<u32>,
  /// Interleaved x, y, one pair per id.
  pub positions: Vec<f32>,
  pub limb: Vec<u32>,
  pub index_in_limb: Vec<u32>,
  pub part_type: Vec<u32>,
}

impl Placement {
  pub fn len(&self) -> usize {
    self.ids.len()
  }

  pub fn is_empty(&self) -> bool {
    self.ids.is_empty()
  }
}

pub fn launch_place<R: Runtime>(
  client: &ComputeClient<R>,
  sph: &SphDevice,
  organism: u32,
  placement: &Placement,
) {
  let n = placement.len();
  if n == 0 {
    return;
  }
  let upload = |v: &[u32]| client.create_from_slice(bytemuck::cast_slice(v));
  let ids = upload(&placement.ids);
  let limb = upload(&placement.limb);
  let index_in_limb = upload(&placement.index_in_limb);
  let part_type = upload(&placement.part_type);
  let positions = client.create_from_slice(bytemuck::cast_slice(&placement.positions));

  place_body_particles::launch::<R>(
    client,
    cube_count(n),
    CubeDim::new_1d(super::CUBE_DIM),
    body_args(sph),
    whole(&ids, n),
    whole(&positions, n * 2),
    whole(&limb, n),
    whole(&index_in_limb, n),
    whole(&part_type, n),
    organism,
    n as u32,
  );
}
