//! Limb geometry: what the brain chunk reads about where a limb ended up.
//!
//! `docs/organism.md`'s token features want a limb's position relative to the
//! root and the angle of its first segment, in the body frame — which for a
//! plant is the world frame — plus its depth in the body tree. One entry per
//! `(organism, limb)`, rewritten after every constraint pass, so the brain
//! reads the shape the projection actually produced rather than the one the
//! genome asked for.
//!
//! An absent limb, or a free organism slot, is written as zeros rather than
//! left alone: a stale entry from a dead organism would otherwise be read as
//! a live one by whatever comes next.

use cubecl::prelude::*;
use cubecl_runtime::server::Handle;
use glam::Vec2;

use super::constraints::{seg_dir, wrap_delta, wrap_delta_ref};
use super::{P_BOUNDS_X, cube_count, whole};
use crate::bodies::{BodyCfg, BodyState, LimbGeometryHost, NO_PARTICLE};
use crate::genome::population::{LimbArgs, Population, limb_args};
use crate::particles::{SphDevice, SphHost};

/// One unit per `(organism, limb)`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn write_limb_geometry(
  ppos: &mut [f32],
  limbs: &LimbArgs,
  map: &[u32],
  alive: &[u32],
  rel_pos: &mut [f32],
  angle: &mut [f32],
  depth: &mut [u32],
  params: &[f32],
  #[comptime] cfg: BodyCfg,
) {
  let t = ABSOLUTE_POS as u32;
  if t >= comptime!(cfg.max_organisms * cfg.max_limbs) {
    terminate!();
  }
  let ml = comptime!(cfg.max_limbs);
  let mp = comptime!(cfg.max_particles_per_limb);
  let o = t / ml;
  let l = t % ml;
  let pbase = o * ml * mp;
  let bounds_x = params[P_BOUNDS_X as usize];

  rel_pos[2 * t as usize] = 0.0f32;
  rel_pos[2 * t as usize + 1] = 0.0f32;
  angle[t as usize] = 0.0f32;
  depth[t as usize] = 0u32;

  if alive[o as usize] == 0u32 || limbs.part_type[t as usize] == 0u32 {
    terminate!();
  }

  let first = map[(pbase + l * mp) as usize];
  let root_first = map[pbase as usize];
  if first == NO_PARTICLE || root_first == NO_PARTICLE {
    terminate!();
  }

  rel_pos[2 * t as usize] = wrap_delta(
    ppos[2 * first as usize] - ppos[2 * root_first as usize],
    bounds_x,
  );
  rel_pos[2 * t as usize + 1] = ppos[2 * first as usize + 1] - ppos[2 * root_first as usize + 1];

  // The limb's first segment: from the parent's last particle for a child,
  // `p0 -> p1` for the root — the same chain `kernels::constraints` projects.
  // A limb whose segment has no length keeps the grow angle it was built at.
  let parent = limbs.parent[t as usize];
  let mut u = NO_PARTICLE.runtime();
  let mut v = NO_PARTICLE.runtime();
  if parent != l {
    let mut n = limbs.length[(o * ml + parent) as usize];
    if n > mp {
      n = mp;
    }
    if limbs.part_type[(o * ml + parent) as usize] != 0u32 && n > 0u32 {
      u = map[(pbase + parent * mp + n - 1u32) as usize];
      v = first;
    }
  } else {
    let mut n = limbs.length[t as usize];
    if n > mp {
      n = mp;
    }
    if n > 1u32 {
      u = first;
      v = map[(pbase + l * mp + 1u32) as usize];
    }
  }

  let mut a = limbs.grow_angle[t as usize];
  if u != NO_PARTICLE && v != NO_PARTICLE {
    let d = seg_dir(ppos, u as usize, v as usize, bounds_x);
    if d.x != 0.0f32 || d.y != 0.0f32 {
      a = f32::atan2(d.y, d.x);
    }
  }
  angle[t as usize] = a;

  // Hops to the root, which is the limb that is its own parent. Bounded by
  // the limb count, so a cycle in a corrupt genome cannot hang a unit.
  let mut walk = l;
  let mut hops = 0u32;
  for _ in 0..ml {
    let p = limbs.parent[(o * ml + walk) as usize];
    if p != walk {
      walk = p;
      hops += 1u32;
    }
  }
  depth[t as usize] = hops;
}

pub fn launch<R: Runtime>(
  client: &ComputeClient<R>,
  sph: &SphDevice,
  bodies: &BodyState,
  pop: &Population,
  params: &Handle,
  cfg: BodyCfg,
  num_particles: usize,
) {
  let limbs = cfg.limb_count();
  write_limb_geometry::launch::<R>(
    client,
    cube_count(limbs),
    CubeDim::new_1d(super::CUBE_DIM),
    whole(&sph.ppos, num_particles * 2),
    limb_args(&pop.device.limbs),
    whole(&bodies.device.limb_particles, cfg.particle_map_len()),
    whole(&pop.device.organisms.alive, cfg.max_organisms as usize),
    whole(&bodies.device.geometry.rel_pos, limbs * 2),
    whole(&bodies.device.geometry.angle, limbs),
    whole(&bodies.device.geometry.depth, limbs),
    whole(params, super::PARAM_COUNT),
    cfg,
  );
}

/// Plain-Rust twin of [`write_limb_geometry`].
pub fn write_limb_geometry_ref(
  particles: &SphHost,
  pop: &Population,
  bodies: &BodyState,
  geom: &crate::world::WorldGeometry,
) -> LimbGeometryHost {
  let cfg = bodies.cfg;
  let (ml, mp) = (cfg.max_limbs as usize, cfg.max_particles_per_limb as usize);
  let mut out = LimbGeometryHost::new(cfg.limb_count());
  let bounds_x = geom.bounds.x;

  for o in 0..cfg.max_organisms as usize {
    for l in 0..ml {
      let t = o * ml + l;
      if pop.organisms.alive[o] == 0 || !pop.limbs.part_type[t].is_present() {
        continue;
      }
      let first = bodies.particle(o, l, 0);
      let root_first = bodies.particle(o, 0, 0);
      if first == NO_PARTICLE || root_first == NO_PARTICLE {
        continue;
      }
      let (first, root_first) = (first as usize, root_first as usize);
      out.rel_pos[t] = Vec2::new(
        wrap_delta_ref(
          particles.ppos[first].x - particles.ppos[root_first].x,
          bounds_x,
        ),
        particles.ppos[first].y - particles.ppos[root_first].y,
      );

      let parent = pop.limbs.parent[t] as usize;
      let segment = if parent != l {
        let precord = o * ml + parent;
        let n = (pop.limbs.length[precord] as usize).min(mp);
        (pop.limbs.part_type[precord].is_present() && n > 0)
          .then(|| (bodies.particle(o, parent, n - 1), first as u32))
      } else {
        let n = (pop.limbs.length[t] as usize).min(mp);
        (n > 1).then(|| (first as u32, bodies.particle(o, l, 1)))
      };

      out.angle[t] = pop.limbs.grow_angle[t];
      if let Some((u, v)) = segment.filter(|(u, v)| *u != NO_PARTICLE && *v != NO_PARTICLE) {
        let (u, v) = (u as usize, v as usize);
        let d = Vec2::new(
          wrap_delta_ref(particles.ppos[v].x - particles.ppos[u].x, bounds_x),
          particles.ppos[v].y - particles.ppos[u].y,
        );
        if d.length() > 1e-6 {
          let d = d.normalize();
          out.angle[t] = d.y.atan2(d.x);
        }
      }

      let mut walk = l;
      let mut hops = 0;
      for _ in 0..ml {
        let p = pop.limbs.parent[o * ml + walk] as usize;
        if p != walk {
          walk = p;
          hops += 1;
        }
      }
      out.depth[t] = hops;
    }
  }
  out
}

#[cfg(test)]
mod tests {
  use super::super::test_support::OrganismHarness;
  use crate::bodies::LimbGeometryHost;
  use crate::genome::PartType;

  fn read(h: &OrganismHarness) -> LimbGeometryHost {
    super::launch(
      &h.client,
      &h.sph,
      &h.bodies,
      &h.pop,
      &h.params_buf,
      h.body_cfg,
      h.geom.num_particles,
    );
    h.bodies.device.geometry.download(&h.client)
  }

  #[test]
  fn matches_reference() {
    let h = OrganismHarness::new();
    let actual = read(&h);
    let expected = super::write_limb_geometry_ref(&h.particles, &h.pop, &h.bodies, &h.geom);

    for t in 0..actual.len() {
      assert!(
        (actual.rel_pos[t] - expected.rel_pos[t])
          .abs()
          .max_element()
          < 1e-6,
        "limb {t} rel_pos: {:?} vs {:?}",
        actual.rel_pos[t],
        expected.rel_pos[t]
      );
      assert!(
        (actual.angle[t] - expected.angle[t]).abs() < 1e-6,
        "limb {t} angle: {} vs {}",
        actual.angle[t],
        expected.angle[t]
      );
      assert_eq!(actual.depth[t], expected.depth[t], "limb {t} depth");
    }
  }

  /// The seed plant is root -> stem -> leaf, so the depths are 0, 1, 2 and
  /// the root sits at the origin of its own frame.
  #[test]
  fn depth_and_rel_pos_describe_the_seed_plant() {
    let h = OrganismHarness::new();
    let g = read(&h);
    let ml = h.body_cfg.max_limbs as usize;

    assert_eq!(h.pop.limbs.part_type[0], PartType::Root);
    assert_eq!(g.depth[0], 0);
    assert_eq!(g.rel_pos[0], glam::Vec2::ZERO);
    assert_eq!(g.depth[1], 1, "the stem hangs off the root");
    assert_eq!(g.depth[2], 2, "the leaf hangs off the stem");
    assert_ne!(g.rel_pos[1], glam::Vec2::ZERO);

    // Absent records and the free organism slot stay zeroed.
    for t in 3..ml {
      assert_eq!(g.depth[t], 0);
      assert_eq!(g.angle[t], 0.0);
    }
    for t in ml..g.len() {
      assert_eq!(g.rel_pos[t], glam::Vec2::ZERO);
    }
  }
}
