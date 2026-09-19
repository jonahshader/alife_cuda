//! The body constraint pass: position-based projection of each organism's
//! limbs, one unit per organism, serial over that organism's particles.
//!
//! `docs/organism.md`, *Implementation layout*: after the fluid integrator has
//! moved the body particles like any other liquid, a fixed number of
//! Gauss-Seidel sweeps put them back where the body says they belong. Per
//! sweep, in order:
//!
//! - **pin** — the root limb's first particle sits on the organism's anchor,
//!   the soil cell it germinated in;
//! - **distance** — consecutive particles within a limb, and a limb's first
//!   particle against its parent limb's last, at `limb_segment_length` times
//!   how much of that segment the limb has grown into
//!   ([`segment_fraction`]), which is 1 for every segment but a growing
//!   limb's last;
//! - **base-joint angle** — a limb's first segment points along its parent's
//!   last segment rotated by the limb record's `grow_angle` (the root limb
//!   measures against the world frame, 0 at +x, counter-clockwise). How far
//!   off it was before the sweep pulled it back is kept per `(organism,
//!   limb)` as a running maximum in `BodyState::bend_max`, which the life
//!   tick reads and resets: that is what a limb breaks on
//!   (`docs/organism.md`, *Growth*). One unit owns the organism, so the
//!   maximum needs no atomic;
//! - **bend** — consecutive segments within a limb stay aligned, or a limb is
//!   a rope.
//!
//! **What a "limb's first segment" is.** The base joint of a non-root limb
//! sits at its parent's last particle `B`, so the limb's chain is `B, p0, p1,
//! ...`: its first segment is `B -> p0`, which is what makes a one-particle
//! limb have an orientation at all. The root limb has no `B`, so its chain is
//! `p0, p1, ...` and its first segment is `p0 -> p1`.
//!
//! **Which particle a projection moves.** A distance constraint moves both
//! ends symmetrically, half the error each — the equal-mass PBD projection.
//! An angle constraint moves only the child-side particle: the parent side is
//! either further up the body (already projected this sweep, Gauss-Seidel) or
//! the pinned root, and moving it would fight the joint above it.
//!
//! The world wraps in x, so every difference between two body particles is
//! taken as the minimum image and every write wraps back into the world. A
//! plant that straddles the seam is otherwise torn in half by its own
//! distance constraints.

use cubecl::prelude::*;
use cubecl_runtime::server::Handle;
use glam::Vec2;

use super::{
  Cfg, P_BEND_STIFFNESS, P_BOUNDS_X, P_BOUNDS_Y, P_CONSTRAINT_ITERS, P_DT, P_DT_PREDICT,
  P_JOINT_STIFFNESS, P_LIMB_SEGMENT_LENGTH, cube_count, whole,
};
use crate::bodies::{BodyCfg, BodyState, NO_PARTICLE};
use crate::genome::population::{LimbArgs, Population, limb_args};
use crate::particles::{SphDevice, SphHost};

/// A unit direction, or zero where there is no direction to be had.
#[derive(CubeType, Clone, Copy)]
pub struct Dir {
  pub x: f32,
  pub y: f32,
}

/// Shortest signed difference in x across the world's seam.
#[cube]
pub fn wrap_delta(d: f32, bounds_x: f32) -> f32 {
  let mut out = d;
  let half = bounds_x * 0.5f32;
  if out > half {
    out -= bounds_x;
  } else if out < -half {
    out += bounds_x;
  }
  out
}

/// Bring an x back inside the world, as `move_particles` does.
#[cube]
pub fn wrap_x(x: f32, bounds_x: f32) -> f32 {
  (x + bounds_x) % bounds_x
}

/// Direction from particle `a` to particle `b`, or zero if they are on top of
/// each other.
///
/// Takes `ppos` mutably although it only reads it: a CubeCL expand type has no
/// reborrow, so a `&mut [f32]` kernel argument cannot be passed to a `&[f32]`
/// parameter. Same for [`limb_axis`] below.
#[cube]
pub fn seg_dir(ppos: &mut [f32], a: usize, b: usize, bounds_x: f32) -> Dir {
  let dx = wrap_delta(ppos[2 * b] - ppos[2 * a], bounds_x);
  let dy = ppos[2 * b + 1] - ppos[2 * a + 1];
  let len = f32::sqrt(dx * dx + dy * dy);
  let mut out = Dir {
    x: 0.0f32,
    y: 0.0f32,
  };
  if len > 1e-6f32 {
    out.x = dx / len;
    out.y = dy / len;
  }
  out
}

/// Rotate a direction counter-clockwise.
#[cube]
pub fn rotate(d: Dir, angle: f32) -> Dir {
  let c = f32::cos(angle);
  let s = f32::sin(angle);
  Dir {
    x: d.x * c - d.y * s,
    y: d.x * s + d.y * c,
  }
}

/// Move `a` and `b` symmetrically until they are `rest` apart.
#[cube]
pub fn project_distance(ppos: &mut [f32], a: usize, b: usize, rest: f32, bounds_x: f32) {
  let ax = ppos[2 * a];
  let ay = ppos[2 * a + 1];
  let bx = ppos[2 * b];
  let by = ppos[2 * b + 1];
  let dx = wrap_delta(bx - ax, bounds_x);
  let dy = by - ay;
  let len = f32::sqrt(dx * dx + dy * dy);
  if len > 1e-6f32 {
    // Half the error each: equal masses, so the correction splits evenly.
    let scale = 0.5f32 * (len - rest) / len;
    ppos[2 * a] = wrap_x(ax + dx * scale, bounds_x);
    ppos[2 * a + 1] = ay + dy * scale;
    ppos[2 * b] = wrap_x(bx - dx * scale, bounds_x);
    ppos[2 * b + 1] = by - dy * scale;
  }
}

/// Swing the segment `u -> v` towards the direction `target`, moving only `v`
/// and keeping the segment's current length. `stiffness` is the fraction of
/// the way there one sweep takes it.
#[cube]
pub fn project_angle(
  ppos: &mut [f32],
  u: usize,
  v: usize,
  target: Dir,
  stiffness: f32,
  bounds_x: f32,
) {
  let ux = ppos[2 * u];
  let uy = ppos[2 * u + 1];
  let vx = ppos[2 * v];
  let vy = ppos[2 * v + 1];
  let dx = wrap_delta(vx - ux, bounds_x);
  let dy = vy - uy;
  let len = f32::sqrt(dx * dx + dy * dy);
  // A zero target is "no direction to aim at" — an absent parent segment or
  // a pair of coincident particles — and leaves the segment alone.
  let aimed = target.x != 0.0f32 || target.y != 0.0f32;
  if aimed && len > 1e-6f32 {
    let goal_x = ux + target.x * len;
    let goal_y = uy + target.y * len;
    ppos[2 * v] = wrap_x(vx + stiffness * wrap_delta(goal_x - vx, bounds_x), bounds_x);
    ppos[2 * v + 1] = vy + stiffness * (goal_y - vy);
  }
}

/// How far the segment `u -> v` is from pointing along `target`, in radians
/// and unsigned, or zero where there is no angle to be had — the same "no
/// direction to aim at" cases [`project_angle`] leaves alone.
///
/// This is the error that projection is about to take a `stiffness` fraction
/// out of, measured before it does. `docs/organism.md`, *Growth*: its running
/// maximum per limb is what the life tick breaks a base joint on.
///
/// Takes the target as two floats rather than a [`Dir`], so a caller can
/// measure and then project against the same one: a CubeCL expand type is not
/// `Copy`, so passing the `Dir` here would move it.
#[cube]
pub fn angle_error(
  ppos: &mut [f32],
  u: usize,
  v: usize,
  target_x: f32,
  target_y: f32,
  bounds_x: f32,
) -> f32 {
  let dx = wrap_delta(ppos[2 * v] - ppos[2 * u], bounds_x);
  let dy = ppos[2 * v + 1] - ppos[2 * u + 1];
  let len = f32::sqrt(dx * dx + dy * dy);
  let aimed = target_x != 0.0f32 || target_y != 0.0f32;
  let mut out = 0.0f32;
  if aimed && len > 1e-6f32 {
    let ux = dx / len;
    let uy = dy / len;
    // The target is a unit direction either way it was built, so the cross
    // and the dot are the sine and cosine of the angle between the two.
    let dot = target_x * ux + target_y * uy;
    let cross = target_x * uy - target_y * ux;
    out = f32::atan2(f32::abs(cross), dot);
  }
  out
}

/// Particles limb `l` of organism `o` actually holds: the leading run of its
/// slice of the map, which is `ceil(grown)`.
///
/// Not the record's `length`, which is the *target* a limb grows towards: a
/// limb part way there holds fewer, and everything that walks a body — this
/// pass, the energy budget, the geometry, the renderer — has to stop where
/// the particles do rather than where the genome is aiming. Particles are
/// claimed in index order and a limb gives them all back at once, so the
/// prefix is the whole of it.
#[cube]
pub fn limb_len(map: &[u32], pbase: u32, l: u32, #[comptime] cfg: BodyCfg) -> u32 {
  let mp = comptime!(cfg.max_particles_per_limb);
  let mut n = 0u32;
  let mut counting = true;
  for i in 0..mp {
    if counting && particle_at(map, pbase, l, i, cfg) == NO_PARTICLE {
      counting = false;
    }
    if counting {
      n += 1u32;
    }
  }
  n
}

/// The rest length of the segment ending at particle `k`, as a fraction of
/// `limb_segment_length`: `clamp(grown - k, 0, 1)`.
///
/// The last segment of a growing limb is the one that lengthens between
/// particle claims, so a limb grows smoothly rather than a whole segment at a
/// time (`docs/organism.md`, *Growth*).
#[cube]
pub fn segment_fraction(grown: f32, k: u32) -> f32 {
  f32::clamp(grown - k as f32, 0.0f32, 1.0f32)
}

/// The particle holding `(organism, limb, index)`.
#[cube]
pub fn particle_at(map: &[u32], pbase: u32, l: u32, i: u32, #[comptime] cfg: BodyCfg) -> u32 {
  map[(pbase + l * comptime!(cfg.max_particles_per_limb) + i) as usize]
}

/// The parent limb's last particle: where a limb's base joint sits. Returns
/// [`NO_PARTICLE`] for the root, whose joint is the world frame instead.
#[cube]
pub fn base_particle(
  limbs: &LimbArgs,
  map: &[u32],
  base: u32,
  pbase: u32,
  l: u32,
  #[comptime] cfg: BodyCfg,
) -> u32 {
  let parent = limbs.parent[(base + l) as usize];
  let mut out = NO_PARTICLE.runtime();
  if parent != l {
    let n = limb_len(map, pbase, parent, cfg);
    if n > 0u32 {
      out = particle_at(map, pbase, parent, n - 1u32, cfg);
    }
  }
  out
}

/// The direction a child limb's base joint is measured against: the limb's
/// last segment, or — for a limb too short to have one — the direction it
/// itself came off its parent at. A one-particle root falls back to its own
/// grow angle in the world frame, which is exactly what the pin and the base
/// joint would have given it.
#[cube]
pub fn limb_axis(
  ppos: &mut [f32],
  limbs: &LimbArgs,
  map: &[u32],
  base: u32,
  pbase: u32,
  l: u32,
  bounds_x: f32,
  #[comptime] cfg: BodyCfg,
) -> Dir {
  let n = limb_len(map, pbase, l, cfg);
  let mut out = Dir {
    x: 0.0f32,
    y: 0.0f32,
  };
  if n >= 2u32 {
    let a = particle_at(map, pbase, l, n - 2u32, cfg);
    let b = particle_at(map, pbase, l, n - 1u32, cfg);
    if a != NO_PARTICLE && b != NO_PARTICLE {
      let d = seg_dir(ppos, a as usize, b as usize, bounds_x);
      out.x = d.x;
      out.y = d.y;
    }
  } else if n == 1u32 {
    let anchor = base_particle(limbs, map, base, pbase, l, cfg);
    let first = particle_at(map, pbase, l, 0u32, cfg);
    if anchor != NO_PARTICLE && first != NO_PARTICLE {
      let d = seg_dir(ppos, anchor as usize, first as usize, bounds_x);
      out.x = d.x;
      out.y = d.y;
    } else if anchor == NO_PARTICLE {
      // The root limb: its grow angle is read in the world frame.
      let g = limbs.grow_angle[(base + l) as usize];
      out.x = f32::cos(g);
      out.y = f32::sin(g);
    }
  }
  out
}

/// Project one organism's limbs. One unit per organism slot; a free slot and
/// an absent limb are left alone.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn project_constraints(
  ppos: &mut [f32],
  pos: &mut [f32],
  vel: &mut [f32],
  ppos_prev: &mut [f32],
  limbs: &LimbArgs,
  map: &[u32],
  anchors: &[f32],
  grown: &[f32],
  bend_max: &mut [f32],
  alive: &[u32],
  params: &[f32],
  #[comptime] cfg: BodyCfg,
) {
  let o = ABSOLUTE_POS as u32;
  if o >= comptime!(cfg.max_organisms) {
    terminate!();
  }
  if alive[o as usize] == 0u32 {
    terminate!();
  }

  let ml = comptime!(cfg.max_limbs);
  let mp = comptime!(cfg.max_particles_per_limb);
  let base = o * ml;
  let pbase = o * ml * mp;

  let rest = params[P_LIMB_SEGMENT_LENGTH as usize];
  let joint_k = params[P_JOINT_STIFFNESS as usize];
  let bend_k = params[P_BEND_STIFFNESS as usize];
  let bounds_x = params[P_BOUNDS_X as usize];
  let bounds_y = params[P_BOUNDS_Y as usize];
  let dt = params[P_DT as usize];
  let dt_predict = params[P_DT_PREDICT as usize];
  let iterations = params[P_CONSTRAINT_ITERS as usize] as u32;

  // Where the integrator left each particle. The projection's displacement
  // from here is what the velocity picks up, so it has to be remembered
  // before the first sweep moves anything.
  for l in 0..ml {
    for i in 0..mp {
      let id = particle_at(map, pbase, l, i, cfg);
      if id != NO_PARTICLE {
        ppos_prev[2 * id as usize] = ppos[2 * id as usize];
        ppos_prev[2 * id as usize + 1] = ppos[2 * id as usize + 1];
      }
    }
  }

  for _ in 0..iterations {
    // (a) pin the root limb's first particle to the anchor. The root is the
    // limb that is its own parent; the genome makes that record 0.
    let root_first = particle_at(map, pbase, 0u32, 0u32, cfg);
    if root_first != NO_PARTICLE {
      ppos[2 * root_first as usize] = anchors[2 * o as usize];
      ppos[2 * root_first as usize + 1] = anchors[2 * o as usize + 1];
    }

    // (b) distance, along each limb and across each base joint. The segment
    // ending at particle `k` is at its full rest length once the limb has
    // grown past `k + 1`, and shorter while it is growing into it, so a
    // limb lengthens between particle claims.
    for l in 0..ml {
      let n = limb_len(map, pbase, l, cfg);
      if n > 0u32 {
        let g = grown[(base + l) as usize];
        let anchor = base_particle(limbs, map, base, pbase, l, cfg);
        let first = particle_at(map, pbase, l, 0u32, cfg);
        if anchor != NO_PARTICLE && first != NO_PARTICLE {
          let r = rest * segment_fraction(g, 0u32);
          project_distance(ppos, anchor as usize, first as usize, r, bounds_x);
        }
        if n > 1u32 {
          for i in 1..n {
            let a = particle_at(map, pbase, l, i - 1u32, cfg);
            let b = particle_at(map, pbase, l, i, cfg);
            if a != NO_PARTICLE && b != NO_PARTICLE {
              let r = rest * segment_fraction(g, i);
              project_distance(ppos, a as usize, b as usize, r, bounds_x);
            }
          }
        }
      }
    }

    // (c) base-joint angle
    for l in 0..ml {
      let n = limb_len(map, pbase, l, cfg);
      if n > 0u32 {
        let g = limbs.grow_angle[(base + l) as usize];
        let anchor = base_particle(limbs, map, base, pbase, l, cfg);
        let first = particle_at(map, pbase, l, 0u32, cfg);
        if anchor != NO_PARTICLE {
          // Against the parent limb's last segment.
          let parent = limbs.parent[(base + l) as usize];
          let axis = limb_axis(ppos, limbs, map, base, pbase, parent, bounds_x, cfg);
          if first != NO_PARTICLE {
            let target = rotate(axis, g);
            let err = angle_error(
              ppos,
              anchor as usize,
              first as usize,
              target.x,
              target.y,
              bounds_x,
            );
            if err > bend_max[(base + l) as usize] {
              bend_max[(base + l) as usize] = err;
            }
            project_angle(
              ppos,
              anchor as usize,
              first as usize,
              target,
              joint_k,
              bounds_x,
            );
          }
        } else if n > 1u32 {
          // The root limb: its first segment is `p0 -> p1`, in the world
          // frame.
          let second = particle_at(map, pbase, l, 1u32, cfg);
          let target = Dir {
            x: f32::cos(g),
            y: f32::sin(g),
          };
          if first != NO_PARTICLE && second != NO_PARTICLE {
            let err = angle_error(
              ppos,
              first as usize,
              second as usize,
              target.x,
              target.y,
              bounds_x,
            );
            if err > bend_max[(base + l) as usize] {
              bend_max[(base + l) as usize] = err;
            }
            project_angle(
              ppos,
              first as usize,
              second as usize,
              target,
              joint_k,
              bounds_x,
            );
          }
        }
      }
    }

    // (d) bend: consecutive segments of a limb's chain stay in line
    for l in 0..ml {
      let n = limb_len(map, pbase, l, cfg);
      if n > 1u32 {
        let anchor = base_particle(limbs, map, base, pbase, l, cfg);
        let first = particle_at(map, pbase, l, 0u32, cfg);
        let second = particle_at(map, pbase, l, 1u32, cfg);
        if anchor != NO_PARTICLE && first != NO_PARTICLE && second != NO_PARTICLE {
          let axis = seg_dir(ppos, anchor as usize, first as usize, bounds_x);
          project_angle(
            ppos,
            first as usize,
            second as usize,
            axis,
            bend_k,
            bounds_x,
          );
        }
        if n > 2u32 {
          for i in 2..n {
            let a = particle_at(map, pbase, l, i - 2u32, cfg);
            let b = particle_at(map, pbase, l, i - 1u32, cfg);
            let c = particle_at(map, pbase, l, i, cfg);
            if a != NO_PARTICLE && b != NO_PARTICLE && c != NO_PARTICLE {
              let axis = seg_dir(ppos, a as usize, b as usize, bounds_x);
              project_angle(ppos, b as usize, c as usize, axis, bend_k, bounds_x);
            }
          }
        }
      }
    }

    // (e) the world. x wraps as it does everywhere; y clamps rather than
    // reflecting, because a bounce would fight the constraint that pushed
    // the particle out. Inside the sweep, not after it, so the next sweep's
    // distance constraints see the clamped position and can bend the limb
    // along the floor or ceiling instead of compressing it against one.
    for l in 0..ml {
      for i in 0..mp {
        let id = particle_at(map, pbase, l, i, cfg);
        if id != NO_PARTICLE {
          let id = id as usize;
          ppos[2 * id] = wrap_x(ppos[2 * id], bounds_x);
          if ppos[2 * id + 1] < 0.0f32 {
            ppos[2 * id + 1] = 0.0f32;
          } else if ppos[2 * id + 1] > bounds_y {
            ppos[2 * id + 1] = bounds_y;
          }
        }
      }
    }
  }

  // Velocity from the projection, and the prediction the neighbour kernels
  // read next step. `move_particles` already put this step's integrated
  // velocity in `vel` (post-bounce); what the projection moved the particle
  // by is added to it, which is the position-based-dynamics velocity with
  // the step's starting position as the reference.
  for l in 0..ml {
    for i in 0..mp {
      let id = particle_at(map, pbase, l, i, cfg) as usize;
      if id != NO_PARTICLE as usize {
        // Already inside the world: the last sweep's bounds constraint put it
        // there.
        let mut px = ppos[2 * id];
        let mut py = ppos[2 * id + 1];

        let vx = vel[2 * id] + wrap_delta(px - ppos_prev[2 * id], bounds_x) / dt;
        let vy = vel[2 * id + 1] + (py - ppos_prev[2 * id + 1]) / dt;
        vel[2 * id] = vx;
        vel[2 * id + 1] = vy;

        px = wrap_x(px + vx * dt_predict, bounds_x);
        py += vy * dt_predict;
        if py < 0.0f32 {
          py = -py;
        } else if py > bounds_y {
          py = (2.0f32 * bounds_y - py) - 1e-4f32;
        }
        pos[2 * id] = px;
        pos[2 * id + 1] = py;
      }
    }
  }
}

/// Launch the pass over every organism slot. The caller skips it entirely
/// when nothing is alive.
#[allow(clippy::too_many_arguments)]
pub fn launch<R: Runtime>(
  client: &ComputeClient<R>,
  sph: &SphDevice,
  bodies: &BodyState,
  pop: &Population,
  params: &Handle,
  cfg: BodyCfg,
  world: Cfg,
) {
  let n = world.num_particles as usize;
  project_constraints::launch::<R>(
    client,
    cube_count(cfg.max_organisms as usize),
    CubeDim::new_1d(super::CUBE_DIM),
    whole(&sph.ppos, n * 2),
    whole(&sph.pos, n * 2),
    whole(&sph.vel, n * 2),
    whole(&bodies.device.ppos_prev, n * 2),
    limb_args(&pop.device.limbs),
    whole(&bodies.device.limb_particles, cfg.particle_map_len()),
    whole(&bodies.device.anchors, cfg.max_organisms as usize * 2),
    whole(&bodies.device.grown, cfg.limb_count()),
    whole(&bodies.device.bend_max, cfg.limb_count()),
    whole(&pop.device.organisms.alive, cfg.max_organisms as usize),
    whole(params, super::PARAM_COUNT),
    cfg,
  );
}

// --- Plain-Rust reference ---

/// Host twin of [`project_constraints`], over the host copies of everything.
///
/// Same order, same projections, same wrapping; written as a sequential walk
/// because that is what the kernel is per organism.
///
/// `bend_max` is `[max_organisms x max_limbs]` and is accumulated into, not
/// overwritten: it is the running maximum the life tick reads and resets, so
/// the caller owns its zeroing exactly as the pack kernel does on the device.
pub fn project_constraints_ref(
  particles: &mut SphHost,
  pop: &Population,
  bodies: &BodyState,
  bend_max: &mut [f32],
  params: &crate::SimParams,
  geom: &crate::world::WorldGeometry,
) {
  let cfg = bodies.cfg;
  let (ml, mp) = (cfg.max_limbs as usize, cfg.max_particles_per_limb as usize);
  let bounds = geom.bounds;
  let rest = params.limb_segment_length;

  for o in 0..cfg.max_organisms as usize {
    if pop.organisms.alive[o] == 0 {
      continue;
    }
    let ids: Vec<Option<usize>> = (0..ml * mp)
      .map(|k| {
        let id = bodies.limb_particles[o * ml * mp + k];
        (id != NO_PARTICLE).then_some(id as usize)
      })
      .collect();
    let at = |l: usize, i: usize| ids[l * mp + i];
    let len = |l: usize| bodies.limb_len(o, l);
    let grown = |l: usize| bodies.grown[o * ml + l];
    let parent = |l: usize| pop.limbs.parent[o * ml + l] as usize;
    let grow_angle = |l: usize| pop.limbs.grow_angle[o * ml + l];
    let base_particle = |l: usize| -> Option<usize> {
      let p = parent(l);
      if p == l {
        return None;
      }
      let n = len(p);
      (n > 0).then(|| at(p, n - 1)).flatten()
    };

    let prev: Vec<Option<glam::Vec2>> = ids
      .iter()
      .map(|id| id.map(|id| particles.ppos[id]))
      .collect();

    let axis =
      |particles: &SphHost, l: usize| limb_axis_ref(&particles.ppos, pop, bodies, o, l, bounds.x);

    for _ in 0..params.constraint_iterations.max(0) {
      if let Some(root) = at(0, 0) {
        particles.ppos[root] = bodies.anchors[o];
      }

      for l in 0..ml {
        let n = len(l);
        if n == 0 {
          continue;
        }
        let g = grown(l);
        if let (Some(a), Some(b)) = (base_particle(l), at(l, 0)) {
          project_distance_ref(particles, a, b, rest * segment_fraction_ref(g, 0), bounds.x);
        }
        for i in 1..n {
          if let (Some(a), Some(b)) = (at(l, i - 1), at(l, i)) {
            project_distance_ref(particles, a, b, rest * segment_fraction_ref(g, i), bounds.x);
          }
        }
      }

      for l in 0..ml {
        let n = len(l);
        if n == 0 {
          continue;
        }
        let g = grow_angle(l);
        let mut record_bend = |err: f32| {
          let slot = &mut bend_max[o * ml + l];
          if err > *slot {
            *slot = err;
          }
        };
        match (base_particle(l), at(l, 0)) {
          (Some(a), Some(first)) => {
            let target = rotate_ref(axis(particles, parent(l)), g);
            record_bend(angle_error_ref(particles, a, first, target, bounds.x));
            project_angle_ref(
              particles,
              a,
              first,
              target,
              params.joint_stiffness,
              bounds.x,
            );
          }
          (None, Some(first)) if n > 1 => {
            if let Some(second) = at(l, 1) {
              let target = Vec2::new(g.cos(), g.sin());
              record_bend(angle_error_ref(particles, first, second, target, bounds.x));
              project_angle_ref(
                particles,
                first,
                second,
                target,
                params.joint_stiffness,
                bounds.x,
              );
            }
          }
          _ => {}
        }
      }

      for l in 0..ml {
        let n = len(l);
        if n <= 1 {
          continue;
        }
        if let (Some(a), Some(b), Some(c)) = (base_particle(l), at(l, 0), at(l, 1)) {
          let target = seg_dir_ref(particles, a, b, bounds.x);
          project_angle_ref(particles, b, c, target, params.bend_stiffness, bounds.x);
        }
        for i in 2..n {
          if let (Some(a), Some(b), Some(c)) = (at(l, i - 2), at(l, i - 1), at(l, i)) {
            let target = seg_dir_ref(particles, a, b, bounds.x);
            project_angle_ref(particles, b, c, target, params.bend_stiffness, bounds.x);
          }
        }
      }

      for id in ids.iter().flatten() {
        let mut p = particles.ppos[*id];
        p.x = wrap_x_ref(p.x, bounds.x);
        p.y = p.y.clamp(0.0, bounds.y);
        particles.ppos[*id] = p;
      }
    }

    for (k, id) in ids.iter().enumerate() {
      let Some(id) = *id else { continue };
      let old = prev[k].expect("a particle that is there has a previous position");
      let p = particles.ppos[id];

      let mut v = particles.vel[id];
      v.x += wrap_delta_ref(p.x - old.x, bounds.x) / params.dt;
      v.y += (p.y - old.y) / params.dt;
      particles.vel[id] = v;

      let mut pred = p + v * params.dt_predict;
      pred.x = wrap_x_ref(pred.x, bounds.x);
      if pred.y < 0.0 {
        pred.y = -pred.y;
      } else if pred.y > bounds.y {
        pred.y = (2.0 * bounds.y - pred.y) - 1e-4;
      }
      particles.pos[id] = pred;
    }
  }
}

/// Host twin of [`limb_axis`]: the direction a child limb's base joint is
/// measured against. Shared with `kernels::spawn::layout_limbs_ref`, which
/// lays a new limb out along it.
pub fn limb_axis_ref(
  ppos: &[Vec2],
  pop: &Population,
  bodies: &BodyState,
  organism: usize,
  limb: usize,
  bounds_x: f32,
) -> Vec2 {
  let cfg = bodies.cfg;
  let record = organism * cfg.max_limbs as usize + limb;
  let n = bodies.limb_len(organism, limb);
  let at = |l: usize, i: usize| {
    let id = bodies.particle(organism, l, i);
    (id != NO_PARTICLE).then_some(id as usize)
  };
  let dir = |a: usize, b: usize| {
    let d = Vec2::new(
      wrap_delta_ref(ppos[b].x - ppos[a].x, bounds_x),
      ppos[b].y - ppos[a].y,
    );
    if d.length() > 1e-6 {
      d / d.length()
    } else {
      Vec2::ZERO
    }
  };

  if n >= 2 {
    return match (at(limb, n - 2), at(limb, n - 1)) {
      (Some(a), Some(b)) => dir(a, b),
      _ => Vec2::ZERO,
    };
  }
  if n == 0 {
    return Vec2::ZERO;
  }

  // One particle: the limb's own base segment says which way it points, and
  // for the root — which has none — its grow angle in the world frame does.
  let parent = pop.limbs.parent[record] as usize;
  if parent == limb {
    let g = pop.limbs.grow_angle[record];
    return Vec2::new(g.cos(), g.sin());
  }
  let pn = bodies.limb_len(organism, parent);
  if pn == 0 {
    return Vec2::ZERO;
  }
  match (at(parent, pn - 1), at(limb, 0)) {
    (Some(a), Some(b)) => dir(a, b),
    _ => Vec2::ZERO,
  }
}

/// Host twin of [`segment_fraction`].
pub fn segment_fraction_ref(grown: f32, k: usize) -> f32 {
  (grown - k as f32).clamp(0.0, 1.0)
}

pub fn wrap_delta_ref(d: f32, bounds_x: f32) -> f32 {
  let half = bounds_x * 0.5;
  if d > half {
    d - bounds_x
  } else if d < -half {
    d + bounds_x
  } else {
    d
  }
}

pub fn wrap_x_ref(x: f32, bounds_x: f32) -> f32 {
  (x + bounds_x) % bounds_x
}

fn seg_dir_ref(particles: &SphHost, a: usize, b: usize, bounds_x: f32) -> Vec2 {
  let d = Vec2::new(
    wrap_delta_ref(particles.ppos[b].x - particles.ppos[a].x, bounds_x),
    particles.ppos[b].y - particles.ppos[a].y,
  );
  let len = d.length();
  if len > 1e-6 { d / len } else { Vec2::ZERO }
}

pub fn rotate_ref(d: Vec2, angle: f32) -> Vec2 {
  let (s, c) = (angle.sin(), angle.cos());
  Vec2::new(d.x * c - d.y * s, d.x * s + d.y * c)
}

fn project_distance_ref(particles: &mut SphHost, a: usize, b: usize, rest: f32, bounds_x: f32) {
  let (pa, pb) = (particles.ppos[a], particles.ppos[b]);
  let d = Vec2::new(wrap_delta_ref(pb.x - pa.x, bounds_x), pb.y - pa.y);
  let len = d.length();
  if len > 1e-6 {
    let scale = 0.5 * (len - rest) / len;
    let mut na = pa + d * scale;
    let mut nb = pb - d * scale;
    na.x = wrap_x_ref(na.x, bounds_x);
    nb.x = wrap_x_ref(nb.x, bounds_x);
    particles.ppos[a] = na;
    particles.ppos[b] = nb;
  }
}

/// Host twin of [`angle_error`].
fn angle_error_ref(particles: &SphHost, u: usize, v: usize, target: Vec2, bounds_x: f32) -> f32 {
  if target == Vec2::ZERO {
    return 0.0;
  }
  let (pu, pv) = (particles.ppos[u], particles.ppos[v]);
  let d = Vec2::new(wrap_delta_ref(pv.x - pu.x, bounds_x), pv.y - pu.y);
  let len = d.length();
  if len <= 1e-6 {
    return 0.0;
  }
  let u = d / len;
  let dot = target.x * u.x + target.y * u.y;
  let cross = target.x * u.y - target.y * u.x;
  cross.abs().atan2(dot)
}

fn project_angle_ref(
  particles: &mut SphHost,
  u: usize,
  v: usize,
  target: Vec2,
  stiffness: f32,
  bounds_x: f32,
) {
  if target == Vec2::ZERO {
    return;
  }
  let (pu, pv) = (particles.ppos[u], particles.ppos[v]);
  let d = Vec2::new(wrap_delta_ref(pv.x - pu.x, bounds_x), pv.y - pu.y);
  let len = d.length();
  if len > 1e-6 {
    let goal = pu + target * len;
    let mut out = Vec2::new(
      pv.x + stiffness * wrap_delta_ref(goal.x - pv.x, bounds_x),
      pv.y + stiffness * (goal.y - pv.y),
    );
    out.x = wrap_x_ref(out.x, bounds_x);
    particles.ppos[v] = out;
  }
}

#[cfg(test)]
mod tests {
  use super::super::test_support::{OrganismHarness, assert_close, assert_close_vec2};
  use crate::bodies::NO_PARTICLE;
  use crate::particles::{SphDevice, SphHost};

  /// Every body particle of the organism, in map order.
  fn body_ids(h: &OrganismHarness) -> Vec<usize> {
    h.bodies
      .limb_particles
      .iter()
      .filter(|id| **id != NO_PARTICLE)
      .map(|id| *id as usize)
      .collect()
  }

  fn run_kernel(h: &OrganismHarness) -> SphHost {
    super::launch(
      &h.client,
      &h.sph,
      &h.bodies,
      &h.pop,
      &h.params_buf,
      h.body_cfg,
      h.cfg,
    );
    h.read_particles()
  }

  /// The per-limb bend maxima the pass has accumulated on the device.
  fn read_bend_max(h: &OrganismHarness) -> Vec<f32> {
    crate::soa::download_field(
      &h.client,
      &h.bodies.device.bend_max,
      h.body_cfg.limb_count(),
    )
  }

  #[test]
  fn matches_reference() {
    let h = OrganismHarness::new();
    let actual = run_kernel(&h);

    let mut expected = h.particles.clone();
    let mut bend = vec![0.0f32; h.body_cfg.limb_count()];
    super::project_constraints_ref(
      &mut expected,
      &h.pop,
      &h.bodies,
      &mut bend,
      &h.params,
      &h.geom,
    );

    assert_close(&read_bend_max(&h), &bend, 1e-6, "bend maxima");
    assert!(
      bend.iter().any(|v| *v > 1e-3),
      "no joint was bent, so the maxima prove nothing: {bend:?}"
    );
    assert_close_vec2(&actual.ppos, &expected.ppos, 1e-6, "ppos after projection");
    assert_close_vec2(&actual.pos, &expected.pos, 1e-6, "pos after projection");
    assert_close_vec2(&actual.vel, &expected.vel, 1e-6, "vel after projection");
    assert_ne!(
      actual.ppos, h.particles.ppos,
      "the projection moved nothing, so the test proves nothing"
    );
  }

  #[test]
  fn the_fluid_is_untouched() {
    let h = OrganismHarness::new();
    let actual = run_kernel(&h);
    let body = body_ids(&h);
    for i in 0..h.particles.len() {
      if body.contains(&i) {
        continue;
      }
      assert_eq!(actual.pos[i], h.particles.pos[i], "particle {i} pos");
      assert_eq!(actual.ppos[i], h.particles.ppos[i], "particle {i} ppos");
      assert_eq!(actual.vel[i], h.particles.vel[i], "particle {i} vel");
    }
  }

  /// Running the pass repeatedly is the same as running more sweeps: the
  /// distances converge on the rest length and the root stays on its anchor.
  #[test]
  fn repeated_sweeps_converge_on_the_rest_shape() {
    let mut h = OrganismHarness::new();
    for _ in 0..60 {
      h.particles = run_kernel(&h);
      h.sph = SphDevice::upload(&h.client, &h.particles);
    }

    let root = h.bodies.particle(0, 0, 0) as usize;
    let anchor = h.bodies.anchors[0];
    assert!(
      (h.particles.ppos[root] - anchor).length() < 1e-3,
      "the root drifted off its anchor: {:?} vs {anchor:?}",
      h.particles.ppos[root]
    );

    let rest = h.params.limb_segment_length;
    let ml = h.body_cfg.max_limbs as usize;
    let mp = h.body_cfg.max_particles_per_limb as usize;
    for limb in 0..ml {
      let record = h.pop.limb_index(0, limb);
      if !h.pop.limbs.part_type[record].is_present() {
        continue;
      }
      let n = (h.pop.limbs.length[record] as usize).min(mp);
      for i in 1..n {
        let a = h.bodies.particle(0, limb, i - 1) as usize;
        let b = h.bodies.particle(0, limb, i) as usize;
        let len = (h.particles.ppos[b] - h.particles.ppos[a]).length();
        assert!(
          (len / rest - 1.0).abs() < 0.05,
          "limb {limb} segment {i} is {len} against a rest length of {rest}"
        );
      }
    }
  }

  /// A limb part way into its last segment is held at that fraction of the
  /// rest length, which is what makes growth visible between particle claims.
  #[test]
  fn a_growing_limb_settles_at_its_fractional_rest_length() {
    let mut h = OrganismHarness::new();
    // The stem two thirds of the way into its second segment; the first is
    // full length either way.
    let stem = h.body_cfg.limb_index(0, 1);
    h.bodies.grown[stem] = 1.65;
    // The third particle of a stem that has not grown that far is not there.
    let slice = h.body_cfg.limb_slice(0, 1);
    h.bodies.limb_particles[slice.start + 2] = NO_PARTICLE;
    h.bodies.upload(&h.client);

    for _ in 0..60 {
      h.particles = run_kernel(&h);
      h.sph = SphDevice::upload(&h.client, &h.particles);
    }

    let rest = h.params.limb_segment_length;
    let a = h.bodies.particle(0, 1, 0) as usize;
    let b = h.bodies.particle(0, 1, 1) as usize;
    let len = (h.particles.ppos[b] - h.particles.ppos[a]).length();
    let expected = rest * 0.65;
    assert!(
      (len / expected - 1.0).abs() < 0.05,
      "the growing segment is {len} against {expected}"
    );
  }

  /// The maximum runs across steps, not just across one pass's sweeps: two
  /// launches with no reset between them leave the worst of both, and the
  /// reference accumulates identically.
  #[test]
  fn the_bend_maximum_spans_the_whole_interval() {
    let mut h = OrganismHarness::new();

    // One step, on the host alone, as the yardstick.
    let mut one_step = h.particles.clone();
    let mut after_one = vec![0.0f32; h.body_cfg.limb_count()];
    super::project_constraints_ref(
      &mut one_step,
      &h.pop,
      &h.bodies,
      &mut after_one,
      &h.params,
      &h.geom,
    );

    // Two steps with no reset between them, kernel and reference together.
    let mut expected = h.particles.clone();
    let mut bend = vec![0.0f32; h.body_cfg.limb_count()];
    for _ in 0..2 {
      h.particles = run_kernel(&h);
      h.sph = SphDevice::upload(&h.client, &h.particles);
      super::project_constraints_ref(
        &mut expected,
        &h.pop,
        &h.bodies,
        &mut bend,
        &h.params,
        &h.geom,
      );
    }

    assert_close(
      &read_bend_max(&h),
      &bend,
      1e-6,
      "bend maxima over two steps",
    );
    assert!(
      bend.iter().zip(&after_one).all(|(two, one)| two >= one),
      "a maximum went down over a step that never reset it"
    );
  }

  /// A dead slot is a no-op, whatever its map says.
  #[test]
  fn a_free_organism_slot_is_left_alone() {
    let mut h = OrganismHarness::new();
    h.pop.organisms.alive[0] = 0;
    h.pop.upload(&h.client);
    let actual = run_kernel(&h);
    assert_eq!(actual.ppos, h.particles.ppos);
    assert_eq!(actual.pos, h.particles.pos);
    assert_eq!(actual.vel, h.particles.vel);
  }
}
