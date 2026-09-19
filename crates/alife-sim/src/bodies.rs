//! Bodies: which particle holds which limb particle, and where each organism
//! is anchored.
//!
//! From `docs/organism.md`'s *Body* and *Implementation layout*. A limb is a
//! chain of particles in the fluid's particle system; the genome says how many
//! and of what type, and this is the map from `(organism, limb, index)` to the
//! particle slot that holds it. The constraint pass
//! ([`crate::kernels::constraints`]) walks that map, and the geometry kernel
//! ([`crate::kernels::limb_geometry`]) publishes what the brain will read.
//!
//! Capacity is `max_organisms x max_limbs x max_particles_per_limb`, the same
//! product the particle system reserves as body slots.

use cubecl::prelude::*;
use cubecl_runtime::server::Handle;
use glam::Vec2;

use crate::SimParams;
use crate::define_soa;
use crate::world::WorldGeometry;

define_soa! {
    /// What the brain chunk reads about a limb's placement in the world, one
    /// entry per `(organism, limb)`. Written by
    /// [`crate::kernels::limb_geometry`] after every constraint pass.
    LimbGeometryHost / LimbGeometryDevice {
        /// The limb's first particle, relative to the root limb's first
        /// particle. The body frame is the world frame for plants.
        rel_pos: Vec2,
        /// Direction of the limb's first segment: from the parent's last
        /// particle to this limb's first, or `p0 -> p1` for the root.
        angle: f32,
        /// Hops from the root limb; the root itself is 0.
        depth: u32,
    }
}

/// An entry of [`BodyState::limb_particles`] that holds no particle: the limb
/// is absent, or has not grown that far.
pub const NO_PARTICLE: u32 = u32::MAX;

/// The shapes the organism kernels bake in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BodyCfg {
  pub max_organisms: u32,
  pub max_limbs: u32,
  pub max_particles_per_limb: u32,
}

impl BodyCfg {
  pub fn from_params(params: &SimParams) -> Self {
    Self {
      max_organisms: params.max_organisms.max(1) as u32,
      max_limbs: params.max_limbs.max(1) as u32,
      max_particles_per_limb: params.max_particles_per_limb.max(1) as u32,
    }
  }

  /// Entries in the `(organism, limb, index)` map.
  pub fn particle_map_len(&self) -> usize {
    (self.max_organisms * self.max_limbs * self.max_particles_per_limb) as usize
  }

  /// Entries in a per-limb buffer.
  pub fn limb_count(&self) -> usize {
    (self.max_organisms * self.max_limbs) as usize
  }

  /// Flat index of one limb's particle slice.
  pub fn limb_slice(&self, organism: usize, limb: usize) -> std::ops::Range<usize> {
    let mp = self.max_particles_per_limb as usize;
    let start = (organism * self.max_limbs as usize + limb) * mp;
    start..start + mp
  }
}

/// Device mirrors of everything the organism kernels read or write.
#[derive(Debug, Clone)]
pub struct BodyDevice {
  /// `[max_organisms x max_limbs x max_particles_per_limb]` particle ids.
  pub limb_particles: Handle,
  /// `[max_organisms]` anchor positions, interleaved x,y.
  pub anchors: Handle,
  /// Where the integrator left each particle, before the projection moved it.
  /// `[num_particles]` interleaved x,y; only body particles are written.
  pub ppos_prev: Handle,
  pub geometry: LimbGeometryDevice,
}

/// Host master copies plus their device mirrors, alongside
/// [`crate::genome::Population`].
pub struct BodyState {
  pub cfg: BodyCfg,
  pub limb_particles: Vec<u32>,
  /// Each organism's anchor: the soil cell its root germinated in.
  pub anchors: Vec<Vec2>,
  pub geometry: LimbGeometryHost,
  pub device: BodyDevice,
}

impl BodyState {
  pub fn new<R: Runtime>(
    client: &ComputeClient<R>,
    params: &SimParams,
    geom: &WorldGeometry,
  ) -> Self {
    let cfg = BodyCfg::from_params(params);
    let limb_particles = vec![NO_PARTICLE; cfg.particle_map_len()];
    let anchors = vec![Vec2::ZERO; cfg.max_organisms as usize];
    let geometry = LimbGeometryHost::new(cfg.limb_count());
    let device = BodyDevice {
      limb_particles: client.create_from_slice(bytemuck::cast_slice(&limb_particles)),
      anchors: client.create_from_slice(bytemuck::cast_slice(&flatten(&anchors))),
      ppos_prev: client.empty(geom.num_particles * 2 * size_of::<f32>()),
      geometry: LimbGeometryDevice::upload(client, &geometry),
    };
    Self {
      cfg,
      limb_particles,
      anchors,
      geometry,
      device,
    }
  }

  /// The particle holding `(organism, limb, index)`, or [`NO_PARTICLE`].
  pub fn particle(&self, organism: usize, limb: usize, index: usize) -> u32 {
    self.limb_particles[self.cfg.limb_slice(organism, limb).start + index]
  }

  /// Upload the host master copies of the map and the anchors.
  pub fn upload<R: Runtime>(&mut self, client: &ComputeClient<R>) {
    self.device.limb_particles =
      client.create_from_slice(bytemuck::cast_slice(&self.limb_particles));
    self.device.anchors = client.create_from_slice(bytemuck::cast_slice(&flatten(&self.anchors)));
  }
}

fn flatten(v: &[Vec2]) -> Vec<f32> {
  v.iter().flat_map(|p| [p.x, p.y]).collect()
}
