//! Simulation parameters: one declaration per parameter.
//!
//! Mirrors the X-macro in the C++ tree's `src/config/sim_params.h`. A single
//! line yields the struct field, the compiled default, the TOML key, the CLI
//! flag and the help text shared by `--help` and `--write-config`.
//! Precedence is CLI > TOML > compiled default.

use std::fmt::Write as _;

/// A parameter value that can be read from and written to a TOML document.
pub trait TomlValue: Sized + Copy {
  fn from_toml(value: &toml::Value) -> Option<Self>;
  /// The literal as `--write-config` emits it.
  fn to_toml_literal(self) -> String;
}

impl TomlValue for f32 {
  fn from_toml(value: &toml::Value) -> Option<Self> {
    match value {
      toml::Value::Float(f) => Some(*f as f32),
      toml::Value::Integer(i) => Some(*i as f32),
      _ => None,
    }
  }

  fn to_toml_literal(self) -> String {
    // The shortest literal that reads back as the same `f32`. The C++
    // writer prints `%f` and trims, so its own `dt` comes back as
    // 0.001667 instead of 1/600 — a 2e-4 relative change to the timestep.
    let mut s = format!("{self}");
    if !s.contains('.') && !s.contains('e') && !s.contains("inf") && !s.contains("NaN") {
      s.push_str(".0");
    }
    s
  }
}

impl TomlValue for i32 {
  fn from_toml(value: &toml::Value) -> Option<Self> {
    value.as_integer().map(|i| i as i32)
  }

  fn to_toml_literal(self) -> String {
    self.to_string()
  }
}

impl TomlValue for i64 {
  fn from_toml(value: &toml::Value) -> Option<Self> {
    value.as_integer()
  }

  fn to_toml_literal(self) -> String {
    self.to_string()
  }
}

/// Look a dotted `section.key` path up in a parsed TOML document.
fn toml_lookup<T: TomlValue>(table: &toml::Table, path: &str) -> Option<T> {
  let value = match path.split_once('.') {
    Some((section, key)) => table.get(section)?.as_table()?.get(key)?,
    None => table.get(path)?,
  };
  T::from_toml(value)
}

macro_rules! sim_params {
    ($( $name:ident : $ty:ty = $default:expr, $toml:literal, $cli:literal, $desc:literal; )*) => {
        /// Fully resolved parameters. Every field is a kernel or world constant.
        #[derive(Debug, Clone, Copy, PartialEq)]
        pub struct SimParams {
            $( #[doc = $desc] pub $name: $ty, )*
        }

        impl Default for SimParams {
            fn default() -> Self {
                Self { $( $name: $default, )* }
            }
        }

        /// The same parameters as optional CLI flags, so "not given on the
        /// command line" stays distinguishable from "given the default value".
        #[derive(Debug, Clone, Default, clap::Args)]
        pub struct SimParamsCli {
            $(
                #[arg(long = $cli, help = $desc, value_name = "VALUE")]
                pub $name: Option<$ty>,
            )*
        }

        impl SimParams {
            /// Overlay every value present in a parsed TOML document.
            pub fn apply_toml(&mut self, table: &toml::Table) {
                $( if let Some(v) = toml_lookup::<$ty>(table, $toml) { self.$name = v; } )*
            }

            /// Overlay every flag the user actually passed.
            pub fn apply_cli(&mut self, cli: &SimParamsCli) {
                $( if let Some(v) = cli.$name { self.$name = v; } )*
            }

            /// `(section, key, value literal, description)` per parameter, in
            /// declaration order — the input to the default-config writer.
            fn config_lines(&self) -> Vec<(&'static str, &'static str, String, &'static str)> {
                let mut out = Vec::new();
                $(
                    let (section, key) = match $toml.split_once('.') {
                        Some((s, k)) => (s, k),
                        None => ("", $toml),
                    };
                    out.push((section, key, self.$name.to_toml_literal(), $desc));
                )*
                out
            }
        }
    };
}

sim_params! {
    dt: f32 = 1.0 / 600.0, "fluid.dt", "dt", "Simulation timestep";
    dt_predict: f32 = 1.0 / 120.0, "fluid.dt_predict", "dt-predict",
        "Prediction timestep for position lookahead";
    gravity: f32 = -13.0, "fluid.gravity", "gravity", "Gravity acceleration";
    collision_damping: f32 = 0.5, "fluid.collision_damping", "collision-damping",
        "Velocity damping on boundary collision";
    smoothing_radius: f32 = 0.2, "fluid.smoothing_radius", "smoothing-radius",
        "SPH smoothing radius (lower bound on the grid cell size)";
    target_density: f32 = 234.0, "fluid.target_density", "target-density",
        "Target rest density for pressure calculation";
    pressure_mult: f32 = 225.0, "fluid.pressure_mult", "pressure-mult",
        "Pressure force multiplier";
    near_pressure_mult: f32 = 18.0, "fluid.near_pressure_mult", "near-pressure-mult",
        "Near-field pressure multiplier";
    viscosity_strength: f32 = 0.03, "fluid.viscosity_strength", "viscosity",
        "Viscosity force strength";
    particles_per_cell: i32 = 4, "fluid.particles_per_cell", "particles-per-cell",
        "Initial particles per grid cell";
    max_particles_per_cell: i32 = 128, "fluid.max_particles_per_cell", "max-particles-per-cell",
        "Maximum particles per grid cell";
    world_width: f32 = 32.0, "world.width", "world-width", "World width in meters";
    world_height: f32 = 16.0, "world.height", "world-height", "World height in meters";
    soil_cell_size: f32 = 0.1, "world.soil_cell_size", "soil-cell-size",
        "Soil grid cell size in meters";
    capillary_mult: f32 = 1.0, "fluid.capillary_mult", "capillary-mult",
        "Global multiplier for capillary suction force";
    terrain_mode: i32 = 0, "world.terrain_mode", "terrain-mode",
        "Terrain mode (0=normal, 1=capillary test)";
    evap_rate: f32 = 0.01, "fluid.evap_rate", "evap-rate", "Evaporation rate scaling factor";
    condense_rate: f32 = 0.005, "fluid.condense_rate", "condense-rate",
        "Condensation rate scaling factor";
    vapor_buoyancy: f32 = 3.0, "fluid.vapor_buoyancy", "vapor-buoyancy",
        "Upward acceleration for vapor particles";
    vapor_drift: f32 = 0.5, "fluid.vapor_drift", "vapor-drift",
        "Random horizontal drift strength for vapor";
    condense_altitude_power: f32 = 2.0, "fluid.condense_altitude_power", "condense-alt-power",
        "Power curve for altitude-based condensation";
    seed: i64 = 0, "world.seed", "seed", "RNG seed (0 = random)";
}

impl SimParams {
  /// Resolve `seed = 0` to a value drawn from the OS.
  pub fn resolve_seed(&self) -> u64 {
    if self.seed == 0 {
      // No `rand` dependency here: the sim's own randomness is
      // counter-based, this is only the one-off "pick something".
      std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9e37_79b9_7f4a_7c15)
    } else {
      self.seed as u64
    }
  }

  /// The text `--write-config` emits: one block per TOML section, keys in
  /// declaration order, trailing comments aligned within the section.
  ///
  /// Each section is emitted once. The C++ writer instead groups *runs* of
  /// consecutive parameters, so its output repeats `[fluid]` and `[world]`
  /// and its own loader rejects the file it just wrote.
  pub fn default_config_text() -> String {
    let params = Self::default();
    let lines = params.config_lines();
    let mut out = String::from("# ALife simulation configuration\n\n");

    let mut sections: Vec<&'static str> = Vec::new();
    for (section, ..) in &lines {
      if !sections.contains(section) {
        sections.push(section);
      }
    }

    for section in sections {
      let in_section: Vec<_> = lines.iter().filter(|(s, ..)| *s == section).collect();
      let max_kv = in_section
        .iter()
        .map(|(_, key, value, _)| key.len() + 3 + value.len())
        .max()
        .unwrap_or(0);

      let _ = writeln!(out, "[{section}]");
      for (_, key, value, desc) in in_section {
        let kv = format!("{key} = {value}");
        let pad = (max_kv + 2).saturating_sub(kv.len()).max(1);
        let _ = writeln!(out, "{kv}{:pad$}# {desc}", "", pad = pad);
      }
      out.push('\n');
    }
    out
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn cli_overrides_toml_overrides_default() {
    let mut params = SimParams::default();
    assert_eq!(params.gravity, -13.0);

    let table: toml::Table = "[fluid]\ngravity = -9.81\n".parse().unwrap();
    params.apply_toml(&table);
    assert_eq!(params.gravity, -9.81);

    let cli = SimParamsCli {
      gravity: Some(-1.0),
      ..Default::default()
    };
    params.apply_cli(&cli);
    assert_eq!(params.gravity, -1.0);
  }

  #[test]
  fn default_config_round_trips() {
    let text = SimParams::default_config_text();
    let table: toml::Table = text.parse().expect("emitted config parses");
    let mut params = SimParams {
      gravity: 0.0,
      ..SimParams::default()
    };
    params.apply_toml(&table);
    assert_eq!(params, SimParams::default());
  }
}
