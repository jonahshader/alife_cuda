//! Counter-based randomness: Threefry4x32-20.
//!
//! The C++ tree uses Random123's `r123::Threefry4x32` with its default 20
//! rounds, keyed per particle and countered per kernel launch. This is a
//! transcription of Random123's `_threefry4x_tpl(32)` macro, which is itself
//! fully unrolled; the rounds below are generated from the same rotation
//! constants, so the two agree bit for bit (checked against the C++ in
//! `tests/threefry_parity.rs`).

use cubecl::prelude::*;

// The rotations below are spelled out as shift-or rather than
// `u32::rotate_left`: they run inside `#[cube]`, which offers the shift
// operators and not Rust's integer methods.

/// Skein key-schedule parity constant for the 32-bit word size.
const KS_PARITY: u32 = 0x1BD1_1BDA;

/// `r123::u01<float>`: `x * 2^-32 + 2^-33`, uniform in (0, 1].
const U01_FACTOR: f32 = 2.328_306_4e-10;
const U01_HALF_FACTOR: f32 = 1.164_153_2e-10;

// The rotations are spelled out as shift-or rather than `rotate_left`:
// these run inside `#[cube]`, where the kernel language offers shifts and
// not Rust's integer methods.
/// The four output words of one Threefry block.
#[derive(CubeType, Clone, Copy)]
pub struct Rand4 {
  pub x0: u32,
  pub x1: u32,
  pub x2: u32,
  pub x3: u32,
}

#[cube]
#[allow(clippy::manual_rotate)]
fn rotl_5(x: u32) -> u32 {
  (x << 5u32) | (x >> 27u32)
}

#[cube]
#[allow(clippy::manual_rotate)]
fn rotl_6(x: u32) -> u32 {
  (x << 6u32) | (x >> 26u32)
}

#[cube]
#[allow(clippy::manual_rotate)]
fn rotl_10(x: u32) -> u32 {
  (x << 10u32) | (x >> 22u32)
}

#[cube]
#[allow(clippy::manual_rotate)]
fn rotl_11(x: u32) -> u32 {
  (x << 11u32) | (x >> 21u32)
}

#[cube]
#[allow(clippy::manual_rotate)]
fn rotl_13(x: u32) -> u32 {
  (x << 13u32) | (x >> 19u32)
}

#[cube]
#[allow(clippy::manual_rotate)]
fn rotl_17(x: u32) -> u32 {
  (x << 17u32) | (x >> 15u32)
}

#[cube]
#[allow(clippy::manual_rotate)]
fn rotl_18(x: u32) -> u32 {
  (x << 18u32) | (x >> 14u32)
}

#[cube]
#[allow(clippy::manual_rotate)]
fn rotl_20(x: u32) -> u32 {
  (x << 20u32) | (x >> 12u32)
}

#[cube]
#[allow(clippy::manual_rotate)]
fn rotl_21(x: u32) -> u32 {
  (x << 21u32) | (x >> 11u32)
}

#[cube]
#[allow(clippy::manual_rotate)]
fn rotl_23(x: u32) -> u32 {
  (x << 23u32) | (x >> 9u32)
}

#[cube]
#[allow(clippy::manual_rotate)]
fn rotl_25(x: u32) -> u32 {
  (x << 25u32) | (x >> 7u32)
}

#[cube]
#[allow(clippy::manual_rotate)]
fn rotl_26(x: u32) -> u32 {
  (x << 26u32) | (x >> 6u32)
}

#[cube]
#[allow(clippy::manual_rotate)]
fn rotl_27(x: u32) -> u32 {
  (x << 27u32) | (x >> 5u32)
}

/// Threefry4x32 with 20 rounds, as a device function.
#[cube]
pub fn threefry4x32_20(
  c0: u32,
  c1: u32,
  c2: u32,
  c3: u32,
  k0: u32,
  k1: u32,
  k2: u32,
  k3: u32,
) -> Rand4 {
  let ks0 = k0;
  let ks1 = k1;
  let ks2 = k2;
  let ks3 = k3;
  let ks4 = KS_PARITY ^ k0 ^ k1 ^ k2 ^ k3;

  let mut x0 = c0 + ks0;
  let mut x1 = c1 + ks1;
  let mut x2 = c2 + ks2;
  let mut x3 = c3 + ks3;

  x0 += x1;
  x1 = rotl_10(x1);
  x1 ^= x0;
  x2 += x3;
  x3 = rotl_26(x3);
  x3 ^= x2;
  x0 += x3;
  x3 = rotl_11(x3);
  x3 ^= x0;
  x2 += x1;
  x1 = rotl_21(x1);
  x1 ^= x2;
  x0 += x1;
  x1 = rotl_13(x1);
  x1 ^= x0;
  x2 += x3;
  x3 = rotl_27(x3);
  x3 ^= x2;
  x0 += x3;
  x3 = rotl_23(x3);
  x3 ^= x0;
  x2 += x1;
  x1 = rotl_5(x1);
  x1 ^= x2;
  // InjectKey(r=1)
  x0 += ks1;
  x1 += ks2;
  x2 += ks3;
  x3 += ks4;
  x3 += 1u32;
  x0 += x1;
  x1 = rotl_6(x1);
  x1 ^= x0;
  x2 += x3;
  x3 = rotl_20(x3);
  x3 ^= x2;
  x0 += x3;
  x3 = rotl_17(x3);
  x3 ^= x0;
  x2 += x1;
  x1 = rotl_11(x1);
  x1 ^= x2;
  x0 += x1;
  x1 = rotl_25(x1);
  x1 ^= x0;
  x2 += x3;
  x3 = rotl_10(x3);
  x3 ^= x2;
  x0 += x3;
  x3 = rotl_18(x3);
  x3 ^= x0;
  x2 += x1;
  x1 = rotl_20(x1);
  x1 ^= x2;
  // InjectKey(r=2)
  x0 += ks2;
  x1 += ks3;
  x2 += ks4;
  x3 += ks0;
  x3 += 2u32;
  x0 += x1;
  x1 = rotl_10(x1);
  x1 ^= x0;
  x2 += x3;
  x3 = rotl_26(x3);
  x3 ^= x2;
  x0 += x3;
  x3 = rotl_11(x3);
  x3 ^= x0;
  x2 += x1;
  x1 = rotl_21(x1);
  x1 ^= x2;
  x0 += x1;
  x1 = rotl_13(x1);
  x1 ^= x0;
  x2 += x3;
  x3 = rotl_27(x3);
  x3 ^= x2;
  x0 += x3;
  x3 = rotl_23(x3);
  x3 ^= x0;
  x2 += x1;
  x1 = rotl_5(x1);
  x1 ^= x2;
  // InjectKey(r=3)
  x0 += ks3;
  x1 += ks4;
  x2 += ks0;
  x3 += ks1;
  x3 += 3u32;
  x0 += x1;
  x1 = rotl_6(x1);
  x1 ^= x0;
  x2 += x3;
  x3 = rotl_20(x3);
  x3 ^= x2;
  x0 += x3;
  x3 = rotl_17(x3);
  x3 ^= x0;
  x2 += x1;
  x1 = rotl_11(x1);
  x1 ^= x2;
  x0 += x1;
  x1 = rotl_25(x1);
  x1 ^= x0;
  x2 += x3;
  x3 = rotl_10(x3);
  x3 ^= x2;
  x0 += x3;
  x3 = rotl_18(x3);
  x3 ^= x0;
  x2 += x1;
  x1 = rotl_20(x1);
  x1 ^= x2;
  // InjectKey(r=4)
  x0 += ks4;
  x1 += ks0;
  x2 += ks1;
  x3 += ks2;
  x3 += 4u32;
  x0 += x1;
  x1 = rotl_10(x1);
  x1 ^= x0;
  x2 += x3;
  x3 = rotl_26(x3);
  x3 ^= x2;
  x0 += x3;
  x3 = rotl_11(x3);
  x3 ^= x0;
  x2 += x1;
  x1 = rotl_21(x1);
  x1 ^= x2;
  x0 += x1;
  x1 = rotl_13(x1);
  x1 ^= x0;
  x2 += x3;
  x3 = rotl_27(x3);
  x3 ^= x2;
  x0 += x3;
  x3 = rotl_23(x3);
  x3 ^= x0;
  x2 += x1;
  x1 = rotl_5(x1);
  x1 ^= x2;
  // InjectKey(r=5)
  x0 += ks0;
  x1 += ks1;
  x2 += ks2;
  x3 += ks3;
  x3 += 5u32;

  Rand4 { x0, x1, x2, x3 }
}

/// `r123::u01<float>(x)`: uniform in (0, 1], never exactly 0.
#[cube]
pub fn u01(x: u32) -> f32 {
  x as f32 * U01_FACTOR + U01_HALF_FACTOR
}

/// Plain-Rust reference for [`threefry4x32_20`], and the host-side generator
/// the kernel references use.
pub fn threefry4x32_20_ref(ctr: [u32; 4], key: [u32; 4]) -> [u32; 4] {
  let ks0 = key[0];
  let ks1 = key[1];
  let ks2 = key[2];
  let ks3 = key[3];
  let ks4 = KS_PARITY ^ key[0] ^ key[1] ^ key[2] ^ key[3];

  let mut x0 = ctr[0].wrapping_add(ks0);
  let mut x1 = ctr[1].wrapping_add(ks1);
  let mut x2 = ctr[2].wrapping_add(ks2);
  let mut x3 = ctr[3].wrapping_add(ks3);

  x0 = x0.wrapping_add(x1);
  x1 = x1.rotate_left(10);
  x1 ^= x0;
  x2 = x2.wrapping_add(x3);
  x3 = x3.rotate_left(26);
  x3 ^= x2;
  x0 = x0.wrapping_add(x3);
  x3 = x3.rotate_left(11);
  x3 ^= x0;
  x2 = x2.wrapping_add(x1);
  x1 = x1.rotate_left(21);
  x1 ^= x2;
  x0 = x0.wrapping_add(x1);
  x1 = x1.rotate_left(13);
  x1 ^= x0;
  x2 = x2.wrapping_add(x3);
  x3 = x3.rotate_left(27);
  x3 ^= x2;
  x0 = x0.wrapping_add(x3);
  x3 = x3.rotate_left(23);
  x3 ^= x0;
  x2 = x2.wrapping_add(x1);
  x1 = x1.rotate_left(5);
  x1 ^= x2;
  // InjectKey(r=1)
  x0 = x0.wrapping_add(ks1);
  x1 = x1.wrapping_add(ks2);
  x2 = x2.wrapping_add(ks3);
  x3 = x3.wrapping_add(ks4);
  x3 = x3.wrapping_add(1u32);
  x0 = x0.wrapping_add(x1);
  x1 = x1.rotate_left(6);
  x1 ^= x0;
  x2 = x2.wrapping_add(x3);
  x3 = x3.rotate_left(20);
  x3 ^= x2;
  x0 = x0.wrapping_add(x3);
  x3 = x3.rotate_left(17);
  x3 ^= x0;
  x2 = x2.wrapping_add(x1);
  x1 = x1.rotate_left(11);
  x1 ^= x2;
  x0 = x0.wrapping_add(x1);
  x1 = x1.rotate_left(25);
  x1 ^= x0;
  x2 = x2.wrapping_add(x3);
  x3 = x3.rotate_left(10);
  x3 ^= x2;
  x0 = x0.wrapping_add(x3);
  x3 = x3.rotate_left(18);
  x3 ^= x0;
  x2 = x2.wrapping_add(x1);
  x1 = x1.rotate_left(20);
  x1 ^= x2;
  // InjectKey(r=2)
  x0 = x0.wrapping_add(ks2);
  x1 = x1.wrapping_add(ks3);
  x2 = x2.wrapping_add(ks4);
  x3 = x3.wrapping_add(ks0);
  x3 = x3.wrapping_add(2u32);
  x0 = x0.wrapping_add(x1);
  x1 = x1.rotate_left(10);
  x1 ^= x0;
  x2 = x2.wrapping_add(x3);
  x3 = x3.rotate_left(26);
  x3 ^= x2;
  x0 = x0.wrapping_add(x3);
  x3 = x3.rotate_left(11);
  x3 ^= x0;
  x2 = x2.wrapping_add(x1);
  x1 = x1.rotate_left(21);
  x1 ^= x2;
  x0 = x0.wrapping_add(x1);
  x1 = x1.rotate_left(13);
  x1 ^= x0;
  x2 = x2.wrapping_add(x3);
  x3 = x3.rotate_left(27);
  x3 ^= x2;
  x0 = x0.wrapping_add(x3);
  x3 = x3.rotate_left(23);
  x3 ^= x0;
  x2 = x2.wrapping_add(x1);
  x1 = x1.rotate_left(5);
  x1 ^= x2;
  // InjectKey(r=3)
  x0 = x0.wrapping_add(ks3);
  x1 = x1.wrapping_add(ks4);
  x2 = x2.wrapping_add(ks0);
  x3 = x3.wrapping_add(ks1);
  x3 = x3.wrapping_add(3u32);
  x0 = x0.wrapping_add(x1);
  x1 = x1.rotate_left(6);
  x1 ^= x0;
  x2 = x2.wrapping_add(x3);
  x3 = x3.rotate_left(20);
  x3 ^= x2;
  x0 = x0.wrapping_add(x3);
  x3 = x3.rotate_left(17);
  x3 ^= x0;
  x2 = x2.wrapping_add(x1);
  x1 = x1.rotate_left(11);
  x1 ^= x2;
  x0 = x0.wrapping_add(x1);
  x1 = x1.rotate_left(25);
  x1 ^= x0;
  x2 = x2.wrapping_add(x3);
  x3 = x3.rotate_left(10);
  x3 ^= x2;
  x0 = x0.wrapping_add(x3);
  x3 = x3.rotate_left(18);
  x3 ^= x0;
  x2 = x2.wrapping_add(x1);
  x1 = x1.rotate_left(20);
  x1 ^= x2;
  // InjectKey(r=4)
  x0 = x0.wrapping_add(ks4);
  x1 = x1.wrapping_add(ks0);
  x2 = x2.wrapping_add(ks1);
  x3 = x3.wrapping_add(ks2);
  x3 = x3.wrapping_add(4u32);
  x0 = x0.wrapping_add(x1);
  x1 = x1.rotate_left(10);
  x1 ^= x0;
  x2 = x2.wrapping_add(x3);
  x3 = x3.rotate_left(26);
  x3 ^= x2;
  x0 = x0.wrapping_add(x3);
  x3 = x3.rotate_left(11);
  x3 ^= x0;
  x2 = x2.wrapping_add(x1);
  x1 = x1.rotate_left(21);
  x1 ^= x2;
  x0 = x0.wrapping_add(x1);
  x1 = x1.rotate_left(13);
  x1 ^= x0;
  x2 = x2.wrapping_add(x3);
  x3 = x3.rotate_left(27);
  x3 ^= x2;
  x0 = x0.wrapping_add(x3);
  x3 = x3.rotate_left(23);
  x3 ^= x0;
  x2 = x2.wrapping_add(x1);
  x1 = x1.rotate_left(5);
  x1 ^= x2;
  // InjectKey(r=5)
  x0 = x0.wrapping_add(ks0);
  x1 = x1.wrapping_add(ks1);
  x2 = x2.wrapping_add(ks2);
  x3 = x3.wrapping_add(ks3);
  x3 = x3.wrapping_add(5u32);

  [x0, x1, x2, x3]
}

/// Plain-Rust reference for [`u01`].
pub fn u01_ref(x: u32) -> f32 {
  x as f32 * U01_FACTOR + U01_HALF_FACTOR
}

/// The per-launch counter, incremented like `r123array4x32::incr()`: a carry
/// chain starting at word 0.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RngCounter(pub [u32; 4]);

impl RngCounter {
  /// Where the counter stands after `steps` simulation steps. Each step
  /// consumes two values, one per RNG-using kernel.
  pub fn after_steps(steps: u32) -> Self {
    let mut ctr = Self::default();
    for _ in 0..(steps as u64 * 2) {
      ctr.incr();
    }
    ctr
  }

  pub fn incr(&mut self) {
    for word in self.0.iter_mut() {
      *word = word.wrapping_add(1);
      if *word != 0 {
        return;
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn counter_carries_like_random123() {
    let mut ctr = RngCounter([u32::MAX, 0, 0, 0]);
    ctr.incr();
    assert_eq!(ctr.0, [0, 1, 0, 0]);
  }

  #[test]
  fn u01_is_open_at_zero_and_closed_at_one() {
    assert!(u01_ref(0) > 0.0);
    assert!(u01_ref(u32::MAX) <= 1.0);
  }

  /// `(counter, key, expected)` printed by `r123::Threefry4x32` from the
  /// Random123 headers the C++ tree builds against.
  const RANDOM123_VECTORS: &[([u32; 4], [u32; 4], [u32; 4])] = &[
    (
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [2624366954, 3783175782, 4228967636, 1381410776],
    ),
    (
      [0, 0, 0, 0],
      [17, 1, 0, 0],
      [1900637895, 2209859351, 848054186, 1153971404],
    ),
    (
      [0, 0, 0, 0],
      [4294967295, 4294967295, 4294967295, 4294967295],
      [2539217096, 3231088984, 4042460720, 211643642],
    ),
    (
      [0, 0, 0, 0],
      [51199, 2, 0, 0],
      [120033002, 1975665344, 3447002123, 2275249799],
    ),
    (
      [0, 0, 0, 0],
      [99, 1, 0, 0],
      [1958672351, 351364218, 1680712183, 1882504937],
    ),
    (
      [1, 0, 0, 0],
      [0, 0, 0, 0],
      [1617335461, 1437161258, 674141268, 1102938572],
    ),
    (
      [1, 0, 0, 0],
      [17, 1, 0, 0],
      [3224505032, 345392900, 539703372, 444199289],
    ),
    (
      [1, 0, 0, 0],
      [4294967295, 4294967295, 4294967295, 4294967295],
      [1712308496, 1426319049, 1255765531, 3678027971],
    ),
    (
      [1, 0, 0, 0],
      [51199, 2, 0, 0],
      [3408273736, 334777799, 1376462330, 1721840545],
    ),
    (
      [1, 0, 0, 0],
      [99, 1, 0, 0],
      [2227681302, 4260027809, 3123308690, 1328240462],
    ),
    (
      [4294967295, 4294967295, 4294967295, 4294967295],
      [0, 0, 0, 0],
      [4235937169, 1604547912, 2928385169, 3391890648],
    ),
    (
      [4294967295, 4294967295, 4294967295, 4294967295],
      [17, 1, 0, 0],
      [494672533, 3434183182, 744580762, 1314371458],
    ),
    (
      [4294967295, 4294967295, 4294967295, 4294967295],
      [4294967295, 4294967295, 4294967295, 4294967295],
      [713561750, 1459692167, 4140254318, 2708105010],
    ),
    (
      [4294967295, 4294967295, 4294967295, 4294967295],
      [51199, 2, 0, 0],
      [1772200848, 3461742559, 2169039890, 2279169592],
    ),
    (
      [4294967295, 4294967295, 4294967295, 4294967295],
      [99, 1, 0, 0],
      [39922490, 2605237759, 856352327, 2839346228],
    ),
    (
      [7, 0, 0, 0],
      [0, 0, 0, 0],
      [2900134268, 3389447134, 365879591, 1810017690],
    ),
    (
      [7, 0, 0, 0],
      [17, 1, 0, 0],
      [1476270434, 828491930, 2517866932, 4129765472],
    ),
    (
      [7, 0, 0, 0],
      [4294967295, 4294967295, 4294967295, 4294967295],
      [3410902520, 4247125805, 343539648, 779502512],
    ),
    (
      [7, 0, 0, 0],
      [51199, 2, 0, 0],
      [3467127192, 3910063297, 450263205, 4272636344],
    ),
    (
      [7, 0, 0, 0],
      [99, 1, 0, 0],
      [3374160965, 2557746287, 984918781, 1719546756],
    ),
    (
      [123456, 0, 0, 0],
      [0, 0, 0, 0],
      [449853086, 1167338143, 1256242291, 2542011980],
    ),
    (
      [123456, 0, 0, 0],
      [17, 1, 0, 0],
      [2322170439, 221203093, 3192167309, 4092993435],
    ),
    (
      [123456, 0, 0, 0],
      [4294967295, 4294967295, 4294967295, 4294967295],
      [3205681534, 3017696448, 2253649930, 1617724250],
    ),
    (
      [123456, 0, 0, 0],
      [51199, 2, 0, 0],
      [1192800103, 1916170633, 67047865, 390460339],
    ),
    (
      [123456, 0, 0, 0],
      [99, 1, 0, 0],
      [1904087105, 1687466961, 191545359, 999595747],
    ),
  ];

  #[test]
  fn matches_random123() {
    for (ctr, key, expected) in RANDOM123_VECTORS {
      assert_eq!(
        &threefry4x32_20_ref(*ctr, *key),
        expected,
        "ctr {ctr:?} key {key:?}"
      );
    }
  }

  #[test]
  fn u01_matches_random123() {
    // Bit patterns, so the comparison catches a differently rounded constant.
    const SAMPLES: &[(u32, u32)] = &[
      (0, 0x2f000000),
      (1, 0x2fc00000),
      (2147483647, 0x3f000000),
      (2147483648, 0x3f000000),
      (4294967295, 0x3f800000),
      (3141592653, 0x3f3b40e6),
    ];
    for (input, bits) in SAMPLES {
      assert_eq!(u01_ref(*input).to_bits(), *bits, "u01({input})");
    }
  }
}
