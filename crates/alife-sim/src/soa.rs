//! One field list per structure-of-arrays type.
//!
//! Mirrors `DEFINE_STRUCTS(Name, FIELDS)` in the C++ tree's
//! `src/systems/soa_helper.h`: a single declaration generates the host struct,
//! the device buffers, the upload/download pair and the dump/load field order.
//! Add a field by adding a line, never by hand-editing the generated structs.

use bytemuck::Pod;
use glam::Vec2;

/// How one logical field is stored on the host, on the device, and in a dump.
///
/// The device element type is deliberately allowed to differ from the host
/// type: CubeCL kernels work in `f32`/`u32`, while the host side keeps the
/// meaningful type (`Vec2`, an enum) and the dump keeps the C++ field layout so
/// a dump written by either binary loads into the other.
pub trait SoaField: Copy + Default + 'static {
  /// Element type of the device buffer backing this field.
  type Device: Pod + Default;
  /// This field's representation in the binary dump — the C++ field type.
  type Raw: Pod + Default;
  /// Device elements per logical value (2 for a `Vec2`).
  const COMPONENTS: usize;

  fn write_device(self, out: &mut Vec<Self::Device>);
  fn read_device(src: &[Self::Device]) -> Self;
  fn to_raw(self) -> Self::Raw;
  fn from_raw(raw: Self::Raw) -> Self;
}

impl SoaField for f32 {
  type Device = f32;
  type Raw = f32;
  const COMPONENTS: usize = 1;

  fn write_device(self, out: &mut Vec<f32>) {
    out.push(self);
  }

  fn read_device(src: &[f32]) -> Self {
    src[0]
  }

  fn to_raw(self) -> f32 {
    self
  }

  fn from_raw(raw: f32) -> Self {
    raw
  }
}

impl SoaField for Vec2 {
  type Device = f32;
  /// `float2` in the C++ dump: two contiguous floats.
  type Raw = [f32; 2];
  const COMPONENTS: usize = 2;

  fn write_device(self, out: &mut Vec<f32>) {
    out.push(self.x);
    out.push(self.y);
  }

  fn read_device(src: &[f32]) -> Self {
    Vec2::new(src[0], src[1])
  }

  fn to_raw(self) -> [f32; 2] {
    [self.x, self.y]
  }

  fn from_raw(raw: [f32; 2]) -> Self {
    Vec2::new(raw[0], raw[1])
  }
}

/// A small fixed-width vector of floats — a limb's identity vector. The
/// components are contiguous on the device, so element `i` starts at
/// `i * N`. The `Default` bound is the std one, which reaches `N <= 32`.
impl<const N: usize> SoaField for [f32; N]
where
  [f32; N]: Default,
{
  type Device = f32;
  type Raw = [f32; N];
  const COMPONENTS: usize = N;

  fn write_device(self, out: &mut Vec<f32>) {
    out.extend_from_slice(&self);
  }

  fn read_device(src: &[f32]) -> Self {
    let mut out = [0.0; N];
    out.copy_from_slice(&src[..N]);
    out
  }

  fn to_raw(self) -> Self {
    self
  }

  fn from_raw(raw: Self) -> Self {
    raw
  }
}

/// A flag or an index that is already a kernel-native word: no conversion
/// either way.
impl SoaField for u32 {
  type Device = u32;
  type Raw = u32;
  const COMPONENTS: usize = 1;

  fn write_device(self, out: &mut Vec<u32>) {
    out.push(self);
  }

  fn read_device(src: &[u32]) -> Self {
    src[0]
  }

  fn to_raw(self) -> u32 {
    self
  }

  fn from_raw(raw: u32) -> Self {
    raw
  }
}

/// `uint8_t` in the C++ SoA. Kernels see a `u32`; the dump keeps the byte.
impl SoaField for u8 {
  type Device = u32;
  type Raw = u8;
  const COMPONENTS: usize = 1;

  fn write_device(self, out: &mut Vec<u32>) {
    out.push(self as u32);
  }

  fn read_device(src: &[u32]) -> Self {
    src[0] as u8
  }

  fn to_raw(self) -> u8 {
    self
  }

  fn from_raw(raw: u8) -> Self {
    raw
  }
}

/// Generate the host struct, the device buffers and the dump order for one SoA
/// type. See the module docs; the field order is the dump order.
#[macro_export]
macro_rules! define_soa {
    (
        $(#[$meta:meta])*
        $host:ident / $device:ident {
            $(#[$fmeta0:meta])* $field0:ident : $ty0:ty $(= $init0:expr)?,
            $( $(#[$fmeta:meta])* $field:ident : $ty:ty $(= $init:expr)?, )*
        }
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Default, PartialEq)]
        pub struct $host {
            $(#[$fmeta0])* pub $field0: Vec<$ty0>,
            $( $(#[$fmeta])* pub $field: Vec<$ty>, )*
        }

        impl $host {
            /// Bytes one element occupies in a dump: the sum of the fields'
            /// raw sizes, so a reader can check a file's length against its
            /// header before allocating anything.
            pub const RAW_BYTES: usize =
                ::core::mem::size_of::<<$ty0 as $crate::soa::SoaField>::Raw>()
                $( + ::core::mem::size_of::<<$ty as $crate::soa::SoaField>::Raw>() )*;

            /// `n` elements per field, at the field's declared initial value.
            pub fn new(n: usize) -> Self {
                Self {
                    $field0: vec![
                        {
                            #[allow(unused_variables)]
                            let v: $ty0 = <$ty0 as Default>::default();
                            $( let v: $ty0 = $init0; )?
                            v
                        };
                        n
                    ],
                    $( $field: vec![
                        {
                            #[allow(unused_variables)]
                            let v: $ty = <$ty as Default>::default();
                            $( let v: $ty = $init; )?
                            v
                        };
                        n
                    ], )*
                }
            }

            /// Element count. Every field has the same length by construction.
            pub fn len(&self) -> usize {
                self.$field0.len()
            }

            pub fn is_empty(&self) -> bool {
                self.len() == 0
            }

            /// Write the fields, in declaration order, as the flat
            /// little-endian arrays the C++ `--dump` writer emits.
            pub fn write_fields<W: std::io::Write>(&self, out: &mut W) -> std::io::Result<()> {
                $crate::soa::write_field(out, &self.$field0)?;
                $( $crate::soa::write_field(out, &self.$field)?; )*
                Ok(())
            }

            /// Read `n` elements per field, in declaration order.
            pub fn read_fields<R: std::io::Read>(
                input: &mut R,
                n: usize,
            ) -> std::io::Result<Self> {
                Ok(Self {
                    $field0: $crate::soa::read_field(input, n)?,
                    $( $field: $crate::soa::read_field(input, n)?, )*
                })
            }
        }

        /// Device-side mirror: one CubeCL buffer per field. A handle is a slice
        /// of a pooled buffer, so it is always bound with its own offset — see
        /// the spike outcome in `docs/organism.md`.
        #[derive(Debug, Clone)]
        pub struct $device {
            pub $field0: ::cubecl_runtime::server::Handle,
            $( pub $field: ::cubecl_runtime::server::Handle, )*
            len: usize,
        }

        impl $device {
            /// Logical element count (particles, cells, ...), not buffer length.
            pub fn len(&self) -> usize {
                self.len
            }

            pub fn is_empty(&self) -> bool {
                self.len == 0
            }

            pub fn upload<R: ::cubecl::prelude::Runtime>(
                client: &::cubecl::prelude::ComputeClient<R>,
                host: &$host,
            ) -> Self {
                Self {
                    $field0: $crate::soa::upload_field(client, &host.$field0),
                    $( $field: $crate::soa::upload_field(client, &host.$field), )*
                    len: host.len(),
                }
            }

            pub fn download<R: ::cubecl::prelude::Runtime>(
                &self,
                client: &::cubecl::prelude::ComputeClient<R>,
            ) -> $host {
                $host {
                    $field0: $crate::soa::download_field(client, &self.$field0, self.len),
                    $( $field: $crate::soa::download_field(client, &self.$field, self.len), )*
                }
            }
        }
    };
}

pub fn write_field<F: SoaField, W: std::io::Write>(
  out: &mut W,
  values: &[F],
) -> std::io::Result<()> {
  let raw: Vec<F::Raw> = values.iter().map(|v| v.to_raw()).collect();
  out.write_all(bytemuck::cast_slice(&raw))
}

pub fn read_field<F: SoaField, R: std::io::Read>(
  input: &mut R,
  n: usize,
) -> std::io::Result<Vec<F>> {
  let mut raw = vec![F::Raw::default(); n];
  input.read_exact(bytemuck::cast_slice_mut(&mut raw))?;
  Ok(raw.into_iter().map(F::from_raw).collect())
}

pub fn to_device_vec<F: SoaField>(values: &[F]) -> Vec<F::Device> {
  let mut out = Vec::with_capacity(values.len() * F::COMPONENTS);
  for v in values {
    v.write_device(&mut out);
  }
  out
}

pub fn upload_field<F: SoaField, R: cubecl::prelude::Runtime>(
  client: &cubecl::prelude::ComputeClient<R>,
  values: &[F],
) -> cubecl_runtime::server::Handle {
  client.create_from_slice(bytemuck::cast_slice(&to_device_vec(values)))
}

pub fn download_field<F: SoaField, R: cubecl::prelude::Runtime>(
  client: &cubecl::prelude::ComputeClient<R>,
  handle: &cubecl_runtime::server::Handle,
  n: usize,
) -> Vec<F> {
  let bytes = client.read_one_unchecked(handle.clone());
  let elems: &[F::Device] = bytemuck::cast_slice(&bytes);
  (0..n)
    .map(|i| F::read_device(&elems[i * F::COMPONENTS..(i + 1) * F::COMPONENTS]))
    .collect()
}
