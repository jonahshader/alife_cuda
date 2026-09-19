//! The windowed front end. Not built yet — headless is the working surface.

use alife_sim::{SimParams, runtime::AnySim};
use anyhow::Result;

pub fn run(_sim: AnySim, _params: SimParams) -> Result<()> {
  anyhow::bail!("the GUI is not wired up yet; run with --headless")
}
