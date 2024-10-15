#pragma once

#include "systems/DefaultScreen.cuh"
#include "systems/ParameterManager.h"

#include "systems/ParticleFluid2.cuh"

class FluidTest2 : public DefaultScreen
{
public:
  explicit FluidTest2(Game &game);

  void render(float dt) override;

private:
  // ParameterManager pm{"fluid2_params.txt"}; // todo: move to p2::ParticleFluid, combine with TunableParams

  p2::ParticleFluid fluid{20.0f, 15.0f, true};

};