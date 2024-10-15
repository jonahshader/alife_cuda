#pragma once

#include "systems/DefaultScreen.cuh"
#include "systems/ParameterManager.h"

class FluidTest2 : public DefaultScreen
{
public:
  explicit FluidTest2(Game &game);

  void render(float dt) override;

private:
  ParameterManager pm{"fluid2_params.txt"};
  float dt{0.05f};
  float particle_mass{1.0f};
  float kernel_radius{0.1f};

};