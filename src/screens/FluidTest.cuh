//
// Created by jonah on 3/12/2024.
//

#pragma once

#include "systems/Screen.h"
#include "systems/Game.cuh"
#include "graphics/ExtendViewport.h"

#include "systems/Fluid.cuh"

using namespace fluid;

class FluidTest : public Screen {
public:
    explicit FluidTest(Game &game);

    void show() override;
    void render(float dt) override;
    void resize(int width, int height) override;
    void hide() override;
    void handleInput(SDL_Event event) override;

private:
    Game& game;
    ExtendViewport vp{720, 720};
    Cell* read_cells{nullptr};
    Cell* write_cells{nullptr};

    Fluid fluid;
};
