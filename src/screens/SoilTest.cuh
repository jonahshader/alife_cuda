#pragma once

#include "systems/Screen.h"
#include "systems/Game.cuh"
#include "systems/Soil.cuh"
#include "graphics/ExtendViewport.h"

class SoilTest : public Screen {
public:
    explicit SoilTest(Game &game);

    void show() override;
    void render(float dt) override;
    void resize(int width, int height) override;
    void hide() override;
    void handleInput(SDL_Event event) override;

private:
    Game& game;
    ExtendViewport vp{720, 720};
    ExtendViewport hud_vp{720, 720};

    Soil soil;

};
