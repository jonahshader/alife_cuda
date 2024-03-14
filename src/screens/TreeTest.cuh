//
// Created by jonah on 3/11/2024.
//

#pragma once

#include "systems/Screen.h"
#include "systems/Game.cuh"
#include "graphics/ExtendViewport.h"

class TreeTest : public Screen {
public:
    explicit TreeTest(Game &game);

    void show() override;
    void render(float dt) override;
    void resize(int width, int height) override;
    void hide() override;
    void handleInput(SDL_Event event) override;

private:
    Game& game;
    ExtendViewport vp{720, 720};

};
