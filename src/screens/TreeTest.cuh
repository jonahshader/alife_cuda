//
// Created by jonah on 3/11/2024.
//

#pragma once

#include <vector>

#include "systems/Screen.h"
#include "systems/Game.cuh"
#include "systems/Trees.cuh"
#include "graphics/ExtendViewport.h"

using namespace trees;

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
    ExtendViewport hud_vp{720, 720};

    TreeBatch read_tree{}, write_tree{};

    bool mixing{false};
    bool mutating_len_rot{false};
    bool updating_parallel{false};
    bool mutating_pos{false};

};
