//
// Created by jonah on 3/11/2024.
//

#pragma once

#include <vector>

#include "systems/Screen.h"
#include "systems/Game.cuh"
#include "systems/Trees.cuh"
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
    ExtendViewport hud_vp{720, 720};

    trees2::TreeBatch read_tree{}, write_tree{};
    trees2::TreeBatchDevice read_tree_device{}, write_tree_device{};

    bool mixing{true};
    bool mutating_len_rot{false};
    bool updating_cpu{false};
    bool updating_parallel{false};
    bool mutating_pos{false};

    bool tree_vbo_buffered{false};

};
