#include <iostream>
#include "systems/Game.cuh"
#include "screens/TreeTest.cuh"
#include "screens/FluidTest.cuh"
#include "spatial_sort.cuh"

int main() {
    Game game;
    game.resize(1280, 720);
    game.pushScreen(std::make_shared<TreeTest>(game));

    while (game.isRunning()) {
        game.render(1/165.0f);
    }

    return 0;
}