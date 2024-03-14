//
// Created by jonah on 4/26/2023.
//

#pragma once


#include <memory>
#include <stack>
#include "Screen.h"
#include "Resources.cuh"

#include <random>

class Game {
public:
    void render(float dt);
    void resize(int width, int height);
    void pushScreen(std::shared_ptr<Screen> screen);
    void popScreen();
    void switchScreen(const std::shared_ptr<Screen>& screen);
    [[nodiscard]] bool isRunning() const;
    void stopGame();
    void handleInput(SDL_Event event);
    [[nodiscard]] unsigned int getSeed() const;
    Resources &getResources();

private:
    std::stack<std::shared_ptr<Screen>> screenStack;
    unsigned int seed{std::random_device{}()};
    Resources resources{seed};
    int width{}, height{};
    bool running{true};
};
