//
// Created by jonah on 4/26/2023.
//

#include "Game.cuh"


void Game::render(float dt) {
    if (!screenStack.empty())
        screenStack.top()->render(dt);
}

void Game::resize(int width, int height) {
    if (!screenStack.empty())
        screenStack.top()->resize(width, height);
    this->width = width;
    this->height = height;
}

void Game::pushScreen(std::shared_ptr<Screen> screen) {
    screenStack.push(screen);
    screenStack.top()->resize(width, height);
}

void Game::popScreen() {
    if (!screenStack.empty()) {
        screenStack.top()->hide();
        screenStack.pop();
        screenStack.top()->show();
        screenStack.top()->resize(width, height);
    }
}

void Game::switchScreen(const std::shared_ptr<Screen>& screen) {
    if (!screenStack.empty()) {
        screenStack.top()->hide();
        screenStack.pop();
    }
    screenStack.push(screen);
    screenStack.top()->show();
    screenStack.top()->resize(width, height);
}


Resources &Game::getResources() {
    return resources;
}

bool Game::isRunning() const {
    return running;
}

void Game::stopGame() {
    running = false;
}

void Game::handleInput(SDL_Event event) {
    if (!screenStack.empty())
        screenStack.top()->handleInput(event);
}

unsigned int Game::getSeed() const {
    return seed;
}


