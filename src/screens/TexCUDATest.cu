#include "TexCUDATest.cuh"

TexCUDATest::TexCUDATest(Game &game) : game(game) {}

void TexCUDATest::show() {

}

__global__
void write_tex_test(unsigned char* data, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int index = (y * width + x) * channels;
        data[index] = 255;
        data[index + 1] = 0;
        data[index + 2] = 0;
    }
}

void TexCUDATest::render(float dt) {
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // auto& bold = game.getResources().extra_bold_font;
    // auto& rect = game.getResources().rect_renderer;
    // auto& line = game.getResources().line_renderer;
    // bold.set_transform(vp.get_transform());
    // rect.set_transform(vp.get_transform());
    // line.set_transform(vp.get_transform());

    rect.cuda_register_buffer();
    auto cuda_resource = rect.cuda_map_buffer();
    unsigned char* data = (unsigned char*)cuda_resource;
    auto width = rect.get_width();
    auto height = rect.get_height();
    auto channels = rect.get_channels();
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    write_tex_test<<<grid, block>>>(data, width, height, channels);

    rect.set_transform(vp.get_transform());

    rect.begin();
    rect.add_rect(0.0f, 0.0f, 32.0f, 16.0f, glm::vec3(1.0f, 1.0f, 1.0f));
    rect.end();

    rect.render();

    rect.cuda_unmap_buffer();
    rect.cuda_unregister_buffer();
}

void TexCUDATest::resize(int width, int height) {
    vp.update(width, height);
    hud_vp.update(width, height);
}

void TexCUDATest::hide() {

}


void TexCUDATest::handleInput(SDL_Event event) {
    if (event.type == SDL_KEYDOWN) {
        switch (event.key.keysym.sym) {
            case SDLK_ESCAPE:
                game.stopGame();
                break;
            default:
                break;
        }
    } else if (event.type == SDL_MOUSEWHEEL) {
        vp.handle_scroll(event.wheel.y);
    } else if (event.type == SDL_MOUSEMOTION) {
        if (event.motion.state & SDL_BUTTON_LMASK) {
            vp.handle_pan(event.motion.xrel, event.motion.yrel);
        }
    }
}