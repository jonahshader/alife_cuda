#version 430 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexPos;

out vec2 a_texPos;

void main() {
    gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);
}