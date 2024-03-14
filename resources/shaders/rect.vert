#version 430 core

layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aOffset;
layout (location = 2) in vec2 aSize;
layout (location = 3) in float aRadius;
layout (location = 4) in vec4 aColor;

out vec2 originalPos;
out vec2 size;
out vec4 color;
out float radius;

uniform mat4 transform;

void main() {
    originalPos = aPos;
    color = aColor;
    radius = aRadius;
    size = aSize;
    gl_Position = transform * vec4(aPos * aSize + aOffset, 0.0, 1.0);
}