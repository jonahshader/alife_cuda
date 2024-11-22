#version 430 core

layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aOffset;
layout (location = 2) in vec2 aSize;
layout (location = 3) in vec4 aColor;

out vec4 color;

uniform mat4 transform;

void main() {
    color = aColor;
    gl_Position = transform * vec4(aPos * aSize + aOffset, 0.0, 1.0);
}
