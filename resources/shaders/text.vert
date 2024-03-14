#version 430 core

layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;
layout (location = 2) in vec4 aColor;

uniform mat4 transform;

out vec2 texCoord;
out vec4 color;

void main() {
    texCoord = aTexCoord;
    color = aColor;
    gl_Position = transform * vec4(aPos, 0.0, 1.0); // hard-coding z to 1 for now
//    gl_Position = vec4(aPos, 1.0, 1.0); // hard-coding z to 1 for now
}