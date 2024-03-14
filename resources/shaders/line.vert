// line.vert
// uses sdf to draw a line
// quad should be just big enough to contain the line
#version 430 core

layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;
layout (location = 2) in float aLength; // start is always 0,0
layout (location = 3) in float aRadius;
layout (location = 4) in vec4 aColor;

out vec2 texCoord;
out float lineLength;
out float radius;
out vec4 color;

uniform mat4 transform;

void main() {
    texCoord = aTexCoord;
    lineLength = aLength;
    radius = aRadius;
    color = aColor;
    gl_Position = transform * vec4(aPos, 0.0, 1.0);
}