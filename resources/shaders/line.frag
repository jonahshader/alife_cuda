#version 430 core

in vec2 texCoord;
in float lineLength;
in float radius;
in vec4 color;

out vec4 FragColor;

float screenPxRange(vec2 pos) {
    vec2 unitRange = vec2(1.0, 1.0);
    vec2 screenTexSize = vec2(1.0)/fwidth(pos);
    return max(0.5*dot(unitRange, screenTexSize), 1.0);
}
float sdVerticalCapsule( vec2 p, float h, float r )
{
    p.y -= clamp( p.y, 0.0, h );
    return length(p) - r;
}

void main() {
    float d = -sdVerticalCapsule(texCoord, lineLength, radius);
    float screenPxDistance = screenPxRange(texCoord)*(d);
    float opacity = clamp(screenPxDistance + 0.5, 0.0, 1.0);
    FragColor = vec4(color.rgb, opacity*color.a);
}
