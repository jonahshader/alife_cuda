#version 430 core

in vec2 originalPos;
in vec2 size;
in vec4 color;
in float radius;

out vec4 FragColor;

float screenPxRange(vec2 pos) {
    vec2 unitRange = vec2(1.0, 1.0);
    vec2 screenTexSize = vec2(1.0)/fwidth(pos);
    return max(0.5*dot(unitRange, screenTexSize), 1.0);
}

float box(vec2 position, vec2 halfSize, float cornerRadius) {
    position = abs(position) - halfSize + cornerRadius;
    return length(max(position, 0.0)) + min(max(position.x, position.y), 0.0) - cornerRadius;
}


void main() {
//    float d = 1-box(originalPosScaled, size / 2, radius);
//    float screenPxDistance = screenPxRange()*(d - 0.5);
//    float opacity = clamp(screenPxDistance + 0.5, 0.0, 1.0);

    vec2 pos = originalPos * size;
    float d = -box(pos, size/2.0, radius);

//    float d = box(originalPosScaled - size/2, size/2, radius);
    float screenPxDistance = screenPxRange(pos)*(d - 0.5);
    float opacity = clamp(screenPxDistance + 0.5, 0.0, 1.0);
//    float opacity = smoothstep(0.0, 1.0, d);
    FragColor = vec4(color.rgb, opacity * color.a);
//    FragColor = vec4(cos(d*1000), sin(d), 0.0, 1.0);
}