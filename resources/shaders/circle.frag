#version 430 core

in vec2 originalPos;
in vec4 color;

out vec4 FragColor;

//float dist2(vec2 a, vec2 b) {
//    vec2 c = a - b;
//    return dot(c, c);
//}

float screenPxRange() {
    //    vec2 pxRange =
    vec2 unitRange = vec2(1.0, 1.0);
    vec2 screenTexSize = vec2(1.0)/fwidth(originalPos);
    return max(0.5*dot(unitRange, screenTexSize), 1.0);
}

void main() {
//    if (dot(originalPos, originalPos) > .25) discard;
//    FragColor = vec4(color.rgb, 1);

    float d = 1-length(originalPos);
    float screenPxDistance = screenPxRange()*(d - 0.5);
    float opacity = clamp(screenPxDistance + 0.5, 0.0, 1.0);
    FragColor = vec4(color.rgb, opacity * color.a);
}