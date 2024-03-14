#version 430 core

in vec2 texCoord;
in vec4 color;

uniform sampler2D msdf;
//uniform vec4 bgColor;
//uniform vec4 fgColor;

out vec4 FragColor;

float median(float r, float g, float b) {
    return max(min(r, g), min(max(r, g), b));
}

float screenPxRange() {
//    vec2 pxRange =
    vec2 unitRange = vec2(3.0, 3.0)/vec2(textureSize(msdf, 0));
    vec2 screenTexSize = vec2(1.0)/fwidth(texCoord);
    return max(0.5*dot(unitRange, screenTexSize), 1.0);
}

void main() {
    vec3 msd = texture(msdf, texCoord).rgb;
    float sd = median(msd.r, msd.g, msd.b);
    float screenPxDistance = screenPxRange()*(sd - 0.5);
    float opacity = clamp(screenPxDistance + 0.5, 0.0, 1.0);
    FragColor = vec4(color.rgb, opacity * color.a);
//    FragColor = vec4(texture(msdf, texCoord).rgb, 1.0);
//    FragColor = vec4(cos(texCoord.x * 50), sin(texCoord.y * 50), 1.0, 1.0);
}