precision highp float;
precision highp sampler2D;
precision highp image2D;
uniform highp sampler2D InputTexture;
uniform vec3 blackLevel;
out vec3 Output;
void main() {
    ivec2 xy = ivec2(gl_FragCoord.xy);
    vec3 color = texelFetch(InputTexture, xy, 0).rgb;
    // Normalize the color values based on the black level. Scene-referred:
    // keep the lower (black) normalization but do NOT clamp the top to 1, so
    // recoverable highlight headroom survives to the Ultra HDR gain-map pass
    // (sceneluma: "scene luminance may legitimately reach/exceed 1.0").
    color = max((color - blackLevel) / (vec3(1.0) - blackLevel), vec3(0.0));
    // Write the normalized color to the output
    Output = color.rgb;
}