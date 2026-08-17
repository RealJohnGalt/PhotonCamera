precision highp sampler2D;
precision highp float;
uniform sampler2D InputBuffer;
uniform sampler2D BrBuffer;
uniform float factor;
out vec2 result;
uniform int yOffset;
#define DH (0.0)
#define FUSIONGAIN 1.0
#define NORM 64.0
#define luminocity(x) dot(x.rgb, vec3(0.299, 0.587, 0.114))
float gammaInverse(float x) {
    return x*x;
}

vec4 reinhard_extended(vec4 v, float max_white) {
    vec4 numerator = v * (vec4(1.0f) + (v / vec4(max_white * max_white)));
    return numerator / (vec4(1.0f) + v);
}

void main() {
    ivec2 xy = ivec2(gl_FragCoord.xy);
    xy+=ivec2(0,yOffset);
    ivec2 inputSize = textureSize(InputBuffer, 0);
    ivec2 safePos = clamp(xy, ivec2(0), inputSize - ivec2(1));
    // The per-pixel gain ratio fused/base. Bounded in very dark regions where
    // base->0 and the raw quotient explodes; the full-resolution guided filter
    // in initial.glsl re-fits a local linear model against the luma guide, so
    // the map only needs to carry the bounded gain, not affine coefficients.
    float lowresVal  = clamp(
            (texelFetch(InputBuffer, safePos, 0).r + 0.001) /
            (texelFetch(BrBuffer, safePos, 0).r + 0.001), 0.0, 8.0);
    // /FUSIONGAIN so getGain()'s *FUSIONGAIN recovers the true gain; *factor
    // preserves the exposure-correction scaling of the legacy coefficient path.
    result = vec2(lowresVal / FUSIONGAIN, 0.0);
    result *= clamp(factor, 0.0, 1.0);
}
