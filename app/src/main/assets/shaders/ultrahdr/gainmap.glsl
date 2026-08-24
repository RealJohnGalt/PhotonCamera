precision highp float;
precision highp sampler2D;

// SDR base rendition (display-encoded sRGB, RGBA8)
uniform sampler2D InputBuffer;
// Linear-sRGB HDR rendition (RGBA16F, >1 allowed)
uniform sampler2D HDRBuffer;
// Downsample factor per axis (e.g. 4 -> gain map is 1/16 the pixels)
uniform int uDown;
// Total log2 range the gain map covers: logBoost in [0, uScale]
uniform float uScale;
// Linear-space epsilon to avoid divides by zero
uniform float uEps;

out vec4 Output;

// sRGB EOTF (display encoded -> linear). Must match Initial.glsl so the HDR
// rendition is consistent with the SDR base.
float srgbToLinear(float c) {
    c = clamp(c, 0.0, 1.0);
    if (c <= 0.04045) return c / 12.92;
    return pow((c + 0.055) / 1.055, 2.4);
}

vec3 srgbToLinear(vec3 c) {
    return vec3(
        srgbToLinear(c.r),
        srgbToLinear(c.g),
        srgbToLinear(c.b)
    );
}

void main() {
    ivec2 outXY = ivec2(gl_FragCoord.xy);
    int sx = outXY.x * uDown;
    int sy = outXY.y * uDown;

    float sdrSum = 0.0;
    float hdrSum = 0.0;
    int count = 0;

    for (int dy = 0; dy < uDown; dy++) {
        for (int dx = 0; dx < uDown; dx++) {
            ivec2 p = ivec2(sx + dx, sy + dy);
            vec4 s = texelFetch(InputBuffer, p, 0);
            vec4 h = texelFetch(HDRBuffer, p, 0);

            // SDR luminance in linear light.
            vec3 sLin = srgbToLinear(s.rgb);
            float sL = 0.2126 * sLin.r +
                       0.7152 * sLin.g +
                       0.0722 * sLin.b;

            // HDR luminance is already linear.
            float hL = 0.2126 * h.r +
                       0.7152 * h.g +
                       0.0722 * h.b;

            // Decode applies (SDR + OffsetSDR) * 2^gain - OffsetHDR, so the
            // encode-side ratio must add the offset, not clamp to it.
            sdrSum += sL + uEps;
            hdrSum += hL + uEps;
            count++;
        }
    }

    float sdrL = sdrSum / float(count);
    float hdrL = hdrSum / float(count);

    // By construction HDR >= SDR outside quantization noise; clamp negative boosts
    // to identity so shadows/midtones don't get inverted/brightened.
    float logBoost = log2(sdrL / hdrL);
    logBoost = max(logBoost, 0.0);

    // Map into [0,1] across the fixed log2 range [0, uScale], then store as 8-bit.
    float v = logBoost / uScale;
    v = clamp(v, 0.0, 1.0);
    int byteVal = int(v * 255.0 + 0.5);

    // Single-channel gain map, packed as R=G=B=value, A=255 (ARGB_8888).
    float chan = float(byteVal) / 255.0;
    Output = vec4(chan, chan, chan, 1.0);
}
