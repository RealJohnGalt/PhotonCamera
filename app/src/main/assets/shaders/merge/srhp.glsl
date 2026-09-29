
precision highp float;

layout(rgba16f, binding = 0) readonly  uniform highp image2D srHpIn;
layout(rgba16f, binding = 1) writeonly uniform highp image2D srHpOut;

// Normalization divisor (accumulated frame count); the merge2o and
// post-upscale gains apply on top of this normalized highpass.
// Noise coring thresholds (absolute, same units): magnitudes below srT0
// are averaged-noise residue and die; above srT1 fully pass. Real edges
// and text strokes carry coherent amplitude and survive.
uniform float srNorm;
uniform float srT0;
uniform float srT1;

#define LAYOUT //
LAYOUT
void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(srHpOut);
    if (coord.x >= size.x || coord.y >= size.y) return;

    // 3x3 box mean on the packed grid (~6x6 raw support: the band a 2x
    // upscale can actually recover). Edge-clamped fetches keep the border
    // exact; OOB imageLoad would be undefined.
    vec4 acc = vec4(0.0);
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            acc += imageLoad(srHpIn, clamp(coord + ivec2(i, j), ivec2(0), size - ivec2(1)));
        }
    }
    vec4 c = imageLoad(srHpIn, coord);
    vec4 hp = (c - acc * (1.0 / 9.0)) * srNorm;
    vec4 gate = smoothstep(vec4(srT0), vec4(srT1), abs(hp));
    vec4 outv = hp * gate;
    // Kill NaN/Inf at the source: a poisoned residual would otherwise ride
    // every downstream consumer (merge2o, export, post) as black pixels.
    bvec4 finite = lessThan(abs(outv), vec4(65504.0));
    outv = mix(vec4(0.0), outv, vec4(finite));
    imageStore(srHpOut, coord, outv);
}
