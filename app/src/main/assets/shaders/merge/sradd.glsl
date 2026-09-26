
precision highp float;

layout(rgba16f, binding = 0) readonly  uniform highp image2D srAccIn;
layout(rgba16f, binding = 1) readonly  uniform highp image2D srDiffIn;
layout(rgba16f, binding = 2) writeonly uniform highp image2D srAccOut;

// Per-texel clamp on the signed contribution in normalized units: consistent
// subpixel residuals (true detail) pass far below it, while motion and
// misalignments (large diffs) saturate to a bounded term instead of
// ghosting. First-frame flag seeds the uninitialized accumulator.
uniform float srClamp;
uniform int srFirst;

#define LAYOUT //
LAYOUT
void main() {
    ivec2 coord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(srAccOut);
    if (coord.x >= size.x || coord.y >= size.y) return;

    vec4 diff = imageLoad(srDiffIn, coord);
    vec4 add = clamp(diff, vec4(-srClamp), vec4(srClamp));
    vec4 acc = srFirst != 0 ? vec4(0.0) : imageLoad(srAccIn, coord);
    imageStore(srAccOut, coord, acc + add);
}
