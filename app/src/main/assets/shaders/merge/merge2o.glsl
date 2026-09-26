precision highp float;
precision highp sampler2D;
uniform highp sampler2D inTexture;
uniform highp sampler2D alignmentTexture;
uniform int yOffset;
// Sensor red-site offset ((cfa%2, cfa/2)); passed as a uniform because GLProg
// clears its define list after every program load, so a CFAPATTERN define set
// once at pipeline start would never reach this late-bound shader.
uniform ivec2 cfaShift;
// Signed SR detail highpass (packed RGBA16F, precomputed by srhp with the
// frame-count normalization folded in) and its gain. Zero when the layer is
// inactive (single frame, no upscale, over budget, or failure), in which
// case the branch below is skipped and output is bit-identical.
uniform highp sampler2D srDetail;
uniform float srGain;
#define TILE 2
#define CONCAT 1
out float Output;

// The merged base is already normalized fp16 (rgba16f, clamped to [0,1] by
// the merge stages); this pass only unpacks the 2x2 quads back onto the raw
// Bayer grid. No white-level re-encode or quantization dither is needed - the
// output pipeline renders straight into an R16F buffer.

void main() {
    ivec2 xy = ivec2(gl_FragCoord.xy);
    xy += ivec2(0, yOffset);
    // Undo the merge00 packing shift: real raw site X lives at packed
    // rel = X + cfaShift (merge00 shifted quad origins by -cfaShift and
    // filled the out-of-range sites with edge duplicates, which land at
    // rel < cfaShift and rel > rawSize-1 and are simply never read here;
    // cfaShift is zero for RGGB/BGGR, so this is the identity for them).
    ivec2 rel = xy + cfaShift;
    ivec2 pq = rel / TILE;
    vec4 bayer = texelFetch(inTexture, pq, 0);
    int ch = (rel.x & 1) + (rel.y & 1) * TILE;
    float det = 0.0;
    if (srGain > 0.0) {
        // Precomputed highpass sampled at this site's packed quad channel.
        vec4 hp = texelFetch(srDetail, clamp(pq, ivec2(0), textureSize(srDetail, 0) - ivec2(1)), 0);
        det = hp[ch] * srGain;
    }
    Output = clamp(bayer[ch] + det, 0.0, 1.0);
}
