precision highp float;
precision highp sampler2D;
uniform highp sampler2D inTexture;
uniform highp sampler2D alignmentTexture;
uniform int yOffset;
// Sensor red-site offset ((cfa%2, cfa/2)); passed as a uniform because GLProg
// clears its define list after every program load, so a CFAPATTERN define set
// once at pipeline start would never reach this late-bound shader.
uniform ivec2 cfaShift;
// Signed SR detail accumulator (packed RGBA16F, same grid as inTexture) and
// its normalized gain (detail strength / accumulated frames). Zero when the
// layer is inactive (single frame, no upscale, over budget, or failure),
// in which case the branch below is skipped and output is bit-identical.
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
        // Detail highpass on the packed grid: center minus 3x3 box mean
        // (~6x6 raw support, the band a 2x upscale can actually recover),
        // normalized by the frame count folded into srGain. Edge-clamped
        // fetches keep the border exact.
        ivec2 ps = textureSize(srDetail, 0);
        vec4 acc = vec4(0.0);
        for (int j = -1; j <= 1; j++) {
            for (int i = -1; i <= 1; i++) {
                acc += texelFetch(srDetail, clamp(pq + ivec2(i, j), ivec2(0), ps - ivec2(1)), 0);
            }
        }
        vec4 c = texelFetch(srDetail, clamp(pq, ivec2(0), ps - ivec2(1)), 0);
        vec4 hp = (c - acc * (1.0 / 9.0)) * srGain;
        det = hp[ch];
    }
    Output = clamp(bayer[ch] + det, 0.0, 1.0);
}
