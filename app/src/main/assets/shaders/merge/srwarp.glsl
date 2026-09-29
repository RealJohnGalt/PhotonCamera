
#define LAYOUT //
LAYOUT
precision highp float;
precision highp sampler2D;
precision highp image2D;
// Drizzle accumulator in/out: the luma and weight halves packed into one
// R32UI word. ES 3.1 image load/store only guarantees 1- or 4-channel
// formats, so the pair travels packed - bit-identical to the previous
// RGBA16F storage, half the bytes. srFirst skips the read so the base frame
// seeds exactly.
layout(r32ui, binding = 0) readonly  uniform highp uimage2D srDriIn;
layout(r32ui, binding = 1) writeonly uniform highp uimage2D srDriOut;
// Per-site cross-channel luma at the crop's raw grid (.r = white-balanced
// luma, .g = the site's own raw value for the clip weight), rebuilt for this
// frame by merge/srluma. This is what the drizzle fuses: the demosaic's
// square-band reconstruction, which the per-channel lattices cannot carry.
uniform highp sampler2D srLumaTex;
// This frame's aligned residual (mergeAlign writes the aligned alter, so the
// gate forms alter - base) and the running base it is measured against.
uniform highp sampler2D diffPacked;
uniform highp sampler2D basePacked;
uniform highp sampler2D alignmentTexture;
// Per-cell sub-pixel motion correction (Lucas-Kanade against the running
// base, packed texels) and its strength.
uniform highp sampler2D srRefMap;
uniform float srRefine;
// Per-frame atlas tile offset, alignment grid size, rawSize/2: identical
// values and convention to the mergeAlign call for this frame.
uniform ivec2 srShift;
uniform ivec2 srAlignSize;
uniform ivec2 srRawHalf;
// Packing shift (cfaShift): packed coord = (cropRaw + cfaShift) / 2.
uniform ivec2 srCfa;
// Full-frame raw px per output px (uniform factor in practice).
uniform vec2 srFullPerOut;
// Crop origin in full-frame raw px ((0,0) when uncropped).
uniform vec2 srOrigin;
// Per-frame exposure normalize (same value mergeAlign uses for this frame).
uniform float srExpose;
// 1 = base frame: motion 0, weight 1 (its atlas slot is meaningless;
// mergeAlign never runs for it, so the atlas must not be sampled).
uniform float srZeroMotion;
// Bound on motion magnitude in packed px: real handheld motion passes,
// garbage vectors cannot mirror the frame no matter what the atlas contains.
uniform float srMotionMax;
// 1 on the very first (base) draw: accumulator starts uninitialized.
uniform int srFirst;
// Synthesized sub-pixel jitter (raw px) and this frame's index. A burst with
// no camera motion samples one lattice phase in every frame, so the sensor's
// aliasing never cancels and the fused field is a single interpolation; a
// small per-pixel, per-frame offset (Bayer-16 dither + the R2 golden-ratio
// sequence, as in the temporal-wavelet upsamplers) restores the diversity.
uniform float srJitter;
uniform int srFrame;
// Fusion trust: minimum weight for a frame that disagrees coherently with the
// running estimate, and the low-frequency residual (normalized units) at
// which the weight halves; the band is widened by four sigma of the
// pre-inflation noise model so the merge denoise setting cannot modulate it.
uniform float srTrustFloor;
uniform float srTrustBand;
uniform float srNoiseS0;
uniform float srNoiseO0;
// Weight attenuation for samples at the raw ceiling: burst frames clip at
// different levels and a clipped sample carries no highlight detail.
uniform float srClipAtten;
#define SR_TILE 2
#define SR_TILE_AL 16

// Catmull-Rom-ish hardware bicubic (four linear taps), the same kernel as
// utils/import_interpolation's textureBicubicHardware, inlined because the
// merge compute shaders carry no imports. Bilinear's response at the sensor
// band edge is ~0.4, and the resolve replaces the output luma with this
// field, so bilinear alone made 2x softer than the native demosaic.
vec4 srCubicW(float x) {
    float x2 = x * x;
    float x3 = x2 * x;
    return vec4(-x3 + 3.0 * x2 - 3.0 * x + 1.0,
                 3.0 * x3 - 6.0 * x2 + 4.0,
                -3.0 * x3 + 3.0 * x2 + 3.0 * x + 1.0,
                 x3) / 6.0;
}

vec4 srBicubic(sampler2D s, vec2 uv) {
    vec2 texSize = vec2(textureSize(s, 0));
    vec2 tc = uv * texSize - 0.5;
    vec2 f = fract(tc);
    tc -= f;
    vec4 xc = srCubicW(f.x);
    vec4 yc = srCubicW(f.y);
    vec4 c = tc.xxyy + vec2(-0.5, 1.5).xyxy;
    vec4 sz = vec4(xc.xz + xc.yw, yc.xz + yc.yw);
    vec4 off = (c + vec4(xc.yw, yc.yw) / sz) / texSize.xxyy;
    vec4 s0 = texture(s, off.xz);
    vec4 s1 = texture(s, off.yz);
    vec4 s2 = texture(s, off.xw);
    vec4 s3 = texture(s, off.yw);
    float sx = sz.x / (sz.x + sz.y);
    float sy = sz.z / (sz.z + sz.w);
    vec4 res = mix(mix(s3, s2, sx), mix(s1, s0, sx), sy);
    vec4 bil = texture(s, uv);
    // High-contrast blend: Catmull-Rom's negative lobes overshoot at strong
    // transitions, and the resolve injects this band straight into the output
    // (bench: ~13 levels of dark undershoot around a bright pot that the
    // aniso's edge-preserving reconstruction does not have). A hard clamp
    // bounds per output pixel and generates banding, so instead blend toward
    // the bilinear - a convex combination of the samples, which cannot
    // overshoot - weighted smoothly by the taps' local range. Texture keeps
    // the bicubic; only strong transitions get the non-ringing form.
    vec4 lo = min(min(s0, s1), min(s2, s3));
    vec4 hi = max(max(s0, s1), max(s2, s3));
    vec4 e = smoothstep(vec4(0.10), vec4(0.30), hi - lo);
    return mix(bil, res, vec4(1.0) - e);
}

float srBayer2(vec2 a) {
    a = floor(a);
    return fract(dot(a, vec2(0.5, a.y * 0.75)));
}
#define SR_BAYER4(a) (srBayer2(0.5 * (a)) * 0.25 + srBayer2(a))
#define SR_BAYER8(a) (SR_BAYER4(0.5 * (a)) * 0.25 + srBayer2(a))
#define SR_BAYER16(a) (SR_BAYER8(0.5 * (a)) * 0.25 + srBayer2(a))
#define SR_PHI2 1.324717957244746

// Decoded motion (packed px) of one alignment cell, atlas-offset by srShift.
vec2 srCellMotion(ivec2 cell) {
    ivec2 am = textureSize(alignmentTexture, 0) - ivec2(1);
    vec4 a = texelFetch(alignmentTexture, clamp(cell + srShift, ivec2(0), am), 0);
    return a.xy * vec2(srRawHalf) + a.zw;
}

void main() {
    ivec2 o = ivec2(gl_GlobalInvocationID.xy);
    ivec2 outSize = imageSize(srDriOut);
    if (o.x >= outSize.x || o.y >= outSize.y) return;
    vec2 acc = srFirst != 0 ? vec2(0.0) : unpackHalf2x16(imageLoad(srDriIn, o).x);
    // Output pixel center in full-frame raw span coords, crop-relative.
    vec2 full = (vec2(o) + vec2(0.5)) * srFullPerOut;
    vec2 rel = full - srOrigin;
    // Continuous packed (texel index) coordinate of this site: channel c of
    // texel (i,j) holds raw site 2*(i,j) - cfaShift + (c&1, c>>1), and raw
    // span -> texel index drops the half-texel sample center.
    vec2 p = (rel + vec2(srCfa)) * 0.5 - vec2(0.25);
    // Local motion: bilinear interpolation of the atlas over the 2x2
    // neighbouring alignment cells (nearest-cell sampling left a
    // piecewise-constant field whose steps read as blocky local
    // misregistration). UNFLOORED: merge floors for denoise, the drizzle
    // keeps the sub-pixel residuals that super-resolution depends on.
    vec2 m = vec2(0.0);
    vec4 refv = vec4(0.0);
    if (srZeroMotion < 0.5) {
        vec2 A = p / (float(SR_TILE_AL) / float(SR_TILE));
        ivec2 c0 = ivec2(floor(A));
        vec2 f = A - vec2(c0);
        ivec2 cmax = srAlignSize - ivec2(1);
        vec2 m00 = srCellMotion(clamp(c0, ivec2(0), cmax));
        vec2 m10 = srCellMotion(clamp(c0 + ivec2(1, 0), ivec2(0), cmax));
        vec2 m01 = srCellMotion(clamp(c0 + ivec2(0, 1), ivec2(0), cmax));
        vec2 m11 = srCellMotion(clamp(c0 + ivec2(1, 1), ivec2(0), cmax));
        m = mix(mix(m00, m10, f.x), mix(m01, m11, f.x), f.y);
        m = clamp(m, vec2(-srMotionMax), vec2(srMotionMax));
        // Sub-pixel alignment refinement (bilinear over the same cell grid):
        // the fusion's sampling positions are only as good as the alignment.
        refv = texture(srRefMap, clamp(
                (A + vec2(0.5)) / vec2(srAlignSize), vec2(0.0), vec2(1.0)));
        // Apply the correction only where the frame actually moved: on a
        // static burst the LK residual is noise, its fit is a random
        // translation, and applying it warps the sampling lattice randomly
        // (bench: it dropped the SR fine band's correlation with the scene to
        // ~0 while the atlas motion was ~0). The atlas motion gates it: full
        // correction from half a raw pixel of motion up.
        m += srRefine * refv.xy * clamp(2.0 * length(m), 0.0, 1.0);
    }
    // Synthesized jitter: none where the frame already moved (the real dither
    // is there), full where it is static relative to the base. The offset is
    // FRAME-GLOBAL, not per pixel: a per-pixel pattern (the earlier Bayer16
    // form) imprints its own texture on the fused field - benched against a
    // tripod raw, it dropped the SR fine band's correlation with the scene to
    // 0.36 where the native render sits at 0.75. A constant per-frame offset
    // still shifts the sampling lattice (so the aliased components cancel in
    // the average) without modulating the result.
    vec2 jit = vec2(0.0);
    if (srJitter > 0.0) {
        vec2 g = fract(vec2(float(srFrame)) / vec2(SR_PHI2 * SR_PHI2, SR_PHI2)) - vec2(0.5);
        jit = g * (srJitter * clamp(1.0 - 2.0 * length(2.0 * m), 0.0, 1.0));
    }
    // Bicubic gather of the per-site luma: a real interpolator (no
    // site-lattice sample-and-hold, which is itself a grid) that keeps the
    // sensor band's upper octave, which bilinear loses. The burst's sub-pixel
    // diversity still fills the finer output grid.
    vec2 luv = clamp((rel + 2.0 * m + jit) / vec2(textureSize(srLumaTex, 0)),
            vec2(0.0), vec2(1.0));
    // Sharpened bicubic: the hardware bicubic plus half of its difference from
    // the hardware bilinear (one extra fetch). Catmull-Rom's response at the
    // sensor band edge is ~0.7 and caps what the fused field can carry; the
    // difference term restores most of it (host sim at 3% per-site noise:
    // texture error 0.0315 -> 0.0263 dithered, 0.0341 -> 0.0319 static) with
    // the edge overshoot staying at or below the ideal's, so no halos.
    // Plain bicubic: the +0.5*(bicubic - bilinear) unsharp that used to sit
    // here was benched against tripod raws and correlated WORSE with the
    // scene than even the bilinear (0.05 vs 0.30) - it imprints ringing
    // rather than detail. The post's edge-gated acutance does the sharpening.
    vec4 l = srBicubic(srLumaTex, luv);
    // Clamp runaway exposure normalization (insane layerMpy metadata would
    // otherwise blow finite samples to Inf downstream of every guard).
    float ex = min(srExpose, 32.0);
    float v = l.r * ex;
    float w = 1.0;
    if (srZeroMotion < 0.5) {
        // Fusion trust: judge only *coherent* disagreement - a 3x3
        // packed-window signed mean of the true residual (aligned alter minus
        // the running base) cancels the dipoles sub-pixel sampling leaves at
        // edges, while a misregistered frame or a moving object still reads
        // as a large coherent mean. The mean magnitude keeps gross outliers
        // (occlusions, specular poison) from voting.
        ivec2 dmax = textureSize(diffPacked, 0) - ivec2(1);
        ivec2 ip = ivec2(floor(p + vec2(0.5)));
        vec3 rSum = vec3(0.0);
        vec3 rAbs = vec3(0.0);
        for (int j = -1; j <= 1; j++) {
            for (int i = -1; i <= 1; i++) {
                ivec2 tp = clamp(ip + ivec2(i, j), ivec2(0), dmax);
                vec3 r = texelFetch(diffPacked, tp, 0).rgb - texelFetch(basePacked, tp, 0).rgb;
                rSum += r;
                rAbs += abs(r);
            }
        }
        vec3 rMean = rSum * (1.0 / 9.0);
        vec3 rMag = rAbs * (1.0 / 9.0);
        float lum = max(dot(texelFetch(diffPacked, clamp(ip, ivec2(0), dmax), 0).rgb,
                vec3(1.0 / 3.0)), 1e-4);
        float var0 = max(srNoiseS0 * lum + srNoiseO0, 1e-12);
        float band = max(srTrustBand, 4.0 * sqrt(var0));
        // With the refinement active, trust the residual left *after* the
        // per-cell translation correction (refv.z): a frame that is merely
        // misregistered is fixed by the correction and must not be rejected,
        // while deformation and moving subjects still read as large.
        float lf = srRefine > 0.5 ? refv.z : dot(abs(rMean), vec3(1.0 / 3.0));
        float mag = dot(rMag, vec3(1.0 / 3.0));
        w = max(1.0 / (1.0 + (lf / band) * (lf / band)), srTrustFloor)
                / (1.0 + pow(mag / (16.0 * band), 4.0));
    }
    // Clipping: a sample at the raw ceiling carries no highlight detail, and
    // burst frames clip at different levels, so a clipped frame drags
    // highlight edges around. A uniformly clipped site is unaffected.
    w *= 1.0 - srClipAtten * smoothstep(0.9, 1.0, clamp(l.g, 0.0, 1.0));
    imageStore(srDriOut, o, uvec4(packHalf2x16(acc + vec2(v * w, w))));
}
