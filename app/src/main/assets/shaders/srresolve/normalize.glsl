precision highp float;
precision highp sampler2D;
// Drizzled accumulation at output size: fused cross-channel luma and the
// accumulated weight packed as two halves into one R32UI word (ES 3.1 image
// load/store only guarantees 1- or 4-channel formats).
uniform highp usampler2D srAccum;
// Demosaiced aniso reconstruction at output size (this node's input): clean
// color and clean luma, from the full mosaic.
uniform sampler2D InputBuffer;
// Tiled rendering origin (output coords of this tile's row 0).
uniform ivec2 u_tileOrigin;
// Weight of the injected high band (0 = pure aniso).
uniform float srBlend;
// Local noise model (post domain) and the detail scale. The injected band
// carries the fused luma's noise: shrink it against the fused noise floor.
uniform float srNoiseS;
uniform float srNoiseO;
uniform float srDetail;
// How much of the aniso's edge-gated acutance term (handed over in the
// reference's alpha) is added to the resolved output. This is the legitimate
// half of the Disabled render's edge sharpening; the KernelNet's band - and
// the quantization mesh in it - is what the replacement drops.
uniform float srAcutance;
// Per-channel ABLC black levels applied to the post image (0 when ABLC is
// off). The fused luma arrives in the pre-ABLC packed domain.
uniform vec3 srBlack;
out vec4 Output;

void main() {
    ivec2 o = ivec2(gl_FragCoord.xy) + u_tileOrigin;
    ivec2 sz = textureSize(srAccum, 0);
    ivec2 cmax = sz - ivec2(1);
    vec2 c = unpackHalf2x16(texelFetch(srAccum, clamp(o, ivec2(0), cmax), 0).x);
    float yFused = c.x / max(c.y, 1e-3);
    vec4 refT = texelFetch(InputBuffer, clamp(o, ivec2(0), textureSize(InputBuffer, 0) - ivec2(1)), 0);
    vec3 ref = refT.rgb;
    float yRef = dot(ref, vec3(0.2126, 0.7152, 0.0722));
    // The GPU handoff skips the merge's CPU sanity scan, so sanitize here:
    // NaN fails every comparison and would otherwise ride the injection.
    if (!(yFused <= 65504.0)) yFused = yRef;
    // The reference is the post-ABLC image; the fused luma is in the pre-ABLC
    // packed domain. Push the fused luma through the same transform (the
    // levels are per-channel, so the luma uses their Rec.709 mix), or the
    // levels disagree.
    float bLuma = dot(srBlack, vec3(0.2126, 0.7152, 0.0722));
    float denom = max(1.0 - bLuma, 1e-4);
    float yFusedPost = max((yFused - bLuma) / denom, 0.0);
    // The injected quantity is the FULL fused-luma difference, and the
    // shrinkage does the selection. Two device-bench findings drove this:
    // - The 5x5 box high-pass it used to inject had an asymmetric response at
    //   a strong edge, so the "band" carried an undershoot by construction
    //   and left a dark fringe along bright edges (the highlight halo).
    // - Splitting the band also meant the aniso's low-frequency
    //   reconstruction errors (blotching on bright walls) were never
    //   replaced, since only its upper band was being substituted.
    // The fused luma is a multi-frame average of the actual raw, so its low
    // band is clean scene level; the Wiener shrinkage below suppresses the
    // small differences (the fused's noise, the KernelNet's errors) while the
    // large ones - real level and detail - pass. No box; the only split is
    // the fallback's smooth low-pass below.
    float hp = yFusedPost - yRef;
    // Wiener (soft) shrinkage against the local noise floor: one-frame sigma
    // from the noise model over the square root of the accumulated weight
    // (the fused luma is a weighted average of that many samples). The model
    // is calibrated per raw site, while the fused luma is additionally a
    // spatial average - the sharpened bicubic gather's noise gain is about
    // 0.5 - so halve it. Without this the threshold sits ~2x above the true
    // noise and the shrinkage eats faint texture (small text strokes) while
    // strong edges pass: exactly the wrong selection.
    float sigma = 0.5 * sqrt(max(srNoiseS * yRef + srNoiseO, 1e-12) / max(c.y, 1.0));
    float t = srDetail * sigma / max(abs(hp), 1e-9);
    float m = max(0.0, 1.0 - t * t);
    // Mesh-proof fallback. Where the shrinkage suppresses (m < 1: flats and
    // fine texture), do NOT fall back to the aniso's raw luma: the aniso
    // carries the KernelNet map's texel-scale mottle (the map is emitted at
    // half crop resolution, so its cells are 4 output px at 2x) and the
    // tripod bench shows it as a faint 2-4 px pattern on flats in the 2x
    // JPEG. The pre-full-difference build was clean there only because it
    // subtracted the reference's band, which is exactly what removed it.
    // Adding back the low-passed suppressed difference replaces the aniso's
    // low+mid band - where the mottle lives - with the fused's clean content,
    // while the top octave (the fused's own noise) still stays the aniso's.
    // Strictly additive where m < 1 and smooth/symmetric, so no edge fringe.
    // The 3x3 Gaussian (sigma ~0.55 output px) passes ~0.7 at the map's cell
    // frequency and ~0.5 at the output Nyquist: the mottle is replaced, the
    // pixel-level grain mostly stays the aniso's.
    float lpF = 0.0;
    float lpR = 0.0;
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            float wgt = (i == 0 ? 0.72 : 0.14) * (j == 0 ? 0.72 : 0.14);
            vec2 s = unpackHalf2x16(texelFetch(srAccum, clamp(o + ivec2(i, j), ivec2(0), cmax), 0).x);
            lpF += wgt * max((s.x / max(s.y, 1e-3) - bLuma) / denom, 0.0);
            vec3 r = texelFetch(InputBuffer,
                    clamp(o + ivec2(i, j), ivec2(0), textureSize(InputBuffer, 0) - ivec2(1)), 0).rgb;
            lpR += wgt * dot(r, vec3(0.2126, 0.7152, 0.0722));
        }
    }
    float dInj = m * hp + (1.0 - m) * (lpF - lpR);
    vec3 rgb = ref + srBlend * dInj + vec3(srAcutance * refT.a);
    Output = vec4(clamp(rgb, vec3(0.0), vec3(8.0)), 1.0);
}
