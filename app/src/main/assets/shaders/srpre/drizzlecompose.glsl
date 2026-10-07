precision highp float;
precision highp sampler2D;
#import interpolation
// Demosaiced crop (raw grid, post-ABLC RGB): the chroma source.
uniform sampler2D InputBuffer;
// Fused drizzle at the OUTPUT grid (R32UI packed halves: .r luma in the
// pre-ABLC packed domain, .g effective accumulated weight from
// merge/srlattice1).
uniform highp usampler2D FusedLuma;
// ABLC black levels: FusedLuma arrives pre-ABLC, InputBuffer post.
uniform vec3 srBlack;
// Raw px per output px per axis (the drizzle's expansion).
uniform vec2 srPerOut;
// Expected accumulated weight per drizzle texel (frames / expansion^2, the
// same coverage reference merge/srrecover uses): the drizzle luma is a local
// average over the deposited samples, so cells with far fewer samples than
// this carry sampling-lattice noise rather than detail and fall back toward
// the reference.
uniform float srCovRef;
out vec4 Output;

// Compose the output-grid RGB the reconstruction consumes: luma from the
// drizzle (which represents above-raw-Nyquist content instead of folding it),
// chroma from the demosaiced crop (smooth, and the drizzle has none). The
// post then runs its aniso at zoom 1; splitChroma keeps this chroma.
void main() {
    ivec2 o = ivec2(gl_FragCoord.xy);
    ivec2 fmax = textureSize(FusedLuma, 0) - ivec2(1);
    ivec2 imax = textureSize(InputBuffer, 0) - ivec2(1);
    vec2 fw = unpackHalf2x16(texelFetch(FusedLuma, clamp(o, ivec2(0), fmax), 0).x);
    float yd = fw.r;
    // Coverage fallback: thin cells (holes, single-sample lattice noise)
    // blend back to the reference instead of carrying the sampling grid.
    // At the expected coverage the drizzle is used whole; the reference
    // scale keeps the fallback meaningful at any expansion or frame count.
    float alpha = clamp(fw.g / max(srCovRef, 1e-3), 0.0, 1.0);
    vec2 c = (vec2(o) + vec2(0.5)) * srPerOut;
    vec2 uv = c / vec2(textureSize(InputBuffer, 0));
    // Bicubic chroma: a bilinear crop upsample would leave a block grid at
    // chromatic edges (which the aniso's splitChroma then carries).
    vec3 ref = textureBicubicHardware(InputBuffer, uv).rgb;
    float bLuma = dot(srBlack, vec3(0.2126, 0.7152, 0.0722));
    float denom = max(1.0 - bLuma, 1e-4);
    float y = max((yd - bLuma) / denom, 0.0);
    float yc = dot(ref, vec3(0.2126, 0.7152, 0.0722));
    // Holes (no drizzle samples): keep the reference's luma there.
    if (!(y > 1e-6)) y = yc;
    y = mix(yc, y, alpha);
    vec3 co = ref - vec3(yc);
    Output = vec4(clamp(y + co, vec3(0.0), vec3(8.0)), 1.0);
}
