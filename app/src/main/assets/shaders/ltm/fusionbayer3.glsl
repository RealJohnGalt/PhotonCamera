precision highp float;
precision highp sampler2D;
uniform sampler2D upsampled;
uniform bool useUpsampled;
uniform float blendMpy;
// Weighting is done using these.
uniform sampler2D normalExpo;

// Blending is done using these.
uniform sampler2D normalExpoDiff;

uniform int level;
// Reciprocal of the output dimensions, computed on the CPU as 1.0/size.
// Multiplying by the precomputed reciprocal keeps full precision and avoids
// the (slower, less precise) per-fragment GPU division.
uniform vec2 upscaleIn;
uniform float gauss;
uniform float target;
//#define TARGET 0.0
//#define GAUSS 0.5
#define MAXLEVEL 4
#define NORM 1.0
#define EPS 1e-6
#define LAPLACEMIN 0.01
#define EXPOMIN 0.01
out float result;
#import gaussian
#import interpolation

vec4 laplace(sampler2D tex, vec4 mid, ivec2 xyCenter) {
        vec4 outp = mid*9.0;
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                ivec2 size = textureSize(tex, 0);
                ivec2 pos = clamp(xyCenter + ivec2(i, j), ivec2(0), size - ivec2(1));
                outp -= texelFetch(tex, pos, 0);
            }
        }
        return abs(outp);
}

void main() {
    ivec2 xyCenter = ivec2(gl_FragCoord.xy);
    // If this is the lowest layer, start with zero.
    vec2 upCoord = vec2(gl_FragCoord.xy) * upscaleIn;
    float base = (useUpsampled)
    ? texture(upsampled, upCoord).r
    : float(0.0);

    // To know that, look at multiple factors.
    vec4 expoVal = texelFetch(normalExpo, xyCenter, 0)*NORM;
    vec4 weights = vec4(1.0, 1.0, 1.0, 1.0);
    // Factor 1: Well-exposedness.
    vec4 normToAvg = (pdf4((expoVal - vec4(target))/gauss));

    weights *= normToAvg + EXPOMIN;

    // Factor 2: Contrast.
    vec4 laplaceVal = laplace(normalExpo, expoVal/NORM, xyCenter)*NORM;

    weights *= laplaceVal + LAPLACEMIN;

    weights *= weights;
    // How are we going to blend these two?
    vec4 expoDiff = texelFetch(normalExpoDiff, xyCenter, 0);
    float detail = (expoDiff.r*weights.r + expoDiff.g*weights.g +
            expoDiff.b*weights.b + expoDiff.a*weights.a) /
            (weights.r + weights.g + weights.b + weights.a);
    float resultVal = base + detail * blendMpy;
    if (useUpsampled) {
        // Keep the reconstruction inside the local base range so the pyramid
        // cannot synthesize bright/dark rings around strong light sources:
        // excursions above the neighborhood max or below the neighborhood min
        // are clipped instead of being added as a halo.
        vec2 texSize = vec2(textureSize(upsampled, 0));
        vec2 d = vec2(1.0) / texSize;
        vec2 n0 = upCoord + vec2(-d.x, 0.0);
        vec2 n1 = upCoord + vec2(d.x, 0.0);
        vec2 n2 = upCoord + vec2(0.0, -d.y);
        vec2 n3 = upCoord + vec2(0.0, d.y);
        float lo = min(base, min(min(texture(upsampled, n0).r, texture(upsampled, n1).r),
                                 min(texture(upsampled, n2).r, texture(upsampled, n3).r)));
        float hi = max(base, max(max(texture(upsampled, n0).r, texture(upsampled, n1).r),
                                 max(texture(upsampled, n2).r, texture(upsampled, n3).r)));
        resultVal = clamp(resultVal, lo, hi);
    }
    result = clamp(resultVal, 0.0, 1.0);
    //if(level == 0){
    //    result = result*result;
    //}
}
