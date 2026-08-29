
precision highp float;
precision highp usampler2D;
precision mediump sampler2D;
uniform usampler2D InputBuffer;
uniform sampler2D GainMap;
uniform sampler2D Kodak;
uniform ivec2 RawSize;
uniform vec2 RawInvSize;
// Crop offset into the full-frame gain map, normalized [0,1] (0,0 when uncropped).
uniform vec2 GainOffset;
uniform vec2 GainFullInvSize;
uniform vec4 blackLevel;
uniform vec3 whitePoint;
uniform int CfaPattern;
uniform uint whitelevel;
uniform int MinimalInd;
#define BLR (0.0)
#define BLG (0.0)
#define BLB (0.0)
#define QUAD 0
#define RGBLAYOUT 0
#define TESTPATTERN 0
// smooth test pattern selector when TESTPATTERN == 1
// All patterns mix a white component into the hue so that every channel
// reaches the clip point -> saturated color transitions to pure white.
//   0 = gray ramp along x, 0..3 (neutral clipping check)
//   1 = hue ramp along x, luminance 0..3 along y (all hues -> white)
//   2 = constant-hue luminance ramps (hue varies smoothly with y, ramp 0..3 along x)
//   3 = radial hue sweep / cone (hue from angle, luminance from radius, clipped rim)
#define TP 2
// white component mixed into each hue [0..1]; 1/3 is the min that reaches white at l=3
#define WHITE 0.7
#define OFFSET 0,0
#define USEGAIN 1
#import interpolation

vec3 hue2rgb(float h) {
    h = fract(h);
    h *= 6.0;
    if (h < 1.0) return vec3(1.0, h, 0.0);
    if (h < 2.0) return vec3(2.0 - h, 1.0, 0.0);
    if (h < 3.0) return vec3(0.0, 1.0, h - 2.0);
    if (h < 4.0) return vec3(0.0, 4.0 - h, 1.0);
    if (h < 5.0) return vec3(h - 4.0, 0.0, 1.0);
    return vec3(1.0, 0.0, 6.0 - h);
}
#if RGBLAYOUT == 1
out vec3 Output;
#else
out float Output;
#endif


void main() {
    ivec2 xy = ivec2(gl_FragCoord.xy) - ivec2(OFFSET);
    ivec2 fact = (xy)%2;
    xy+=ivec2(CfaPattern%2,CfaPattern/2);
    #if QUAD == 1
        fact = (xy/2)%2;
        xy+=ivec2(CfaPattern%2,CfaPattern/2)*2;
    #endif
    float balance;
    #if USEGAIN == 1
    vec4 gains;
    #ifdef HAS_GAIN_OFFSET
    // Sample the crop's sub-region of the full-frame lens-shading map.
    gains = texture(GainMap, vec2(xy)*vec2(GainFullInvSize) + vec2(GainOffset));
    #else
    gains = texture(GainMap, vec2(xy)*vec2(RawInvSize));
    #endif
    gains.rgb = vec3(gains.r,(gains.g+gains.b)/2.0,gains.a);
    gains.rgb /= dot(gains.rgb,vec3(1.0/3.0));
    #else
    vec3 gains = vec3(1.0);
    #endif
    //gains.rgb = vec3(1.f);
    vec3 level = vec3(blackLevel.r,(blackLevel.g+blackLevel.b)/2.0,blackLevel.a);
    #if RGBLAYOUT == 1
    //Output = vec3(texelFetch(InputBuffer, (xy+ivec2(0,0)), 0).rgb)/(float(whitelevel));
    Output = vec3(texelFetch(InputBuffer, (xy), 0).rgb)/(float(whitelevel));
    Output = gains.rgb*(Output-level.rgb)/(vec3(1.0)-level.rgb);
    #else
    vec3 col = vec3(0.0);
    if(fact.x+fact.y == 1){
            col.g = 1.0;
            balance = whitePoint.g;
            Output = float(texelFetch(InputBuffer, (xy+ivec2(0,0)), 0).x)/(float(whitelevel));
            Output = gains.g*(Output-level.g-BLG)/(1.0-level.g);
        } else {
            if(fact.x == 0){
                col.r = 1.0;
                balance = whitePoint.r;
                Output = float(texelFetch(InputBuffer, (xy), 0).x)/(float(whitelevel));
                Output = gains.r*(Output-level.r-BLR)/(1.0-level.r);
            } else {
                col.b = 1.0;
                balance = whitePoint.b;
                Output = float(texelFetch(InputBuffer, (xy), 0).x)/(float(whitelevel));
                Output = gains.b*(Output-level.b-BLB)/(1.0-level.b);
            }
        }
    // Scene-referred: keep values > 1 so the Ultra HDR gain-map pass can
    // measure recoverable highlight headroom (see ultrahdr/sceneluma.glsl:
    // "scene luminance may legitimately reach/exceed 1.0"). Downstream
    // tone-map/display stages clamp for output; only the negative guard is
    // kept here. (Amaze does not clamp, ABLC previously did — see
    // ABLC/levelcorrection.glsl.)
    Output = max(Output/balance,0.0);
    #endif
    #if TESTPATTERN == 1
        ivec2 diag = ivec2(xy.x+xy.y,xy.x-xy.y);
        float t = float(xy.x)*RawInvSize.x;
        float u = float(xy.y)*RawInvSize.y;
        vec3 col2;
        float l;
        vec3 base;
        #if TP == 0
            // gray smooth ramp along x, 0..3
            l = t*3.0;
            col2 = vec3(l);
        #elif TP == 1
            // hue ramp along x, luminance 0..3 along y
            base = mix(vec3(1.0), hue2rgb(t), 1.0-WHITE);
            l = u*3.0;
            col2 = base*l;
        #elif TP == 2
            // constant-hue luminance ramps: hue smooth with y, ramp 0..3 along x
            base = mix(vec3(1.0), hue2rgb(u), 1.0-WHITE);
            l = t*3.0;
            col2 = base*l;
        #elif TP == 3
            // radial hue sweep: hue from angle, luminance from radius, clipped rim
            vec2 c = (vec2(xy)+vec2(0.5))*vec2(RawInvSize);
            float ang = atan(c.y-0.5, c.x-0.5)/6.28318530718 + 0.5;
            base = mix(vec3(1.0), hue2rgb(ang), 1.0-WHITE);
            l = min(length(c-vec2(0.5))*2.2, 3.0);
            col2 = base*l;
        #endif
        Output = length(col*col2)*balance;
    #endif
}
