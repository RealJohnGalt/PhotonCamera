precision highp float;
precision highp sampler2D;
uniform sampler2D InputBuffer;
uniform sampler2D DetailMap;
// Full-frame raw px per output px, per axis (uniform factor in practice).
uniform vec2 srRawPerOut;
// Crop origin in full-frame raw px ((0,0) when uncropped).
uniform vec2 srOrigin;
// Packing shift (cfaShift): packed coord = (cropRaw + cfaShift) / 2.
uniform ivec2 srCfa;
uniform float srStrength;
// Tiled rendering origin (output coords of this tile's row 0). (0,0) on the
// legacy path: identical.
uniform ivec2 u_tileOrigin;
out vec4 Output;

void main() {
    ivec2 o = ivec2(gl_FragCoord.xy) + u_tileOrigin;
    ivec2 inSize = textureSize(InputBuffer, 0);
    vec4 s = texelFetch(InputBuffer, clamp(o, ivec2(0), inSize - ivec2(1)), 0);
    // Crop-relative raw site, then the merge00 packing convention shared
    // with merge2o: packed coord AND channel index both key off (raw +
    // cfaShift). Raw parity alone misassigns quad channels on non-RGGB
    // layouts, landing wrong-sign detail on edges.
    vec2 r = max(vec2(o) * srRawPerOut - srOrigin, vec2(0.0));
    vec2 rp = r + vec2(srCfa);
    vec2 pq = rp * 0.5;
    vec2 ps = vec2(textureSize(DetailMap, 0));
    vec4 d = texture(DetailMap, clamp((pq + vec2(0.5)) / ps, vec2(0.0), vec2(1.0)));
    int ch = int(mod(floor(rp.x), 2.0)) + int(mod(floor(rp.y), 2.0)) * 2;
    float det = ch == 0 ? d.x : (ch == 1 ? d.y : (ch == 2 ? d.z : d.w));
    // Defense in depth: NaN sanitize (ES2 has no isnan; NaN fails every
    // comparison and would render black), and never darken clipped white -
    // destroyed information can only produce artifact there.
    if (!(det <= 0.0 || det >= 0.0)) det = 0.0;
    // Luma-preserving add: ratios intact, luma carries the detail.
    float luma = dot(s.rgb, vec3(0.2126, 0.7152, 0.0722));
    if (luma >= 1.0 && det < 0.0) det = 0.0;
    float nl = max(luma + det * srStrength, 0.0);
    vec3 ratio = s.rgb / max(luma, 1e-4);
    Output = vec4(nl * ratio, s.a);
}
