precision highp float;
precision highp sampler2D;

// Passthrough: copies the already-demosaiced linear buffer (Initial's input
// from the first run) into the post-pipeline main texture so the cheap HDR
// pass can start directly at the color/tone stage instead of redoing
// demosaic / denoise / fusion.
uniform sampler2D InputBuffer;

out vec4 Output;

void main() {
    ivec2 xy = ivec2(gl_FragCoord.xy);
    Output = texelFetch(InputBuffer, xy, 0);
}
