package com.particlesdevs.photoncamera.processing.opengl.postpipeline;

import org.junit.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.charset.StandardCharsets;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

/** Regression guards for shader contracts that are not exercised by JVM-only rendering tests. */
public class LtmShaderRegressionTest {
    private static String asset(String path) throws IOException {
        Path modulePath = Path.of("src/main/assets", path);
        if (Files.exists(modulePath)) {
            return read(modulePath);
        }
        return read(Path.of("app", modulePath.toString()));
    }

    private static String source(String path) throws IOException {
        Path modulePath = Path.of("src/main/java", path);
        if (Files.exists(modulePath)) {
            return read(modulePath);
        }
        return read(Path.of("app", modulePath.toString()));
    }

    private static String read(Path path) throws IOException {
        return new String(Files.readAllBytes(path), StandardCharsets.UTF_8);
    }

    @Test
    public void bayer2ExposureUsesWhitePointAndPreservesHighlightLimit() throws IOException {
        String shader = asset("shaders/ltm/exposebayer2.glsl");

        assertTrue(shader.contains("inp /= max(neutral.rggb, vec4(1e-6))"));
        assertTrue(shader.contains("clamp(br,0.0,highLim)"));
    }

    @Test
    public void fusionUsesReciprocalUpscaleIn() throws IOException {
        String java2 = source(
                "com/particlesdevs/photoncamera/processing/opengl/postpipeline/ExposureFusionBayer2.java");
        String java3 = source(
                "com/particlesdevs/photoncamera/processing/opengl/postpipeline/ExposureFusionBayer3.java");
        String shader3 = asset("shaders/ltm/fusionbayer3.glsl");
        String shader2 = asset("shaders/ltm/fusionbayer2.glsl");

        // The shader takes the CPU-precomputed reciprocal of the target size so
        // the full-resolution sampling stays a multiply, avoiding the slower and
        // less precise per-fragment GPU division.
        for (String shader : new String[]{shader3, shader2}) {
            assertTrue(shader.contains("uniform vec2 upscaleIn"));
            assertTrue(shader.contains("vec2(gl_FragCoord.xy) * upscaleIn"));
            assertFalse(shader.contains("textureBicubicHardware(upsampled"));
            assertFalse(shader.contains("ivec2 targetSize"));
            assertFalse(shader.contains("/ vec2(targetSize)"));
        }
        for (String java : new String[]{java2, java3}) {
            assertTrue(java.contains("setVar(\"upscaleIn\",1.0f/binnedFuse.mSize.x,1.0f/binnedFuse.mSize.y)"));
            assertTrue(java.contains("setVar(\"upscaleIn\",1.0f/normalExpo.sizes[i].x, 1.0f/normalExpo.sizes[i].y)"));
            // The dehazing detail boost stays on the intermediate levels (not the
            // coarsest, which amplified low-frequency detail in flat shadows and
            // caused banding) and the finest level is not amplified at all: the
            // finest Laplacian carries the pyramid's even/odd phase pattern, and
            // amplifying it paints a grid on fine detail.
            assertTrue(java.contains("float blendMpy = (i == 0) ? 1.0f"));
            assertTrue(java.contains("1.0f + dehazing - dehazing * ((float) i) / (normalExpo.laplace.length - 1.f)"));
        }
    }

    @Test
    public void ultraHdrAlphaCarriesFusedLuminance() throws IOException {
        String shader = asset("shaders/initial.glsl");
        String gainMap = asset("shaders/ultrahdr/gainmap.glsl");
        String gainMapGenerator = source(
                "com/particlesdevs/photoncamera/processing/opengl/postpipeline/GainMapGenerator.java");

        // The alpha must carry the luminance AFTER the LTM boost but BEFORE the
        // SDR clamp so the gain map recovers lifted highlights instead of
        // exposing the clipped base.
        assertTrue(shader.contains("linearLum = min(luminocity(pRGBBoosted), 16.0)"));
        assertTrue(shader.contains("pRGB = clamp(reinhard_extended(pRGBBoosted, max(1.0, tonemapGain)), 0.0, 1.0)"));
        assertFalse(shader.contains("pRGB = clamp(pRGBBoosted, 0.0, 1.0)"));
        // Shadow side must fade gain smoothly instead of a binary deadband jump
        // and must not amplify the alpha noise floor in near-black pixels.
        assertTrue(gainMap.contains("smoothstep(0.0, GAIN_DEADBAND, L)"));
        assertTrue(gainMap.contains("smoothstep(0.0, 0.02, sdrLin)"));
        assertTrue(gainMap.contains("encoded / 255.0"));
        assertTrue(gainMap.contains("float hashDither(ivec2 p)"));
        assertTrue(gainMapGenerator.contains("setVar(\"CROP_Y\", inSize.y - cropH)"));
    }

    @Test
    public void fusionMapGuidedApplicationIsEdgeAware() throws IOException {
        String shader = asset("shaders/initial.glsl");
        String fusionMap = asset("shaders/ltm/fusionmap.glsl");

        // The map is notch-filtered edge-aware in fusionmap.glsl and must carry
        // the bounded gain ratio (not (a,b) fit coefficients), clamped on the
        // way out so the full-res application never sees an out-of-range gain.
        assertTrue(fusionMap.contains("float lowresVal  = clamp("));
        assertTrue(fusionMap.contains("clamp(mix(1.0, ratio, ratioConfidence), 0.0, FUSIONCAP)"));
        assertFalse(fusionMap.contains(", 0.0, 8.0)"));
        assertTrue(fusionMap.contains("float ratioConfidence = smoothstep(0.001, 0.01, baseValue)"));
        assertTrue(fusionMap.contains("mix(1.0, ratio, ratioConfidence)"));
        assertTrue(fusionMap.contains("result = vec2(lowresVal / FUSIONGAIN, 0.0)"));
        assertFalse(fusionMap.contains("result = vec2(a,b)"));
        assertFalse(fusionMap.contains("result *= clamp(factor"));
        // The map-side smoothing must kill the pyramid's even/odd phase pattern
        // with a kernel that has spectral nulls at f=1/2, 1/4, 1/8 (a Gaussian
        // has none), while the range gate on the clean base keeps the gain edge
        // from being bled across (the halo).
        assertTrue(fusionMap.contains("float wi = (abs(i) == 4) ? 1.0 : 2.0"));
        assertTrue(fusionMap.contains("float spatialWeight = wi * ((abs(j) == 4) ? 1.0 : 2.0)"));
        assertTrue(fusionMap.contains("float rangeWeight = exp("));
        assertTrue(fusionMap.contains("ivec2 safePos = clamp(xy"));
        // getGain() must sample the FusionMap once per 2x2 output block, at the
        // block center, so all four pixels in the block share the same gain
        // prior. The UV must be normalized against the full-res input size so
        // half-res texel k covers exactly the output block [2k, 2k+2): texel k
        // sits at block center (2k+1, 2k+1), i.e. (k+0.5)/halfRes, so the UV is
        // blockCenter/inputSize with NO +0.5 texel offset.
        assertTrue(shader.contains("float getGain(ivec2 centerPos)"));
        assertTrue(shader.contains("ivec2 blockBase = (centerPos / 2) * 2"));
        assertTrue(shader.contains("ivec2 blockCenter = blockBase + ivec2(1, 1)"));
        assertTrue(shader.contains("vec2 uv = vec2(blockCenter) / vec2(inputSize)"));
        assertFalse(shader.contains("vec2 uv = (vec2(blockCenter) + vec2(0.5)) / vec2(inputSize)"));
        assertTrue(shader.contains("return texture(FusionMap, uv).r * FUSIONGAIN * mapNorm;"));
        // The guided-filter window reads the map at each tap's own block center:
        // block-aligned, so the moments carry no even/odd phase. It must not
        // re-derive a non-block-aligned per-pixel sample, or windowed moments
        // would alternate gain phase between even/odd pixels and paint a 2x2
        // grid on detail.
        assertTrue(shader.contains("float gain = getGain(xy + ivec2(i, j))"));
        assertFalse(shader.contains("float gain = getGain(xy, offset)"));
        assertFalse(shader.contains("ivec2 mapOffset = ivec2(sign(offset))"));
        assertFalse(shader.contains("s += w * texture(FusionMap, uv + vec2(i, j) / vec2(inputSize)).r"));
        assertFalse(shader.contains("float w = (i == 0 ? 2.0 : 1.0) * (j == 0 ? 2.0 : 1.0)"));
        assertFalse(shader.contains("return (s / 16.0) * FUSIONGAIN;"));
        assertFalse(shader.contains("/ vec2(fusionSize))"));
        assertFalse(shader.contains("ivec2 fusionSize = textureSize(FusionMap, 0)"));
        assertTrue(shader.contains("float rangeWeight = exp("));
        assertTrue(shader.contains("const float lumaSigma = LTMLUMASIGMA"));
        assertTrue(shader.contains("#define LTMLUMASIGMA 0.16"));
        assertTrue(shader.contains("#define LTMREFIT 0.45"));
        // The affine refit is clamped to the gain envelope actually present in
        // the window: an unclamped a*luma+b extrapolation is what over/under-
        // shoots (halos) at strong edges. The refit deviation is damped toward
        // the window's smoothed mean gain and the range gate is softened so the
        // gain field fades gradually across tonal edges instead of stepping
        // (the shadow plateau boundary stops printing a bright rim), while a
        // sub-unity refit strength keeps the model insensitive to the center
        // pixel's luma (no per-pixel grid on gradients).
        assertTrue(shader.contains("float guidedGain = meanY + (a * (centerLightness - meanX)) * LTMREFIT"));
        assertFalse(shader.contains("float guidedGain = a * centerLightness + b"));
        assertTrue(shader.contains("tonemapGain = clamp(tonemapGain, localMinGain, localMaxGain)"));
        // The halo/shadow gates must run on spatial-only window statistics, not
        // on the raw per-pixel luma or a range-weighted mean: range weighting
        // locks onto the center pixel's coherent 1px structure and the smoothstep
        // slope amplifies that ripple into a per-pixel gain grid on gradients and
        // diagonal edges. The bright tail is a smooth mean+std proxy, never the
        // raw window max, which jumps by a 1px feature's amplitude as it enters
        // and leaves the window.
        assertTrue(shader.contains("gateLumaSum += lightness * spatialWeight"));
        assertTrue(shader.contains("float gateLuma = gateLumaSum / gateLumaWs"));
        assertTrue(shader.contains("float gateStd = sqrt(max(gateVar, 0.0))"));
        assertTrue(shader.contains("float brightTail = gateLuma + 1.5 * gateStd"));
        assertTrue(shader.contains("float haloGate = smoothstep(brightTail * 0.85, max(brightTail, 1e-4), gateLuma)"));
        assertTrue(shader.contains("float shadowFloor = smoothstep(0.15, 0.45, gateLuma)"));
        assertTrue(shader.contains("float highlightMask = haloGate * shadowFloor"));
        assertFalse(shader.contains("float brightTail = max(localMaxLightness, meanX)"));
        assertFalse(shader.contains("smoothstep(brightTail * 0.85, max(brightTail, 1e-4), meanX)"));
        assertFalse(shader.contains("float localMaxLightness = centerLightness"));
        assertFalse(shader.contains("localMaxLightness = max(localMaxLightness, lightness)"));
        // The FUSIONGAIN shadow cap must stay a monotonic min(): a "smooth"
        // cap dips below FUSIONGAIN on the highlight side to rejoin the
        // uncapped gain, inverting and posterizing highlight-adjacent pixels.
        // The wide shadowFloor band spreads the release gradually (monotonic
        // in gain), which softens the shadow-edge rim without that artifact.
        assertTrue(shader.contains("mix(min(tonemapGain, FUSIONGAIN), tonemapGain, highlightMask)"));
        assertFalse(shader.contains("CAP_SOFTNESS"));
        assertFalse(shader.contains("cappedGain"));
        assertTrue(shader.contains("tonemapGain = clamp(tonemapGain, 0.25, FUSIONCAP)"));
        assertFalse(shader.contains("tonemapGain = 1.0;"));
        assertFalse(shader.contains("FUSIONCURVE"));
        assertFalse(shader.contains("baseDepth"));
        assertFalse(shader.contains("protectedDepth"));
        assertFalse(shader.contains("FUSIONLO"));
        assertFalse(shader.contains("tonemapGain = clamp(tonemapGain, 0.25, 8.0)"));
    }

    @Test
    public void fusionMapMeanIsGuardedFromGlobalDarkening() throws IOException {
        String shader = asset("shaders/initial.glsl");
        String fusion2 = source(
                "com/particlesdevs/photoncamera/processing/opengl/postpipeline/ExposureFusionBayer2.java");
        String pipeline = source(
                "com/particlesdevs/photoncamera/processing/opengl/postpipeline/PostPipeline.java");
        String initial = source(
                "com/particlesdevs/photoncamera/processing/opengl/postpipeline/Initial.java");

        // The fused/base gain map can average below 1 in bright scenes, which
        // multiplies a sub-unity gain into every pixel and darkens the whole
        // frame. The guard must rescale the map's mean back up to 1 (never
        // down) so the shadow lift in dark scenes is preserved, and it must be
        // applied on top of the FUSIONGAIN recovery in getGain.
        assertTrue(shader.contains("uniform float mapNorm;"));
        assertTrue(shader.contains("return texture(FusionMap, uv).r * FUSIONGAIN * mapNorm;"));
        // The guard is a CPU uniform: histogram the map's .r on the GPU, read
        // back only the bins, and derive a uniform scale.
        assertTrue(fusion2.contains("float mapNormalization(GLTexture map)"));
        assertTrue(fusion2.contains("glUtils.convertVec4(map, \"in1.r\")"));
        assertTrue(fusion2.contains("histogram.Compute(vectored)"));
        assertTrue(fusion2.contains("meanStored * ((PostPipeline) basePipeline).fusionGain"));
        assertTrue(fusion2.contains("mapNormalization(((PostPipeline)basePipeline).FusionMap)"));
        // The scale only ever lifts the map toward a mean gain of 1.
        assertTrue(fusion2.contains("if (meanGain >= 1.0f) return 1.0f;"));
        assertFalse(fusion2.contains("ltmMeanTarget"));
        // The pipeline field must be reset per capture so a stale value from a
        // previous frame (or a frame where fusion is disabled) cannot leak in,
        // and it must be bound by the Initial shader before drawing.
        assertTrue(pipeline.contains("float mapNorm = 1.0f"));
        assertTrue(pipeline.contains("mapNorm = 1.0f;"));
        assertTrue(initial.contains("setVar(\"mapNorm\", ((PostPipeline) basePipeline).mapNorm)"));
    }

    @Test
    public void fusionLaplacianClampsBorderFetches() throws IOException {
        String initial = asset("shaders/initial.glsl");
        String bayer2 = asset("shaders/ltm/fusionbayer2.glsl");
        String bayer3 = asset("shaders/ltm/fusionbayer3.glsl");
        String bayer = asset("shaders/ltm/fusionbayer.glsl");
        String fusionMap = asset("shaders/ltm/fusionmap.glsl");

        assertTrue(bayer2.contains("clamp(xyCenter - ivec2(1, 0)"));
        assertTrue(bayer2.contains("clamp(xyCenter + ivec2(0, 1)"));
        assertTrue(bayer3.contains("clamp(xyCenter + ivec2(i, j)"));
        assertTrue(bayer.contains("clamp(xyCenter + ivec2(0, 1)"));
        assertTrue(fusionMap.contains("ivec2 safePos = clamp(xy"));
        assertTrue(initial.contains("ivec2 clampInputPos(ivec2 pos, ivec2 inputSize)"));
        assertTrue(initial.contains("clampInputPos(xy + ivec2(i, j), inputSize)"));
        assertTrue(initial.contains("vec2 offset = vec2(float(i), float(j))"));
        assertTrue(initial.contains("const float sigma = 2.0"));
    }

    @Test
    public void finestLevelLaplacianIsNotAmplified() throws IOException {
        String java2 = source(
                "com/particlesdevs/photoncamera/processing/opengl/postpipeline/ExposureFusionBayer2.java");
        String shader3 = asset("shaders/ltm/fusionbayer3.glsl");

        // The finest-level Laplacian carries the pyramid's one-texel even/odd
        // phase pattern; it must not be amplified (blendMpy stays 1.0 there) or
        // the map picks up a grid on fine detail.
        assertTrue(java2.contains("float blendMpy = (i == 0) ? 1.0f"));
        assertTrue(shader3.contains("float resultVal = base + detail * blendMpy"));
    }

    @Test
    public void fusionDetailIsRangeClampedToBaseNeighborhood() throws IOException {
        String shader = asset("shaders/ltm/fusionbayer3.glsl");

        // The pyramid reconstruction must clip Laplacian overshoot to the local
        // base range; otherwise a bright source produces bright/dark halo rings.
        assertTrue(shader.contains("vec2 upCoord = vec2(gl_FragCoord.xy) * upscaleIn"));
        assertTrue(shader.contains("vec2 texSize = vec2(textureSize(upsampled, 0))"));
        assertTrue(shader.contains("resultVal = clamp(resultVal, lo, hi)"));
        assertTrue(shader.contains("float lo = min(base, min(min(texture(upsampled, n0).r, texture(upsampled, n1).r)"));
        assertTrue(shader.contains("float hi = max(base, max(max(texture(upsampled, n0).r, texture(upsampled, n1).r)"));
        assertFalse(shader.contains("result = base + detail * blendMpy"));
    }

    @Test
    public void autoExposureAndGainMapHandleExtremeInputs() throws IOException {
        String autoExposure = source(
                "com/particlesdevs/photoncamera/processing/opengl/postpipeline/AutoExposure.java");
        String apply = asset("shaders/autoexposure/apply.glsl");

        assertTrue(autoExposure.contains("cnt <= 0 || !Float.isFinite(sum) || sum <= 0.0f"));
        assertTrue(autoExposure.contains("!Float.isFinite(mpy) || mpy <= 0.0f"));
        assertTrue(autoExposure.contains("!Float.isFinite(normR) || normR <= 0.0f"));
        assertTrue(apply.contains("Output.a = inp.a"));
        assertFalse(apply.contains("Output.a = min(inp.a * max(mpy, 0.0), 16.0)"));
    }

    @Test
    public void pipelineResetsRunStateAndCleansUpFailures() throws IOException {
        String pipeline = source(
                "com/particlesdevs/photoncamera/processing/opengl/postpipeline/PostPipeline.java");
        String gainMap = source(
                "com/particlesdevs/photoncamera/processing/opengl/postpipeline/GainMapGenerator.java");

        assertTrue(pipeline.contains("totalGain = 1.0f"));
        assertTrue(pipeline.contains("finally {"));
        assertTrue(pipeline.contains("GLTexture.closeAll()"));
        assertTrue(gainMap.contains("outBitmap.recycle()"));
    }

    @Test
    public void textureTrackingUsesSlotsAndIsIdempotent() throws IOException {
        String texture = source(
                "com/particlesdevs/photoncamera/processing/opengl/GLTexture.java");
        String fusion2 = source(
                "com/particlesdevs/photoncamera/processing/opengl/postpipeline/ExposureFusionBayer2.java");
        String fusion3 = source(
                "com/particlesdevs/photoncamera/processing/opengl/postpipeline/ExposureFusionBayer3.java");

        assertTrue(texture.contains("private static GLTexture[] owners"));
        assertTrue(texture.contains("private int trackingSlot = -1"));
        assertTrue(texture.contains("if (closed)"));
        assertTrue(fusion2.contains("normalExpo.gauss[i].close()"));
        assertFalse(fusion2.contains("normalExpo.gauss[ind].close();\n        //highExpo.gauss[ind]"));
        assertTrue(fusion3.contains("normalExpo.gauss[i].close()"));
        assertFalse(fusion3.contains("normalExpo.gauss[ind].close();\n        //highExpo.gauss[ind]"));
    }

    @Test
    public void bicubicBorderTapsAreExplicitlyClamped() throws IOException {
        String shader = asset("shaders/utils/import_interpolation.glsl");

        assertTrue(shader.contains("clamp(offset.xz, vec2(0.0), vec2(1.0))"));
        assertTrue(shader.contains("clamp(offset.yw, vec2(0.0), vec2(1.0))"));
    }

    @Test
    public void shadowCurveIsBuiltFromShadowPoints() throws IOException {
        String source = source(
                "com/particlesdevs/photoncamera/processing/opengl/postpipeline/ExposureFusionBayer2.java");

        assertTrue(source.contains("shadowX.add(shadowCurveX[i])"));
        assertTrue(source.contains("shadowY.add(shadowCurveY[i])"));
        assertTrue(source.contains("SplineInterpolator.createMonotoneCubicSpline(shadowX, shadowY)"));
    }
}
