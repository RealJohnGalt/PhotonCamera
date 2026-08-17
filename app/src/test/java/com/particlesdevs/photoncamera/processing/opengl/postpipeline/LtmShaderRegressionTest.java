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
            assertFalse(shader.contains("ivec2 targetSize"));
            assertFalse(shader.contains("/ vec2(targetSize)"));
        }
        for (String java : new String[]{java2, java3}) {
            assertTrue(java.contains("setVar(\"upscaleIn\",1.0f/binnedFuse.mSize.x,1.0f/binnedFuse.mSize.y)"));
            assertTrue(java.contains("setVar(\"upscaleIn\",1.0f/normalExpo.sizes[i].x, 1.0f/normalExpo.sizes[i].y)"));
        }
    }

    @Test
    public void ultraHdrAlphaCarriesFusedLuminance() throws IOException {
        String shader = asset("shaders/initial.glsl");
        String gainMap = asset("shaders/ultrahdr/gainmap.glsl");

        // The alpha must carry the luminance AFTER the LTM boost but BEFORE the
        // SDR clamp so the gain map recovers lifted highlights instead of
        // exposing the clipped base.
        assertTrue(shader.contains("linearLum = min(luminocity(pRGBBoosted), 16.0)"));
        assertTrue(shader.contains("pRGB = clamp(pRGBBoosted, 0.0, 1.0)"));
        // Shadow side must fade gain smoothly instead of a binary deadband jump
        // and must not amplify the alpha noise floor in near-black pixels.
        assertTrue(gainMap.contains("smoothstep(0.0, GAIN_DEADBAND, L)"));
        assertTrue(gainMap.contains("smoothstep(0.0, 0.02, sdrLin)"));
    }

    @Test
    public void fusionMapGuidedApplicationIsEdgeAware() throws IOException {
        String shader = asset("shaders/initial.glsl");
        String fusionMap = asset("shaders/ltm/fusionmap.glsl");

        // Full-res application is a guided filter: re-fits the affine model
        // against the local full-res luma instead of bilinearly upsampling
        // coefficients. The map must carry the bounded gain ratio (not (a,b)
        // fit coefficients) so getGain() returns a real gain.
        assertTrue(fusionMap.contains("float lowresVal  = clamp("));
        assertTrue(fusionMap.contains(", 0.0, 8.0)"));
        assertTrue(fusionMap.contains("float ratioConfidence = smoothstep(0.001, 0.01, baseValue)"));
        assertTrue(fusionMap.contains("mix(1.0, ratio, ratioConfidence)"));
        assertTrue(fusionMap.contains("result = vec2(lowresVal / FUSIONGAIN, 0.0)"));
        assertFalse(fusionMap.contains("result = vec2(a,b)"));
        assertTrue(shader.contains("float gain = getGain(offset)"));
        assertTrue(shader.contains("tonemapGain = a * luminocity(sRGB) + b"));
        assertTrue(shader.contains("tonemapGain = clamp(tonemapGain, 0.25, 8.0)"));
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
        assertTrue(initial.contains("clampInputPos(xy + ivec2(i*2+1, j*2+1), inputSize)"));
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
