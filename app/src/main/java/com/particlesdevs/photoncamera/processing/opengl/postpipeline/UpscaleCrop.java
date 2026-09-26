package com.particlesdevs.photoncamera.processing.opengl.postpipeline;

import android.graphics.Bitmap;
import android.graphics.Point;
import android.util.Half;

import com.particlesdevs.photoncamera.processing.ml.KernelNetResult;
import com.particlesdevs.photoncamera.util.Allocator;
import com.particlesdevs.photoncamera.util.Log;
import com.particlesdevs.photoncamera.processing.opengl.GLDrawParams;
import com.particlesdevs.photoncamera.processing.opengl.GLFormat;
import com.particlesdevs.photoncamera.processing.opengl.GLTexture;
import com.particlesdevs.photoncamera.processing.opengl.nodes.Node;
import com.particlesdevs.photoncamera.settings.annotations.Tunable;

import java.nio.ByteBuffer;
import java.nio.ShortBuffer;

import static android.opengl.GLES20.GL_CLAMP_TO_EDGE;
import static android.opengl.GLES20.GL_LINEAR;

/**
 * Expands or shrinks the capture to the requested output size.
 *
 * <p>Output size is the zoom-expanded full size when cropped, otherwise the
 * input size, scaled by the active per-sensor factor ({@code upscaleFactor} /
 * {@code upscaleFactorQb} on {@link com.particlesdevs.photoncamera.processing.render.Parameters}).
 * Applies to cropped and uncropped shots alike; Disabled (default) keeps the
 * legacy behavior (crops expand, uncropped passthrough).</p>
 *
 * <p>When KernelNet params are available (exported by the merge pass, or
 * inferred on the single-frame path by {@link KernelNetPrep}), the resize is
 * reconstructed with a locally anisotropic Gaussian kernel (Wronski et al.,
 * "Procedural Kernel Networks", section 4.3) steered by those params -
 * edge-aligned interpolation at no additional inference cost, plus an
 * optional edge-aligned unsharp term for acutance at extreme zoom. Used for
 * both up and down resizes. Otherwise
 * the pipeline's existing bicubic GPU interpolation path is used.</p>
 */
public final class UpscaleCrop extends Node {

    @Tunable(title = "KernelNet upscale sigma scale", category = "Upscale", description = "Fine trim on the KernelNet map sigmas after the automatic map-to-crop rescaling (1.0 = calibrated)", min = 0.1f, max = 4.0f, step = 0.05f, defaultValue = 0.8f)
    float sigmaScale;

    @Tunable(title = "KernelNet upscale abs min sigma", category = "Upscale", description = "Absolute floor on the reconstruction kernel sigma in crop pixels (numerical guard against tap-weight collapse)", min = 0.05f, max = 1.0f, step = 0.01f, defaultValue = 0.25f)
    float absMinPx;

    @Tunable(title = "KernelNet upscale min sigma (output px)", category = "Upscale", description = "Additional sigma floor measured in output pixels - keeps the reconstruction equally crisp at every zoom factor (lower = sharper at extreme zoom)", min = 0.1f, max = 8.0f, step = 0.05f, defaultValue = 0.9f)
    float outFloorPx;

    @Tunable(title = "KernelNet upscale max sigma", category = "Upscale", description = "Cap on the reconstruction kernel sigma in crop pixels (kept below radius/2 so the window rim never clips the kernel)", min = 0.5f, max = 6.0f, step = 0.1f, defaultValue = 1.0f)
    float sigmaMaxPx;

    @Tunable(title = "KernelNet upscale blend", category = "Upscale", description = "Mix between bicubic (0) and the anisotropic KernelNet reconstruction (1)", min = 0.0f, max = 1.0f, step = 0.05f, defaultValue = 1.0f)
    float anisoStrength;

    @Tunable(title = "KernelNet upscale radius", category = "Upscale", description = "Half-width of the anisotropic reconstruction window in crop pixels (5 = 11x11 taps). Must stay above sharpWide*sigmaMax*2 so the wide unsharp pass fits the window", min = 1, max = 5, step = 1, defaultValue = 5)
    int kernelRadius;

    @Tunable(title = "KernelNet upscale max elongation", category = "Upscale", description = "Caps the sigma ratio max(s1,s2)/min(s1,s2) - prevents knife-thin edge kernels", min = 1.0f, max = 8.0f, step = 0.5f, defaultValue = 7.0f)
    float maxElong;

    @Tunable(title = "KernelNet upscale gate curve", category = "Upscale", description = "Exponent reshaping the edge-confidence gate for the unsharp term (fraction of allowed elongation). Below 1 steepens so medium edges sharpen too; 1 is linear; flats stay near 0 either way", min = 0.1f, max = 2.0f, step = 0.05f, defaultValue = 0.35f)
    float gateExp;

    @Tunable(title = "KernelNet split chroma", category = "Upscale", description = "Reconstruct luma with the anisotropic kernels and take chroma from bicubic (1) instead of filtering all channels anisotropically (0) - same detail, less color moire", min = 0, max = 1, step = 1, defaultValue = 1)
    int splitChroma;

    @Tunable(title = "KernelNet upscale acutance", category = "Upscale", description = "Edge-aligned unsharp-mask amount: sharpened = aniso + amt*gate*(aniso - wider aniso), where gate is the used fraction of maxElong (0 in flats, 1 on strong edges); 0 keeps the output strictly convex", min = 0.0f, max = 1.5f, step = 0.05f, defaultValue = 1.3f)
    float sharpAmt;

    @Tunable(title = "KernelNet upscale acutance width", category = "Upscale", description = "Sigma multiplier of the wide pass used by the unsharp term (higher = softer wide pass, stronger bandpass). Effective value is capped at radius/(2*sigmaMax) so the wide kernel fits the window", min = 1.1f, max = 4.0f, step = 0.05f, defaultValue = 2.2f)
    float sharpWide;

    @Tunable(title = "KernelNet downscale min sigma", category = "Upscale", description = "Frozen sigma floor in input pixels used when downscaling (zoom > 1): unlike the upscale path it is NOT multiplied by zoom, so kernels stay tight and crisp. Capped at the effective max below", min = 0.1f, max = 8.0f, step = 0.05f, defaultValue = 0.65f)
    float downFloorPx;

    @Tunable(title = "KernelNet downscale max-sigma growth", category = "Upscale", description = "Scale-aware AA growth: effective max = min(sigmaMax, downFloor + growth*(zoomMax-1)). 0 keeps the tightest cap at all downscale factors; higher lets strong downscales widen toward sigmaMax for antialiasing", min = 0.0f, max = 2.0f, step = 0.05f, defaultValue = 0.5f)
    float downSigmaGrowth;

    @Tunable(title = "KernelNet downscale acutance growth", category = "Upscale", description = "Scale-aware sharpness: effective sharpAmt = sharpAmt*(1+growth*log2(zoomMax)) on downscales, recovering acutance lost to the wider AA kernel. 0 disables the boost", min = 0.0f, max = 2.0f, step = 0.05f, defaultValue = 0.75f)
    float downSharpMpy;

    @Tunable(title = "Debug: dump kernelnet params", category = "Upscale", description = "Renders the KernelNet params map (s1, s2, rho as RGB) into the debug overlay", min = 0, max = 1, step = 1, defaultValue = 0)
    int debugParams;

    @Tunable(title = "Debug: bilinear vs reconstruction", category = "Upscale", description = "0=full anisotropic reconstruction (default), 1=plain bilinear upscale of input", min = 0, max = 1, step = 1, defaultValue = 0)
    int debugUpscale;

    private GLTexture kernelsMapTex;

    // Aniso-branch proof state for the harness oracle (T2c): zoom, sigma
    // floor and target captured when the reconstruction actually renders.
    // Untouched on every passthrough/bicubic path (oracle skips then).
    private boolean anisoDone = false;
    private Point anisoTarget = null;
    private float anisoZoomX = 1f;
    private float anisoZoomY = 1f;
    private float anisoMinX = 0f;
    private float anisoMinY = 0f;
    private float anisoSigmaMaxEff = 1.0f;
    private int anisoRadiusEff = 5;
    private float anisoSharpAmtEff = 1.3f;

    public UpscaleCrop() {
        super("", "UpscaleCrop");
    }

    @Override
    public void Compile() {
    }

    /**
     * Downstream nodes (LocalLaplacian, CaptureSharpening, CorrectingFlow,
     * Sharpen2) draw into the pipeline's main ping-pong textures, which
     * Bayer2Float created at crop size. Once the crop has been expanded to
     * the full-frame output size, those targets must be rebuilt to match.
     */
    private void resizeMainTextures(Point size) {
        GLFormat fmt = new GLFormat(GLFormat.DataType.FLOAT_16, GLDrawParams.WorkDim);
        GLTexture[] mains = {basePipeline.main1, basePipeline.main2, basePipeline.main3};
        for (int i = 0; i < mains.length; i++) {
            if (mains[i] != null) {
                mains[i].close();
            }
            // main3 is demand-allocated and dead past the demosaic stage:
            // never resurrect it here (getMain3 re-creates on demand).
            if (i == 2 && mains[i] == null) {
                continue;
            }
            mains[i] = new GLTexture(size, fmt, null, GL_LINEAR, GL_CLAMP_TO_EDGE);
        }
        basePipeline.main1 = mains[0];
        basePipeline.main2 = mains[1];
        basePipeline.main3 = mains[2];
    }

    /** Frees a malloc-backed result buffer exactly once; null/view-safe. */
    private static void freeBase(ByteBuffer base) {
        if (base != null && base.isDirect()) {
            try {
                Allocator.free(base);
            } catch (Exception ignored) {
            }
        }
    }

    /**
     * Frees the KernelNet params CPU ferry on paths that never upload it
     * (non-cropped shots, invalid sizes, no-resize targets). UpscaleCrop is
     * its only consumer, so without this the ESD4D result malloc survives the
     * whole render and then leaks until process death; PostPipeline.close()
     * keeps a safety net for paths that throw before this node runs.
     */
    private static void freeUnusedKernelParams(PostPipeline pp) {
        if (pp == null) return;
        freeBase(pp.kernelParamsBase);
        pp.kernelParamsBase = null;
        pp.kernelParams = null;
        pp.kernelParamsSize = null;
        if (pp.kernelNetSingleThread != null) {
            try {
                pp.kernelNetSingleThread.join();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            pp.kernelNetSingleThread = null;
            KernelNetResult result = pp.kernelNetSingleResult.getAndSet(null);
            if (result != null) {
                freeBase(result.params());
            }
        }
    }

    /** Renders the interleaved (s1, s2, rho, 1) fp16 param buffer as a debug bitmap. */
    private void dumpParams(ShortBuffer params, Point paramsSize) {
        if (debugParams == 0 || params == null || paramsSize == null) return;
        PostPipeline pp = (PostPipeline) basePipeline;
        int[] pix = new int[paramsSize.x * paramsSize.y];
        params.position(0);
        // RGBA-interleaved (s1, s2, rho, 1) halves per texel, row-major.
        for (int i = 0; i < pix.length; i++) {
            float s1 = Math.min(Half.toFloat(params.get(i * 4)) * 0.5f, 1.0f);
            float s2 = Math.min(Half.toFloat(params.get(i * 4 + 1)) * 0.5f, 1.0f);
            float rho = Math.min(Math.max((Half.toFloat(params.get(i * 4 + 2)) + 1.0f) * 0.5f, 0.0f), 1.0f);
            int r = (int) (s1 * 255.0f);
            int g = (int) (s2 * 255.0f);
            int b = (int) (rho * 255.0f);
            pix[i] = 0xFF000000 | (r << 16) | (g << 8) | b;
        }
        Bitmap bmp = Bitmap.createBitmap(paramsSize.x, paramsSize.y, Bitmap.Config.ARGB_8888);
        bmp.setPixels(pix, 0, paramsSize.x, 0, 0, paramsSize.x, paramsSize.y);
        pp.debugData.add(bmp);
    }

    @Override
    public int halo() {
        return kernelRadius; // tunable, capped at KERN_R_MAX=5 in-shader
    }

    @Override
    public void Run() {
        GLTexture input = previousNode.WorkingTexture;
        PostPipeline pp = (PostPipeline) basePipeline;

        if (input == null) {
            freeUnusedKernelParams(pp);
            WorkingTexture = null;
            return;
        }

        // Last main3 reader (demosaic stage) is behind us in every pipeline
        // variant; nothing downstream touches it. Release the ~514 MB
        // (64 MP) before the local-contrast/sharpen chain instead of holding
        // it to runAll's tail close. Any future reader re-allocates via
        // getMain3(), so this is purely a lifetime change.
        if (basePipeline.main3 != null) {
            basePipeline.main3.close();
            basePipeline.main3 = null;
        }

        if (basePipeline.mParameters.fullRawSize == null &&
                com.particlesdevs.photoncamera.processing.render.Parameters.isResizeDisabled(
                        basePipeline.mParameters.getActiveUpscaleFactor()) &&
                !basePipeline.mParameters.isCropped) {
            // Fully native path with no resize requested: params have no
            // consumer, free the ferry instead of leaking it to close().
            freeUnusedKernelParams(pp);
            WorkingTexture = input;
            return;
        }

        if (input.mSize.x <= 0 ||
                input.mSize.y <= 0) {
            freeUnusedKernelParams(pp);
            WorkingTexture = input;
            return;
        }

        /*
         * Output size: zoom-expanded full size when cropped, otherwise the
         * input size, scaled by the active per-sensor factor. Keeps output
         * dimensions divisible by four, matching the crop/output sizing
         * convention already used by PostPipeline. Applies to cropped and
         * uncropped shots alike.
         */
        Point target = com.particlesdevs.photoncamera.processing.render.Parameters.computeResizedTarget(
                basePipeline.mParameters, input.mSize);

        if (target.equals(input.mSize)) {
            freeUnusedKernelParams(pp);
            WorkingTexture = input;
            return;
        }

        ShortBuffer params = pp.kernelParams;
        Point paramsSize = pp.kernelParamsSize;
        ByteBuffer singleBase = null;
        if (params == null && pp.kernelNetSingleThread != null) {
            // Collect the single-frame inference started by KernelNetPrep.
            try {
                pp.kernelNetSingleThread.join();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            pp.kernelNetSingleThread = null;
            KernelNetResult result = pp.kernelNetSingleResult.getAndSet(null);
            if (result != null) {
                // RGBA-interleaved fp16 view, no repack copy.
                singleBase = result.params();
                params = singleBase.asShortBuffer();
                paramsSize = new Point(result.width(), result.height());
            }
        }
        // Exactly one base buffer owns the params (merge ferry or single
        // result); it is freed after a successful upload, or on any failure
        // below since no other owner exists yet.
        ByteBuffer ownedBase = pp.kernelParamsBase != null ? pp.kernelParamsBase : singleBase;

        boolean hasParams = params != null && paramsSize != null
                && paramsSize.x > 0 && paramsSize.y > 0 && ownedBase != null;
        if (hasParams) {
            // Map-health sanity check: if the first/last param texel is NaN or
            // out of the model's range, the map is garbage and the
            // reconstruction would mirror it - fall back to the bicubic path
            // instead. RGBA-interleaved (s1, s2, rho, 1) halves per texel.
            params.position(0);
            int last = (paramsSize.x * paramsSize.y - 1) * 4;
            float s1 = Half.toFloat(params.get(0));
            float s2 = Half.toFloat(params.get(1));
            float rho = Half.toFloat(params.get(2));
            float ls1 = Half.toFloat(params.get(last));
            float ls2 = Half.toFloat(params.get(last + 1));
            float lrho = Half.toFloat(params.get(last + 2));
            if (Float.isNaN(s1) || Float.isNaN(s2) || Float.isNaN(rho)
                    || Float.isNaN(ls1) || Float.isNaN(ls2) || Float.isNaN(lrho)
                    || s1 < 0.0f || s1 > 4.0f || s2 < 0.0f || s2 > 4.0f
                    || rho < -2.0f || rho > 2.0f
                    || ls1 < 0.0f || ls1 > 4.0f || ls2 < 0.0f || ls2 > 4.0f
                    || lrho < -2.0f || lrho > 2.0f) {
                hasParams = false;
            }
        }

        if (hasParams) {
            kernelsMapTex = new GLTexture(paramsSize,
                    new GLFormat(GLFormat.DataType.FLOAT_16, 4), null, GL_LINEAR, GL_CLAMP_TO_EDGE);
            // The inference result is already RGBA-interleaved fp16 halves in
            // the exact GL_RGBA16F layout, so upload the malloc'd buffer as-is
            // with GL_HALF_FLOAT: one driver call, no repack, no FLOAT->HALF
            // conversion.
            try {
                ownedBase.position(0);
                kernelsMapTex.loadRawHalf(ownedBase);
            } catch (Throwable t) {
                freeBase(ownedBase);
                pp.kernelParamsBase = null;
                pp.kernelParams = null;
                pp.kernelParamsSize = null;
                throw t;
            }
            dumpParams(params, paramsSize);

            /*
             * Per-axis sigma floor in input pixels, scale-aware:
             * - Upscale (zoom <= 1): legacy constant floor in output pixels
             *   (outFloorPx*zoom) keeps the reconstruction equally crisp at
             *   every zoom factor, with an absolute input-pixel floor so tap
             *   weights never collapse. Capped at sigmaMaxPx: clamp(s, min,
             *   max) with min > max is undefined in GLSL.
             * - Downscale (zoom > 1): frozen floor (downFloorPx, NOT multiplied
             *   by zoom) keeps kernels tight and crisp; the effective max grows
             *   slowly with zoom for antialiasing: min(sigmaMax,
             *   downFloor+growth*(zoomMax-1)). Floor is re-capped at that
             *   effective max per axis.
             */
            float zoomX = input.mSize.x / (float) target.x;
            float zoomY = input.mSize.y / (float) target.y;
            float zoomMax = Math.max(zoomX, zoomY);
            float minX;
            float minY;
            float sigmaMaxEff;
            float sharpAmtEff;
            int radiusEff = Math.min(Math.max(kernelRadius, 1), 5);
            if (zoomMax <= 1.0f + 1e-4f) {
                minX = Math.min(Math.max(absMinPx, outFloorPx * zoomX), sigmaMaxPx);
                minY = Math.min(Math.max(absMinPx, outFloorPx * zoomY), sigmaMaxPx);
                sigmaMaxEff = sigmaMaxPx;
                sharpAmtEff = sharpAmt;
            } else {
                float downExcess = zoomMax - 1.0f;
                sigmaMaxEff = Math.min(sigmaMaxPx, downFloorPx + downSigmaGrowth * downExcess);
                sigmaMaxEff = Math.max(sigmaMaxEff, absMinPx);
                float downFloorCapped = Math.min(Math.max(absMinPx, downFloorPx), sigmaMaxEff);
                // Per-axis freeze: an axis that still upscales (mixed aspect)
                // keeps the legacy output-pixel floor on that axis.
                minX = zoomX > 1.0f ? downFloorCapped
                        : Math.min(Math.max(absMinPx, outFloorPx * zoomX), sigmaMaxEff);
                minY = zoomY > 1.0f ? downFloorCapped
                        : Math.min(Math.max(absMinPx, outFloorPx * zoomY), sigmaMaxEff);
                sharpAmtEff = sharpAmt
                        * (1.0f + downSharpMpy * (float) (Math.log(zoomMax) / Math.log(2.0)));
                if (sharpAmtEff < 0.0f) sharpAmtEff = 0.0f;
                if (sharpAmtEff > 1.5f) sharpAmtEff = 1.5f;
            }

            GLTexture out = new GLTexture(target, input.mFormat);
            glProg.useAssetProgram("upscalecrop/anisoupscale");
            anisoDone = true;
            anisoTarget = target;
            anisoZoomX = zoomX;
            anisoZoomY = zoomY;
            anisoMinX = minX;
            anisoMinY = minY;
            anisoSigmaMaxEff = sigmaMaxEff;
            anisoRadiusEff = radiusEff;
            anisoSharpAmtEff = sharpAmtEff;
            rebindAniso(input, input, 0, 0);
            glProg.drawBlocks(out);
            glProg.closed = true;
            WorkingTexture = out;
            if (((PostPipeline) basePipeline).debugTiledCompare) {
                verifyAnisoRegions(input);
            }
        } else {
            WorkingTexture = glUtils.interpolate(input, target);
        }

        // CPU copies served their purpose (params now on GPU, or unused on
        // the bicubic path): release so the ~128 MB result (50 MP) doesn't
        // ride along through the render — or leak on the fallback path.
        // Bases are Allocator-backed (see runInference); views alone must
        // never be freed. GC timing can't be trusted here.
        freeBase(pp.kernelParamsBase);
        freeBase(singleBase);
        pp.kernelParamsBase = null;
        pp.kernelParams = null;
        pp.kernelParamsSize = null;
        params = null;

        /*
         * Downstream nodes and the final GL output need to use the new texture
         * dimensions rather than the original crop dimensions. (pp.cropSize
         * intentionally keeps the crop-region size: RotateWatermark sizes its
         * sampling from its actual input texture now.)
         */
        resizeMainTextures(target);
        basePipeline.workSize = new Point(target);
    }

    /**
     * Full aniso bind sequence for an input tile (no program rebind: see
     * Initial.renderInitialBinds). Shared by production Run() (full input,
     * zero origins) and the oracle (windowed input, band origins).
     */
    private void rebindAniso(GLTexture fullIn, GLTexture input, int o0, int wy0) {
        glProg.setVar("fullSize", anisoTarget);
        glProg.setVar("scaleRatio", 1.0f / anisoZoomX, 1.0f / anisoZoomY);
        glProg.setVar("sigmaScale", sigmaScale);
        glProg.setVar("sigmaMinPx", anisoMinX, anisoMinY);
        glProg.setVar("sigmaMaxPx", anisoSigmaMaxEff);
        glProg.setVar("strength", anisoStrength);
        glProg.setVar("kernelRadius", anisoRadiusEff);
        glProg.setVar("sharpAmt", anisoSharpAmtEff);
        glProg.setVar("sharpWide", sharpWide);
        glProg.setVar("maxElong", maxElong);
        glProg.setVar("gateExp", gateExp);
        glProg.setVar("splitChroma", splitChroma);
        glProg.setVar("debugMode", debugUpscale);
        glProg.setTexture("InputBuffer", input);
        glProg.setTexture("KernelsMap", kernelsMapTex);
        glProg.setVar("u_tileOrigin", 0, o0);
        glProg.setVar("u_winOrigin", 0, wy0);
        glProg.setVar("u_winFullSize", (float) fullIn.mSize.x, (float) fullIn.mSize.y);
    }

    /**
     * Harness oracle (debugTiledCompare, T3c): re-renders output bands from
     * halo-expanded INPUT windows (exactly as the production driver will) and
     * requires bit-exactness vs the full aniso render. No edge exclusions:
     * absolute coords, edge-touching clamp equality, and window margins make
     * every band exact, including image borders. Skipped on every non-aniso
     * path (alias/bicubic render nothing new).
     */
    private void verifyAnisoRegions(GLTexture fullIn) {
        if (!anisoDone || anisoTarget == null) {
            Log.d("TiledHarness", "upscale strips skipped (not aniso)");
            return;
        }
        GLTexture fullOut = WorkingTexture;
        int imgW = fullOut.mSize.x;
        int imgH = fullOut.mSize.y;
        int inH = fullIn.mSize.y;
        // Halo covers the reconstruction window plus resampling footprint.
        int halo = Math.max(1, kernelRadius) + 1;
        float worst = 0f;
        for (int[] band : TileDriver.snapBands(imgH, 512)) {
            int o0 = band[0], rows = band[1] - band[0];
            int[] win = TileDriver.inputWindow(o0, o0 + rows, inH, anisoZoomY, halo);
            int wy0 = win[0], wy1 = win[1];
            // Input-sized width: the crop is narrower than the target, and a
            // target-width tile would make the blit scale instead of copy.
            int inW = fullIn.mSize.x;
            GLTexture inTile = new GLTexture(new android.graphics.Point(inW, wy1 - wy0),
                    fullIn.mFormat);
            TileDriver.blitBand(fullIn, inTile, wy0, wy1 - wy0);
            float inDiff = TileDriver.compareBand(fullIn, inTile, inW, wy0, wy1 - wy0,
                    "TiledHarness-blit");
            if (inDiff != 0f) {
                Log.e("TiledHarness", "upscale blit band [" + wy0 + "," + wy1
                        + ") maxDiff=" + inDiff);
            }
            GLTexture reg = new GLTexture(new android.graphics.Point(imgW, rows),
                    fullOut.mFormat);
            tileY0 = o0;
            tileY1 = o0 + rows;
            tileOut = reg;
            WorkingTexture = reg;
            try {
                rebindAniso(fullIn, inTile, o0, wy0);
                glProg.drawBlocks(reg);
                float m = TileDriver.compareBand(fullOut, reg, imgW, o0, rows, "TiledHarness");
                if (m > worst) {
                    worst = m;
                }
            } catch (Throwable t) {
                Log.e("TiledHarness", "upscale band [" + o0 + "," + (o0 + rows) + ") failed", t);
                worst = Float.POSITIVE_INFINITY;
            } finally {
                try {
                    inTile.close();
                } catch (Exception ignored) {
                }
                try {
                    reg.close();
                } catch (Exception ignored) {
                }
            }
        }
        Log.d("TiledHarness", "upscale strips maxDiff=" + worst);
        // Windowed remap rounding (uvWin vs direct uv) can flip the last
        // HALF bit on isolated pixels (~2e-6); anything above 1e-5 is real.
        if (worst > 0f && worst <= 1e-5f) {
            Log.d("TiledHarness", "upscale strips within fp dust, accepted");
        }
        tileY0 = 0;
        tileY1 = -1;
        tileOut = null;
        WorkingTexture = fullOut;
        glProg.setVar("u_tileOrigin", 0, 0);
        glProg.setVar("u_winOrigin", 0, 0);
        glProg.setTexture("InputBuffer", fullIn);
        glProg.setTexture("KernelsMap", kernelsMapTex);
    }

    @Override
    public void AfterRun() {
        if (kernelsMapTex != null) {
            kernelsMapTex.close();
            kernelsMapTex = null;
        }
    }
}
