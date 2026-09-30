package com.particlesdevs.photoncamera.processing.opengl.scripts;

import android.graphics.Point;
import android.util.Pair;

import com.particlesdevs.photoncamera.processing.cpu.HalideAlignment;
import com.particlesdevs.photoncamera.processing.ml.KernelNetNcnnProcessor;
import com.particlesdevs.photoncamera.processing.ml.KernelNetResult;
import com.particlesdevs.photoncamera.processing.opengl.GLBuffer;
import com.particlesdevs.photoncamera.settings.annotations.Tunable;
import com.particlesdevs.photoncamera.util.Log;

import com.particlesdevs.photoncamera.app.PhotonCamera;
import com.particlesdevs.photoncamera.processing.ImageFrame;
import com.particlesdevs.photoncamera.processing.opengl.GLContext;
import com.particlesdevs.photoncamera.processing.opengl.GLCoreBlockProcessing;
import com.particlesdevs.photoncamera.processing.opengl.GLDrawParams;
import com.particlesdevs.photoncamera.processing.opengl.GLFormat;
import com.particlesdevs.photoncamera.processing.opengl.GLOneScript;
import com.particlesdevs.photoncamera.processing.opengl.GLProg;
import com.particlesdevs.photoncamera.processing.opengl.GLTexture;
import com.particlesdevs.photoncamera.processing.opengl.GLUtils;
import com.particlesdevs.photoncamera.processing.render.NoiseModeler;
import com.particlesdevs.photoncamera.processing.render.Parameters;
import com.particlesdevs.photoncamera.settings.DynamicNoiseStore;
import com.particlesdevs.photoncamera.util.Allocator;
import com.particlesdevs.photoncamera.util.BufferUtils;
import com.particlesdevs.photoncamera.util.Math2;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;
import java.util.ArrayList;
import java.util.Objects;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicReference;

import static android.opengl.GLES20.GL_CLAMP_TO_EDGE;
import static android.opengl.GLES20.GL_LINEAR;
import static android.opengl.GLES20.GL_MIRRORED_REPEAT;
import static android.opengl.GLES20.GL_NEAREST;

public class ESD4D extends GLOneScript {
    public Parameters parameters;
    ArrayList<ImageFrame> images;
    //ByteBuffer alignment;
    GLProg glProg;
    GLUtils glUtils;
    public ESD4D(Point size, ArrayList<ImageFrame> images) {
        super(size, new GLCoreBlockProcessing(size,new GLFormat(GLFormat.DataType.FLOAT_16), GLDrawParams.Allocate.Direct),"", "ESD4D", true);
        this.glProg = glOne.glProgram;
        this.images = images;
        //this.alignment = alignment;
    }

    /**
     * KernelNet runs natively through ncnn on the Vulkan backend (see
     * {@link KernelNetNcnnProcessor}).
     */

    @Override
    public void Compile(){}
    private int baseCnt = 0;

    private GLTexture getBase(){
        if(baseCnt == 0){
            baseCnt++;
            return baseAlter;
        } else {
            baseCnt = 0;
            return base;
        }
    }
    float noiseS;
    float noiseO;
    GLBuffer hotPixelBuffer;
    int hotPixelCount;
    /** Sensor red-site offset ((cfa%2, cfa/2)); the packed grid is rawHalf + cfaShift. */
    Point cfaShift;
    /** Packed texture size (rawSize/2 + cfaShift) shared by all quad-packed stages. */
    Point packedSize;
    @Tunable(title = "Max hotPixels", category = "Merge", description = "Statistical cpu filtering count threshold", min = 16384, max = 262144, step = 1000, defaultValue = 65535)
    int MAX_HOT_PIXELS;
    @Tunable(title = "Max reasonable hotPixels", category = "Merge", description = "Statistical cpu filtering count threshold", min = 1000, max = 10000, step = 100, defaultValue = 2000)
    int MAX_REASONABLE_HOTPIXELS;

    @Tunable(title = "Enable hotPixel correction", category = "Merge", min = 0, max = 1, step = 1, defaultValue = 0)
    boolean enableHotPixelCorrection;

    /**
     * Averages up to 10 frames (or fewer if not available) into a single rgba16f texture
     * at rawHalf resolution. Uses incremental mix: mix(current, new, 1/(i+1)) which yields
     * a proper running average without overflow.
     */
    private GLTexture buildAveragedFrame(int tile) {
        int maxFrames = Math.min(10, images.size());

        GLTexture avgA     = new GLTexture(packedSize, new GLFormat(GLFormat.DataType.FLOAT_16, 4), null, GL_NEAREST, GL_CLAMP_TO_EDGE);
        GLTexture avgB     = new GLTexture(packedSize, new GLFormat(GLFormat.DataType.FLOAT_16, 4), null, GL_NEAREST, GL_CLAMP_TO_EDGE);
        GLTexture tempFloat = new GLTexture(packedSize, new GLFormat(GLFormat.DataType.FLOAT_16, 4), null, GL_NEAREST, GL_CLAMP_TO_EDGE);
        GLTexture tempRaw  = maxFrames > 1
                ? new GLTexture(parameters.rawSize, new GLFormat(GLFormat.DataType.FLOAT_16, 1), null, GL_NEAREST, GL_CLAMP_TO_EDGE)
                : null;

        GLTexture avgCurrent = avgA;
        GLTexture avgNext    = avgB;

        for (int i = 0; i < maxFrames; i++) {
            GLTexture rawSrc = (i == 0) ? inputBase : tempRaw;
            if (i > 0) {
                tempRaw.loadRawHalf(images.get(i).buffer);
            }

            // Convert raw Bayer -> normalized rgba16f vec4 (one texel per 2x2 Bayer quad)
            glProg.setLayout(tile, tile, 1);
            glProg.useAssetProgram("merge/merge00", true);
            glProg.setVar("exposure", 1.0f / images.get(0).pair.layerMpy);
            glProg.setVar("createDiff", 0);
            glProg.setVar("cfaShift", cfaShift);
            glProg.setTexture("inTexture", rawSrc);
            glProg.setTextureCompute("outTexture", tempFloat, true);
            glProg.computeAuto(packedSize, 1);

            // Incremental mix: mix(currentAvg, newFrame, 1/(i+1))
            // i=0 → weight=1.0 copies newFrame wholesale (currentAvg is uninitialised zeros)
            float weight = 1.0f / (i + 1);
            glProg.setLayout(tile, tile, 1);
            glProg.useAssetProgram("merge/avermix", true);
            glProg.setTextureCompute("currentTexture", avgCurrent, false);
            glProg.setTextureCompute("newTexture",     tempFloat,  false);
            glProg.setTextureCompute("outTexture",     avgNext,    true);
            glProg.setVar("weight", weight);
            glProg.computeAuto(packedSize, 1);

            // Ping-pong: avgNext becomes the new accumulator
            GLTexture swap = avgCurrent;
            avgCurrent = avgNext;
            avgNext    = swap;
        }

        avgNext.close();
        tempFloat.close();
        if (tempRaw != null) tempRaw.close();
        Log.d(Name, "Averaged " + maxFrames + " frame(s) for hot pixel detection");
        return avgCurrent; // caller must close
    }

    private GLBuffer detectHotPixels(GLTexture avgTex) {
        GLBuffer res = new GLBuffer(MAX_HOT_PIXELS*4+1, new GLFormat(GLFormat.DataType.UNSIGNED_32));
        glProg.setLayout(8,8,1);
        glProg.useAssetProgram("merge/hotpixeldetect", true);
        glProg.setVar("noiseS", noiseS);
        glProg.setVar("noiseO", noiseO);
        glProg.setVar("detectThr", (float) detectThr);
        glProg.setVar("maxCount", MAX_HOT_PIXELS);
        glProg.setTexture("inTexture", avgTex);
        glProg.setBufferCompute("HotPixelList",res);
        glProg.computeAuto(base.mSize, 1);
        int[] outputArr = res.readBufferIntegers(false);
        int rawCount = Math.min(outputArr[0], MAX_HOT_PIXELS);
        Log.d(Name, "Hot pixels detected (raw):" + rawCount);
        
        hotPixelCount = filterHotPixels(outputArr, rawCount, res);
        Log.d(Name, "Hot pixels after filtering:" + hotPixelCount);
        return res;
    }
    
    private int filterHotPixels(int[] data, int count, GLBuffer buffer) {
        if (count <= 0) return 0;
        
        // Structure: data[0] = count, then for each pixel: x, y, channels, strength
        ArrayList<int[]> candidates = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            int idx = 1 + i * 4;
            int x = data[idx];
            int y = data[idx + 1];
            int ch = data[idx + 2];
            int strength = data[idx + 3];
            candidates.add(new int[]{x, y, ch, strength, i});
        }
        
        // If too many detections, likely false positives - filter by strength
        if (count > MAX_REASONABLE_HOTPIXELS) {
            Log.d(Name, "Too many hot pixels, filtering by strength");
            // Sort by strength (descending)
            candidates.sort((a, b) -> Integer.compare(b[3], a[3]));
            // Keep only the strongest
            while (candidates.size() > MAX_REASONABLE_HOTPIXELS) {
                candidates.remove(candidates.size() - 1);
            }
        }
        
        ArrayList<int[]> filtered = candidates;
        
        // Statistical outlier removal based on strength distribution
        if (filtered.size() > 50) {
            // Calculate mean and stddev of strength
            double sum = 0, sumSq = 0;
            for (int[] c : filtered) {
                sum += c[3];
                sumSq += (double)c[3] * c[3];
            }
            double mean = sum / filtered.size();
            double variance = sumSq / filtered.size() - mean * mean;
            double stddev = Math.sqrt(Math.max(variance, 1));
            
            // Remove weak outliers (strength < mean - 1.5*stddev)
            double threshold = mean - 1.5 * stddev;
            ArrayList<int[]> statistical = new ArrayList<>();
            for (int[] c : filtered) {
                if (c[3] >= threshold) {
                    statistical.add(c);
                }
            }
            Log.d(Name, "Statistical filtering: mean=" + (int)mean + " stddev=" + (int)stddev + " thr=" + (int)threshold);
            Log.d(Name, "Removed " + (filtered.size() - statistical.size()) + " weak detections");
            filtered = statistical;
        }
        
        // Repack filtered results back into buffer
        int finalCount = filtered.size();
        data[0] = finalCount;
        for (int i = 0; i < finalCount; i++) {
            int[] c = filtered.get(i);
            int idx = 1 + i * 4;
            data[idx] = c[0];
            data[idx + 1] = c[1];
            data[idx + 2] = c[2];
            data[idx + 3] = c[3];
        }
        buffer.uploadBuffer(data, finalCount * 4 + 1);
        
        return finalCount;
    }

    private void correctHotPixelsBase(GLBuffer buffer, int count){
        if (count > 0) {
            glProg.setLayout(64, 1, 1);
            glProg.useAssetProgram("merge/hotpixelcorrect", true);
            glProg.setBufferCompute("HotPixelList", buffer);
            glProg.setTextureCompute("inTexture", base, false);
            glProg.setTextureCompute("outTexture", base, true);
            glProg.computeManual((count + 63) / 64, 1, 1);
            Log.d(Name, "Hot pixels corrected in base:" + count);
        }
    }

    private void correctHotPixelsInAlter(GLBuffer buffer, int count){
        if (count > 0) {
            glProg.setLayout(64, 1, 1);
            glProg.useAssetProgram("merge/hotpixelcorrect", true);
            glProg.setBufferCompute("HotPixelList", buffer);
            glProg.setTextureCompute("inTexture", alter, false);
            glProg.setTextureCompute("outTexture", alter, true);
            glProg.computeManual((count + 63) / 64, 1, 1);
            Log.d(Name, "Hot pixels corrected in alter:" + count);
        }
    }

    private void hotPixels(){
        GLTexture avgTex = buildAveragedFrame(8);
        hotPixelBuffer = detectHotPixels(avgTex);
        avgTex.close();
        correctHotPixelsBase(hotPixelBuffer, hotPixelCount);
    }

    GLTexture inputBase;
    GLTexture baseDiff;
    GLTexture base;
    GLTexture baseAlter;
    //GLTexture;
    GLTexture brightMap;
    /** CPU copy of brightMap (float32 grayscale luma in [0,1]) set by {@link #exportBrightMap()}. */
    public FloatBuffer brightMapCPU;
    /** Unpacked size of {@link #brightMapCPU} (row-major, width*height floats). */
    public Point brightMapCPUSize;
    /** KernelNet half-res parameter texture (s1, s2, rho in RGBA16F) for the anisotropic filter. */
    public GLTexture kernelsMap;
    /** CPU copy of the KernelNet params (RGBA16F halves: s1, s2, rho, 1 per
     * texel) for reuse in the post pipeline. Set by {@link #createKernelsMap}
     * alongside the texture upload; a view, never freed. */
    public ShortBuffer kernelsMapCPU;
    /** Size of {@link #kernelsMapCPU}. */
    public Point kernelsMapCPUSize;
    /**
     * Base direct buffer behind {@link #kernelsMapCPU} (the inference
     * result). Single owner: whoever holds it frees it exactly once via
     * {@code Allocator.free} after the GPU upload — views don't free.
     */
    public ByteBuffer kernelsMapBase;
    /** Noise sigma fed to KernelNet (captured pre-merge-inflation). */
    float kernelSigma;
    GLTexture result;
    GLTexture inputAlter;
    /**
     * Second slot of the alter-frame upload ring. The merge loop alternates
     * between {@link #inputAlter} and this texture so an upload never targets
     * the texture the previous frame's merge00 is still reading. A GLsync
     * fence recorded right after each merge00 guards the two-frames-later
     * reuse.
     */
    GLTexture inputAlterAlt;
    /** GLsync handles for the two upload-ring slots (0 = none). */
    private final long[] alterUploadFences = new long[2];
    /** GLsync handles for the noise-blend raw upload ring (two slots). */
    private final long[] noiseBlendUploadFences = new long[2];
    GLTexture alter;
    GLTexture alignmentTex;
    /** Halide aligner running on a worker thread (null before launch/after join). */
    private HalideAlignment halideAlignment;
    private Thread halideThread;
    private final AtomicReference<Throwable> halideError = new AtomicReference<>();
    private long halideLaunchMs;
    /** Dense optical-flow alignment (FlowNet); non-null when useNcnnFlow ran. */
    FlowNetAlignment flowNetAlignment;
    @Tunable(title = "SR detail layer", category = "Merge", description = "Accumulate motion-compensated per-frame residuals into a detail layer injected at merge output; active on multi-frame upscales and explicit 1x (enhanced native), silent otherwise", min = 0, max = 1, step = 1, defaultValue = 1)
    boolean srDetailEnable = true;
    @Tunable(title = "SR detail strength", category = "Merge", description = "Gain applied to the normalized SR detail layer at merge output (0 keeps it allocated but inert)", min = 0.0f, max = 2.0f, step = 0.05f, defaultValue = 0.6f)
    float srDetailStrength = 0.6f;
    @Tunable(title = "SR detail clamp", category = "Merge", description = "Per-frame residual clamp in normalized units: consistent subpixel detail passes, motion saturates instead of ghosting", min = 0.005f, max = 0.5f, step = 0.005f, defaultValue = 0.03f)
    float srDetailClamp = 0.03f;
    @Tunable(title = "SR coring low", category = "Merge", description = "Detail magnitudes below this (noise sigmas) are suppressed as averaged-noise residue. Lowered from 1.0: the enhanced-native 1x output read as more denoised but not more detailed than Disabled, because the fine low-amplitude detail that carries that impression sat entirely below the 1-sigma knee; grain is acceptable in this pipeline, so the knee opens to 0.5", min = 0.0f, max = 5.0f, step = 0.1f, defaultValue = 0.5f)
    float srCoring0 = 0.5f;
    @Tunable(title = "SR coring high", category = "Merge", description = "Detail magnitudes above this (noise sigmas) fully pass; smooth ramp between low and high. Lowered from 2.5 with the knee: the full-pass point now sits just above the noise floor instead of 2.5 sigma", min = 0.5f, max = 8.0f, step = 0.1f, defaultValue = 1.5f)
    float srCoring1 = 1.5f;
    @Tunable(title = "SR memory cap", category = "Merge", description = "Skip the SR paths when their extra GPU/ferry memory would exceed this many MB (the detail layer's packed accumulators, or the full drizzle's two output-size accumulators plus the luma texture)", min = 64, max = 8192, step = 64, defaultValue = 2048)
    int srMemoryCapMB = 2048;
    @Tunable(title = "SR max expansion", category = "Merge", description = "Largest output/raw expansion (zoom x upscale factor) the SR paths run at. The fused luma's honest band ends at the raw Nyquist (0.5/expand c/px), so it leaves the eye's sensitive range around 4x while its magnified raw grain keeps growing - past that the aniso reconstruction is the better default. Raise to test high zoom crops; the memory gate still applies (small crops fit easily)", min = 1.0f, max = 20.0f, step = 0.25f, defaultValue = 4.0f)
    float srMaxExpand = 4.0f;
    @Tunable(title = "Full SR drizzle", category = "Merge", description = "Drizzle burst frames onto the output grid (translation + local block motion); replaces merge/inference limits on multi-frame upscales, silent otherwise", min = 0, max = 1, step = 1, defaultValue = 1)
    boolean srFullEnable = true;
    @Tunable(title = "SR trust floor", category = "Merge", description = "Minimum drizzle weight for an alter frame that disagrees coherently with the running estimate (ghost guard; 1 = pure average). Now 0: the drizzle fully compensates sub-pixel motion, unlike the merge's integer warp plus spatial blend, so a residual misregistration of ~1px directly blurs the fusion - and a 0.15 floor let those frames contribute 55% of the weight. Simulated at a +-1px residual: floor 0.15 -> fused band 0.819, injected 0.698; floor 0 -> 1.509 and 0.981, i.e. the sharp base frame wins. Set >0 only if you see ghosting from moving subjects", min = 0.0f, max = 1.0f, step = 0.05f, defaultValue = 0.0f)
    float srTrustFloor = 0.0f;
    @Tunable(title = "SR trust band", category = "Merge", description = "Low-frequency misregistration residual (normalized units; four sigma of the pre-inflation noise model widens it) at which an alter frame's drizzle weight halves", min = 0.005f, max = 0.2f, step = 0.005f, defaultValue = 0.05f)
    float srTrustBand = 0.05f;
    @Tunable(title = "SR clip attenuation", category = "Merge", description = "Weight attenuation for samples at the raw ceiling (0 = ignore clipping, 1 = clipped samples do not vote): burst frames clip at different levels and a clipped sample carries no highlight detail", min = 0.0f, max = 1.0f, step = 0.05f, defaultValue = 0.7f)
    float srClipAtten = 0.7f;
    @Tunable(title = "SR alignment refinement", category = "Merge", description = "Bounded per-cell Lucas-Kanade sub-pixel refinement of the drizzle motion against the running base (0 = off): the fusion's sampling positions are only as good as the alignment", min = 0.0f, max = 1.0f, step = 0.05f, defaultValue = 1.0f)
    float srRefine = 1.0f;
    @Tunable(title = "SR jitter", category = "Merge", description = "Synthesized per-pixel, per-frame sub-pixel jitter (raw px) for static bursts, where every frame would otherwise sample one lattice phase and the sensor's aliasing never cancels; scaled down where the frame already moved, so handheld bursts are unaffected. 0 = off", min = 0.0f, max = 1.0f, step = 0.05f, defaultValue = 0.25f)
    float srJitter = 0.25f;
    /**
     * Pre-inflation noise model (independent of the merge strength setting):
     * the SR fusion band must not scale with the denoise slider.
     */
    float srBaseNoiseS, srBaseNoiseO;

    /**
     * Rebuilds the per-site cross-channel luma for one packed frame. The
     * drizzle fuses this field instead of the packed mosaic itself: a Bayer
     * site's own lattice cannot carry the sensor band's corners, and the
     * demosaic is what does. Returns false when the pass cannot run.
     */
    private boolean dispatchSrLuma(GLTexture packed, float[] rw, float[] gw, float[] bw) {
        if (srLumaTex == null || packed == null || rw == null || gw == null || bw == null) {
            return false;
        }
        try {
            float[] wp = (parameters != null && parameters.whitePoint != null
                    && parameters.whitePoint.length >= 3)
                    ? parameters.whitePoint : new float[]{1f, 1f, 1f};
            glProg.setLayout(8, 8, 1);
            glProg.useAssetProgram("merge/srluma", true);
            glProg.setTexture("alterPacked", packed);
            glProg.setVar("srCfa", cfaShift);
            glProg.setVar("srRw", rw[0], rw[1], rw[2], rw[3]);
            glProg.setVar("srGw", gw[0], gw[1], gw[2], gw[3]);
            glProg.setVar("srBw", bw[0], bw[1], bw[2], bw[3]);
            glProg.setVar("srWhitePoint", wp);
            glProg.setTextureCompute("srLumaOut", srLumaTex, true);
            glProg.computeAuto(srLumaSize, 1);
            return true;
        } catch (Throwable t) {
            Log.e("ESD4D", "SR luma pass failed", t);
            return false;
        }
    }

    /** Binds the SR fusion trust uniforms, shared by both drizzle paths. */
    private void bindSrTrust() {
        glProg.setVar("srTrustFloor", srTrustFloor);
        glProg.setVar("srTrustBand", srTrustBand);
        glProg.setVar("srNoiseS0", srBaseNoiseS);
        glProg.setVar("srNoiseO0", srBaseNoiseO);
        glProg.setVar("srClipAtten", srClipAtten);
    }

    /**
     * Full-SR extra memory estimate against {@link #srMemoryCapMB}: two
     * output-size RGBA16F accumulators, the raw-size luma texture, and the
     * CPU ferry of the accumulator (the resolve lives in the post pipeline) -
     * all alive together at export, since releasing the idle accumulator
     * first put GL deletes in the readback's path. The target is the zoom-expanded
     * full frame times the factor, so a small crop at a high factor is the
     * expensive case. The engagement preview and the real gate must agree: if
     * the preview says the drizzle fits and the gate then fails, the detail
     * layer was skipped for a drizzle that cannot afford to run and the shot
     * gets no multi-frame reconstruction at all.
     */
    private boolean srFullFitsMemory() {
        try {
            if (parameters == null || parameters.rawSize == null) return false;
            if (parameters.rawSize.x <= 0 || parameters.rawSize.y <= 0) return false;
            Point tgt = Parameters.computeResizedTarget(parameters, parameters.rawSize);
            long out = (long) tgt.x * (long) tgt.y;
            long raw = (long) parameters.rawSize.x * (long) parameters.rawSize.y;
            // Packed R32UI accumulators: the two drizzle ping-pong textures
            // (2 x 4 B/output px), the per-site luma texture (8 B/raw px), and
            // the CPU ferry (4 B/output px) unless the shared-group handoff
            // replaces it with a texture name.
            long bytes = out * 8L + raw * 8L + (gpuHandoff ? 0L : out * 4L);
            long cap = (long) srMemoryCapMB * 1024L * 1024L;
            if (bytes > cap) {
                Log.d("ESD4D", "Full-SR skipped: needs " + (bytes / 1048576)
                        + "MB over cap " + srMemoryCapMB + "MB");
                return false;
            }
            return true;
        } catch (Throwable t) {
            return false;
        }
    }

    /**
     * Joins the KernelNet worker and uploads the parameter map, once. SR
     * paths call it before their base seed so the reference frame is
     * reconstructed with the same steered kernel as the alter frames; the
     * merge loop calls it before the first combine pass (a no-op afterwards).
     * Returns null so the caller can clear its thread reference in place.
     */
    private Thread joinKernelNet(Thread t,
            AtomicReference<KernelNetResult> result, String where) {
        if (t == null) return null;
        long joinT = System.currentTimeMillis();
        try {
            t.join();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        kernelsMap = createKernelsMap(result.get());
        // Inference joined and params uploaded: both CPU copies are dead past
        // this point (GPU textures carry on).
        brightMapCPU = null;
        brightMapCPUSize = null;
        gpuSyncProfile();
        Log.d("ESD4D", "Stage[merge:kernelnet-join] elapsed:"
                + (System.currentTimeMillis() - joinT) + " ms " + where);
        return null;
    }

    /**
     * Allocates the per-cell sub-pixel refinement map (alignment grid) once
     * per shot. The refinement pass itself runs per alter; any failure leaves
     * the map null and the drizzles use the raw atlas motion.
     */
    /**
     * Guided upsample of FlowNet's dense flow to a frame-covering motion
     * field (merge/srflowresample) using the running base as the guidance
     * image: the same edge-aware local linear fit mergeAlignFlow applies for
     * its own per-pixel warp. The drizzles sample the result bilinearly, so
     * their motion follows image edges instead of blending across them.
     * Rebuilt per alter frame; the texture is reused across frames.
     */
    private boolean dispatchSrFlowResample(GLTexture flowTex) {
        int gw = Math.max(1, parameters.rawSize.x / 4);
        int gh = Math.max(1, parameters.rawSize.y / 4);
        if (srFlowGuided == null || srFlowGuidedSize == null
                || srFlowGuidedSize.x != gw || srFlowGuidedSize.y != gh) {
            if (srFlowGuided != null) {
                srFlowGuided.close();
            }
            srFlowGuided = new GLTexture(new Point(gw, gh),
                    new GLFormat(GLFormat.DataType.FLOAT_16, 4), null, GL_LINEAR, GL_CLAMP_TO_EDGE);
            srFlowGuidedSize = new Point(gw, gh);
        }
        glProg.setLayout(8, 8, 1);
        glProg.useAssetProgram("merge/srflowresample", true);
        glProg.setTexture("flowIn", flowTex);
        glProg.setTexture("guidePacked", base);
        glProg.setVar("packedSize", new Point(parameters.rawSize.x / 2, parameters.rawSize.y / 2));
        glProg.setTextureCompute("flowOut", srFlowGuided, true);
        glProg.computeAuto(srFlowGuided.mSize, 1);
        return true;
    }

    /**
     * Alignment texture the SR drizzles read: the guided flow when FlowNet
     * produced one this shot, the cell atlas otherwise.
     */
    private GLTexture srAlignTex() {
        return Objects.equals(alignerSelect, "flownet") && srFlowGuided != null
                ? srFlowGuided : alignmentTex;
    }

    private void ensureSrRefMap() {
        if (srRefine <= 0f || srRefTex != null || parameters == null
                || parameters.alignmentSize == null
                || parameters.alignmentSize.x <= 0 || parameters.alignmentSize.y <= 0) {
            return;
        }
        try {
            // Twice the atlas resolution: the per-cell LK then corrects on
            // 8-raw-px cells instead of the atlas's 16, which is where the
            // within-cell residual from rotation and rolling shutter lives
            // (a ~1px uncorrected residual is what blurs the drizzle fusion).
            srRefSize = new Point(parameters.alignmentSize.x * 2, parameters.alignmentSize.y * 2);
            srRefTex = new GLTexture(srRefSize,
                    new GLFormat(GLFormat.DataType.FLOAT_16, 4), null, GL_LINEAR, GL_CLAMP_TO_EDGE);
        } catch (Throwable t) {
            Log.e("ESD4D", "SR refine map alloc failed, disabled", t);
            srRefTex = null;
            srRefSize = null;
        }
    }

    /**
     * Packed-channel to canonical RGB mapping for the drizzle warp: returns
     * {R,G0,G1,B} channel indices, or null for layouts without a defined
     * quad contract (quad/mono/...). RGGB/GRBG/GBRG pack canonically;
     * BGGR swaps R/B in r/a (see merge00/merge2o), which channel-wise merge
     * stages tolerate but RGB assembly must decode explicitly.
     */
    private static int[] srChannelMap(int cfaPattern, Point cfaShift) {
        if (cfaPattern < 0 || cfaPattern > 3 || cfaShift == null) return null;
        int sx = cfaShift.x & 1, sy = cfaShift.y & 1;
        int r = -1, g0 = -1, g1 = -1, b = -1;
        for (int c = 0; c < 4; c++) {
            int px = (sx + (c & 1)) & 1;
            int py = (sy + ((c >> 1) & 1)) & 1;
            int col; // 0=R,1=G,2=B
            if (cfaPattern == 0) { // RGGB
                col = (px == 0) ? (py == 0 ? 0 : 1) : (py == 0 ? 1 : 2);
            } else if (cfaPattern == 1) { // GRBG
                col = (px == 0) ? (py == 0 ? 1 : 2) : (py == 0 ? 0 : 1);
            } else if (cfaPattern == 2) { // GBRG
                col = (px == 0) ? (py == 0 ? 1 : 0) : (py == 0 ? 2 : 1);
            } else { // BGGR
                col = (px == 0) ? (py == 0 ? 2 : 1) : (py == 0 ? 1 : 0);
            }
            if (col == 0) r = c;
            else if (col == 2) b = c;
            else if (g0 < 0) g0 = c;
            else g1 = c;
        }
        if (r < 0 || b < 0 || g0 < 0 || g1 < 0) return null;
        return new int[]{r, g0, g1, b};
    }
    @Tunable(title = "Full SR motion bound", category = "Merge", description = "Maximum drizzle motion magnitude in packed px; real handheld motion passes, garbage vectors cannot mirror the frame", min = 8, max = 256, step = 8, defaultValue = 64)
    float srMotionMax = 64f;
    /** SR detail ping-pong accumulators (packed RGBA16F); null unless srActive. */
    GLTexture srAccA;
    GLTexture srAccB;
    /** Accumulator holding the final SR detail sum (one of the above). */
    GLTexture srAccFinal;
    /** Normalized SR detail highpass (packed RGBA16F); bound by merge2o, exported for post. */
    GLTexture srHP;
    /** SR detail highpass for post-upscale application (packed RGBA16F halves); may be null. */
    public ByteBuffer srDetailBase;
    /** View of {@link #srDetailBase}; never freed (see kernelsMapCPU). */
    public ShortBuffer srDetailCPU;
    /** Size of {@link #srDetailCPU} (packed domain). */
    public Point srDetailCPUSize;
    /** Packing shift used for {@link #srDetailCPU} (copy of {@link #cfaShift}); may be null. */
    public Point srDetailShift;
    /** Full-SR ping-pong accumulators (target R32UI, luma + weight packed as halves); null unless srFullActive. */
    GLTexture srDriA;
    GLTexture srDriB;
    /** Full-SR drizzled image for post (target RGBA16F halves: rgb, weight); may be null. */
    public ByteBuffer srFullBase;
    /** View of {@link #srFullBase}; never freed (see kernelsMapCPU). */
    public ShortBuffer srFullCPU;
    /** Size of {@link #srFullCPU} (output grid). */
    public Point srFullSize;
    /**
     * GPU handoff: the post's context is created in this context's EGL share
     * group, so the drizzle accumulator (and the 3a detail texture) cross the
     * pipeline boundary as texture names instead of full-size CPU ferries.
     * Set by {@link #probeGpuHandoff()} before Run; when false the CPU ferry
     * paths are used exactly as before.
     */
    public boolean gpuHandoff = false;
    /** Shared accumulator texture name handed to the post (0 = CPU ferry). */
    public int srFullTexID = 0;
    /** Shared 3a detail texture name handed to the post (0 = CPU ferry). */
    public int srDetailTexID = 0;
    /** True once the full-SR accumulator is allocated for this shot. */
    boolean srFullActive;
    /** Set by the processor when a DNG save is requested (gates Bayer drizzle). */
    public boolean saveDngWanted = false;
    /** Bayer drizzle ping-pong (target Bayer grid, value + weight packed as halves in R32UI); null unless srBayerActive. */
    GLTexture srBayA;
    GLTexture srBayB;
    /**
     * Per-site cross-channel luma at the crop's raw grid (rebuilt per frame by
     * merge/srluma); the JPEG drizzle fuses this field rather than the packed
     * mosaic, whose per-channel lattices cannot carry the sensor band's
     * corners. Null when the pass is unavailable.
     */
    GLTexture srLumaTex;
    Point srLumaSize;
    /**
     * Per-cell sub-pixel motion correction (alignment grid, packed texels),
     * rebuilt for every alter by the refinement pass; null when disabled or
     * failed.
     */
    GLTexture srRefTex;
    Point srRefSize;
    /**
     * Guided (edge-aware) resample of FlowNet's dense flow, rebuilt per alter
     * from the running base as the guide (merge/srflowresample). The drizzles
     * sample this instead of the raw low-res flow; null outside FlowNet.
     */
    GLTexture srFlowGuided;
    Point srFlowGuidedSize;
    /** True once the Bayer accumulators are allocated for this shot. */
    boolean srBayerActive;
    /** Bayer drizzle export (target Bayer halves, scalar in .r); may be null. */
    public ByteBuffer srBayerBase;
    /** View of {@link #srBayerBase}; never freed (see kernelsMapCPU). */
    public ShortBuffer srBayerCPU;
    /** Size of {@link #srBayerCPU} (target Bayer grid). */
    public Point srBayerCPUSize;
    /** Frames accumulated into {@link #srAccFinal} (normalization divisor). */
    int srAccumFrames;
    /** True once the SR accumulators are allocated for this shot. */
    boolean srActive;
    @Tunable(title = "HotPixels detect threshold", category = "Merge", description = "Higher multiplier detects less hotpixels", min = 0.5f, max = 5.0f, step = 0.1f, defaultValue = 1.5f)
    double detectThr;

    @Tunable(title = "Enable Adaptive Noise Model", category = "Merge", description = "Creates noise multiplier based on stdev", min = 0, max = 1, step = 1, defaultValue = 1)
    boolean enableAdaptiveNoise;

    @Tunable(title = "Alignment start level", category = "Alignment", description = "Finest resolution fed to the block pyramid matcher: 0 = raw/2 (default), 1 = raw/4, 2 = raw/8. Each skipped level removes the most expensive matching pass (~4x faster alignment core per level) at the cost of a coarser alignment vector field", min = 0, max = 2, step = 1, defaultValue = 0)
    int alignmentStartLevel;
    @Tunable(
            title = "Aligner",
            description = "GL - GL block pyramid, FlowNet - network optical flow, Halide - Halide CPU",
            category = "Alignment",
            entries = {"Halide CPU neon", "PyramidAlign GPU", "FlowNet optical flow", "Off"},
            entryValues = {"halide", "gl", "flownet", "off"},
            defaultValue = 0
    )
    String alignerSelect = "halide";

    @Tunable(title = "Aligner debug compare", category = "Alignment", description = "1 = when the Halide aligner runs, also run the GL block pyramid and log per-frame vector differences (diagnoses constant biases/shift conventions)", min = 0, max = 1, step = 1, defaultValue = 0)
    int alignDebugCompare;

    @Tunable(title = "Enable Adaptive Noise Storage", category = "Merge", description = "Persist fitted noise model into the dynamic multisample store", min = 0, max = 1, step = 1, defaultValue = 1)
    boolean enableNoiseStore;

    @Tunable(title = "Network merge noise multiplier", category = "Merge", description = "Scales the noise model fed to the kernel network", min = 0.1f, max = 20.0f, step = 0.05f, defaultValue = 1.0f)
    float noiseMpy;

    @Tunable(title = "Noise blend max frames", category = "Merge", description = "Frames combined into the deliberately misaligned progressive Gaussian blend used for noise estimation (blurs scene detail while noise only drops by a known factor)", min = 1, max = 9, step = 1, defaultValue = 9)
    int noiseBlendMaxFrames;

    @Tunable(title = "Noise blend calibration", category = "Merge", description = "Trim multiplier on the Monte-Carlo noise blend calibration table (1.0 = table value)", min = 0.5f, max = 2.0f, step = 0.05f, defaultValue = 1.0f)
    float noiseBlendCalMpy;

    @Tunable(title = "Noise scan subsample", category = "Merge", description = "Stride between texels evaluated by the noise histogram; the cheap difference operator supports a dense stride (was fixed 3 in the median-chain era)", min = 1, max = 8, step = 1, defaultValue = 3)
    int noiseScanSubsample;

    @Tunable(title = "Noise fit variance bins", category = "Merge", description = "Per-brightness-row cutoff on occupied variance bins kept by the noise fit pass 1 (lower rejects texture harder but undershoots on texture-free scenes; was fixed 45)", min = 8, max = 45, step = 1, defaultValue = 45)
    int noiseFitVarBins;

    @Tunable(title = "Noise fit gate", category = "Merge", description = "Adaptive per-brightness gate: pass 2 keeps only histogram bins whose implied variance is within this multiple of the pass-1 fitted noise (the per-brightness lower part; rejects texture and saturated bins). 0 disables", min = 0.0f, max = 5.0f, step = 0.25f, defaultValue = 2.0f)
    float noiseFitGateMpy;

    @Tunable(title = "Read noise floor multiplier", category = "Merge", description = "Multiplier on the analytic OPlace read-noise floor applied to fitted O; the legacy 3.0 compensated texture leakage that the noise blend now removes", min = 0.5f, max = 4.0f, step = 0.25f, defaultValue = 1.0f)
    float noiseOFloorMpy;

    @Tunable(title = "Fit O correction", category = "Merge", description = "Legacy fitO += 3/8*fitS^2 correction that compensated the under-rescaled fit; keep off with the calibrated blend", min = 0, max = 1, step = 1, defaultValue = 0)
    boolean enableFitOCorrection;

    @Tunable(title = "Adaptive fallback min", category = "Merge", description = "Lower clamp of the fallback adaptive multiplier (was 1.0, up-only)", min = 0.25f, max = 2.0f, step = 0.25f, defaultValue = 0.5f)
    float adaptiveFallbackMin;

    @Tunable(title = "Adaptive fallback max", category = "Merge", description = "Upper clamp of the fallback adaptive multiplier (was 4.0)", min = 1.0f, max = 4.0f, step = 0.25f, defaultValue = 2.0f)
    float adaptiveFallbackMax;

    @Tunable(title = "GPU sync profiling", category = "Merge", description = "Drain the GPU queue after every merge-loop stage so Stage[merge:*] logs measure true per-stage time (submit + GPU execution) instead of submit-only. Measurement only: serializes the pipeline and inflates the shot time", min = 0, max = 1, step = 1, defaultValue = 0)
    boolean profileGpuSync;

    /** Progressive noise-blend grid, must match BLEND_GRID in
     * tools/noise-blend-calibration/mc.py: center first, then edges, then
     * corners, so the first f slots give the temporal kernel shape for f
     * frames (9 -> full 3x3 Gaussian, 5 -> plus, 2 -> two-tap, 1 -> identity). */
    private static final int[][] BLEND_GRID = {
            {0, 0}, {1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    /** End-to-end calibration for the luma difference operator
     * |quad luma - kernel mean| (fixed full 3x3 Gaussian, sigma_g = 1) on
     * the progressive temporal blend, folded for the two-pass fit with the
     * default gate (noiseFitGateMpy = 2.0): E[noisehist "var"] =
     * VAR_STAT[f-1] * sigma for white per-frame noise through the blend,
     * the luma operator, the histogram binning and the gated weighted fit
     * (frame count f = 1..9). Measured by
     * tools/noise-blend-calibration/fixed_mc.py; trim with the
     * noiseBlendCalMpy tunable if device measurements disagree. */
    private static final float[] NOISE_BLEND_VAR_STAT = {
            0.23862f, 0.17435f, 0.14193f, 0.12277f, 0.10907f, 0.10082f, 0.09468f, 0.08916f, 0.08433f};
    /** Variance-axis anchor: bin 63 maps to sigma = SIGMA_REF for every frame
     * count (varScale = 63 / (VAR_STAT[f-1] * SIGMA_REF)), keeping bin
     * resolution in sigma terms constant and 2.4x-31x finer than the old
     * fixed 384 scale. */
    private static final float NOISE_BLEND_SIGMA_REF = 0.12f;

    /**
     * Builds the noise-estimation input: up to {@code noiseBlendMaxFrames}
     * frames (spaced across the burst) each sampled at its own slot of the
     * progressive 3x3 grid (one packed texel = one 2x2 Bayer quad = 2 raw px,
     * CFA-periodic so channels stay aligned) with normalized Gaussian
     * weights (sigma_g = 1 texel). Scene detail is correlated across frames,
     * so the fixed offsets convolve it with the kernel (fine texture
     * suppressed), while frame-independent noise only drops by the known
     * factor sum(w_i^2). noisehist.glsl then applies the luma difference
     * operator |quad luma - kernel mean| with a FIXED full 3x3 Gaussian
     * (sigma_g = 1, filled into spatialKernel, independent of frame count -
     * the temporal kernel is the only f-adaptive part): chroma structure
     * cancels exactly in the luma mean, the luma noise variance equals
     * S*b + O in quad-mean brightness for any white point, and the
     * symmetric kernel annihilates planes (gradients) precisely. Exposure
     * differences are
     * harmless: the conversion is linear, so every frame's variance obeys
     * the same variance = S*brightness + O in normalized units. Reuses
     * {@code alter} as the per-frame conversion target and ping-pongs
     * between {@code baseAlter} and one new texture (both unused until the
     * merge loop) - returns the accumulator, which may be either of the
     * two; close it only if it is not baseAlter.
     */
    private GLTexture buildNoiseBlendFrame(int tile, float[] spatialKernel) {
        int frameCnt = Math.min(Math.min(noiseBlendMaxFrames, BLEND_GRID.length), images.size());
        double[] weights = new double[frameCnt];
        double wSum = 0;
        for (int k = 0; k < frameCnt; k++) {
            weights[k] = Math.exp(-(BLEND_GRID[k][0] * BLEND_GRID[k][0]
                    + BLEND_GRID[k][1] * BLEND_GRID[k][1]) / 2.0);
            wSum += weights[k];
        }
        java.util.Arrays.fill(spatialKernel, 0.0f);
        double opSum = 0;
        double[] opW = new double[BLEND_GRID.length];
        for (int k = 0; k < BLEND_GRID.length; k++) {
            opW[k] = Math.exp(-(BLEND_GRID[k][0] * BLEND_GRID[k][0]
                    + BLEND_GRID[k][1] * BLEND_GRID[k][1]) / 2.0);
            opSum += opW[k];
        }
        for (int k = 0; k < frameCnt; k++) {
            weights[k] /= wSum;
        }
        for (int k = 0; k < BLEND_GRID.length; k++) {
            int dx = BLEND_GRID[k][0], dy = BLEND_GRID[k][1];
            spatialKernel[(dy + 1) * 3 + (dx + 1)] = (float) (opW[k] / opSum);
        }

        // P2 (H9): borrow baseDiff as the blend accumulator instead of a
        // dedicated texture (-24 MB). Safe: noiseblend uses imageLoad/Store
        // only (filter-agnostic), geometry is identically packedSize, and
        // baseDiff is dead until the merge loop fully overwrites it. Owned by
        // the ESD4D lifecycle - never closed here (see guards below).
        GLTexture blendAcc = baseDiff;
        GLTexture tempFloat = alter;
        // Two raw staging textures ping-ponged with the same GLsync scheme as
        // the merge loop: the upload for frame k+1 must not wait on merge00 of
        // frame k still reading the single staging texture.
        GLTexture tempRawA = frameCnt > 1
                ? new GLTexture(parameters.rawSize, new GLFormat(GLFormat.DataType.FLOAT_16, 1), null, GL_NEAREST, GL_CLAMP_TO_EDGE)
                : null;
        GLTexture tempRawB = frameCnt > 1
                ? new GLTexture(parameters.rawSize, new GLFormat(GLFormat.DataType.FLOAT_16, 1), null, GL_NEAREST, GL_CLAMP_TO_EDGE)
                : null;
        int rawSlot = 0;
        GLTexture blendCurrent = baseAlter;
        GLTexture blendNext = blendAcc;
        for (int k = 0; k < frameCnt; k++) {
            int idx = frameCnt == 1 ? 0
                    : (int) Math.round((double) k * (images.size() - 1) / (frameCnt - 1));
            GLTexture tempRaw = rawSlot == 0 ? tempRawA : tempRawB;
            GLTexture rawSrc = (idx == 0) ? inputBase : tempRaw;
            if (idx > 0) {
                waitUploadFence(noiseBlendUploadFences, rawSlot);
                tempRaw.loadRawHalf(images.get(idx).buffer);
            }

            // Convert raw Bayer -> normalized rgba16f vec4 (one texel per 2x2 quad)
            glProg.setLayout(tile, tile, 1);
            glProg.useAssetProgram("merge/merge00", true);
            glProg.setVar("exposure", 1.0f / images.get(0).pair.layerMpy);
            glProg.setVar("createDiff", 0);
            glProg.setVar("cfaShift", cfaShift);
            glProg.setTexture("inTexture", rawSrc);
            glProg.setTextureCompute("outTexture", tempFloat, true);
            glProg.computeAuto(packedSize, 1);
            if (idx > 0) markUploadFence(noiseBlendUploadFences, rawSlot);
            rawSlot ^= 1;

            // Progressive temporal blend accumulate at this frame's grid slot
            glProg.setLayout(tile, tile, 1);
            glProg.useAssetProgram("merge/noiseblend", true);
            glProg.setTextureCompute("currentTexture", blendCurrent, false);
            glProg.setTextureCompute("newTexture", tempFloat, false);
            glProg.setTextureCompute("outTexture", blendNext, true);
            glProg.setVar("weight", (float) weights[k]);
            glProg.setVar("offset", BLEND_GRID[k][0], BLEND_GRID[k][1]);
            glProg.setVar("firstPass", k == 0 ? 1 : 0);
            glProg.computeAuto(packedSize, 1);

            GLTexture swap = blendCurrent;
            blendCurrent = blendNext;
            blendNext = swap;
        }
        // Borrowed baseDiff must survive (see above); nothing to free here.
        if (tempRawA != null) tempRawA.close();
        if (tempRawB != null) tempRawB.close();
        deleteUploadFences(noiseBlendUploadFences);
        Log.d(Name, "Noise blend: " + frameCnt + " frame(s), sum(w^2)="
                + String.format(java.util.Locale.ROOT, "%.4f", java.util.stream.DoubleStream.of(weights).map(w -> w * w).sum()));
        return blendCurrent;
    }

    /**
     * Profiling aid: drains the command queue so the enclosing Stage[merge:*]
     * timer includes GPU execution, not just submit time (dispatches are
     * queued asynchronously; without this the loop's timers read ~0-2 ms
     * while the real cost surfaces later in a single drain). Off by default -
     * it serializes the pipeline and inflates the shot time.
     */
    private void gpuSyncProfile() {
        if (profileGpuSync) android.opengl.GLES30.glFinish();
    }

    /**
     * Blocks until the merge00 pass that last read this upload-ring slot has
     * completed on the GPU, so the following {@code glTexSubImage2D} can
     * overwrite it without the driver inserting its own full-queue
     * write-after-read stall. A timeout is non-fatal: driver command ordering
     * still guarantees correctness, we only lose the overlap.
     */
    private void waitUploadFence(long[] fences, int slot) {
        long fence = fences[slot];
        if (fence == 0) return;
        fences[slot] = 0;
        android.opengl.GLES30.glClientWaitSync(fence,
                android.opengl.GLES30.GL_SYNC_FLUSH_COMMANDS_BIT, 100_000_000L);
        android.opengl.GLES30.glDeleteSync(fence);
    }

    /**
     * Joins the Halide alignment worker if one is running, propagating any
     * failure it captured. Idempotent: the reference is cleared first, so the
     * {@link #close()} safety net after the normal join is a no-op.
     */
    private void joinHalideWorker() {
        Thread t = halideThread;
        halideThread = null;
        if (t == null) return;
        boolean interrupted = false;
        while (t.isAlive()) {
            try {
                t.join();
            } catch (InterruptedException e) {
                interrupted = true;
            }
        }
        if (interrupted) Thread.currentThread().interrupt();
        Throwable err = halideError.getAndSet(null);
        if (err != null) {
            if (err instanceof RuntimeException) throw (RuntimeException) err;
            throw new RuntimeException("Halide alignment failed", err);
        }
    }

    /** Records the completion point of the given slot's last reader. */
    private void markUploadFence(long[] fences, int slot) {
        if (fences[slot] != 0) {
            android.opengl.GLES30.glDeleteSync(fences[slot]);
        }
        fences[slot] = android.opengl.GLES30.glFenceSync(
                android.opengl.GLES30.GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    }

    private void deleteUploadFences(long[] fences) {
        for (int i = 0; i < fences.length; i++) {
            if (fences[i] != 0) {
                android.opengl.GLES30.glDeleteSync(fences[i]);
                fences[i] = 0;
            }
        }
    }

    /**
     * Builds the programs that the merge loop first uses (mergeAlign and
     * mergeCombineWeight1) with a 1x1 scratch dispatch while the GPU is still
     * busy with the noise passes. Adreno defers pipeline-state creation to the
     * first draw, which showed up as a ~100 ms spike on the first combine; the
     * dummy dispatch moves that setup off the critical first frame. The
     * scratch/dummy textures are never read and the real passes re-issue every
     * uniform/texture, so results are unchanged.
     */
    private void prewarmMergePrograms(int tile) {
        if (Objects.equals(alignerSelect, "flownet")) return; // different mergeAlignFlow source
        GLTexture scratch = null, dummyKernels = null, dummyAlign = null;
        try {
            scratch = new GLTexture(new Point(1, 1), new GLFormat(GLFormat.DataType.FLOAT_16, 4), null, GL_NEAREST, GL_CLAMP_TO_EDGE);
            dummyKernels = new GLTexture(new Point(1, 1), new GLFormat(GLFormat.DataType.FLOAT_16, 4), null, GL_NEAREST, GL_CLAMP_TO_EDGE);
            dummyAlign = new GLTexture(new Point(1, 1), new GLFormat(GLFormat.DataType.FLOAT_16, 4), null, GL_NEAREST, GL_CLAMP_TO_EDGE);

            // Exact define order of the loop's mergeAlign pass so the compiled
            // source (and thus the program-cache key) matches.
            glProg.setDefine("TILE_AL", parameters.tile);
            glProg.setLayout(tile, tile, 1);
            glProg.useAssetProgram("merge/mergeAlign", true);
            glProg.setTexture("inTexture", inputBase);
            glProg.setTexture("alignmentTexture", dummyAlign);
            glProg.setTextureCompute("baseTexture", base, false);
            glProg.setTextureCompute("alterTexture", alter, false);
            glProg.setTextureCompute("outTexture", scratch, true);
            glProg.computeAuto(new Point(1, 1), 1);

            // Combine pass: same define sequence (just LAYOUT).
            glProg.setLayout(tile, tile, 1);
            glProg.useAssetProgram("merge/mergeCombineWeight1", true);
            glProg.setTexture("inTex", inputBase);
            glProg.setTexture("kernelsMap", dummyKernels);
            glProg.setTextureCompute("inTexture", base, false);
            glProg.setTextureCompute("diffTexture", baseDiff, false);
            glProg.setTextureCompute("outTexture", scratch, true);
            glProg.computeAuto(new Point(1, 1), 1);
        } catch (Throwable t) {
            Log.w("ESD4D", "merge program pre-warm failed (non-fatal)", t);
        } finally {
            // Never leave a half-consumed define list for the aligner programs.
            glProg.clearDefines();
            if (scratch != null) scratch.close();
            if (dummyKernels != null) dummyKernels.close();
            if (dummyAlign != null) dummyAlign.close();
        }
    }

    /** Upper bound on workers for the parallel raw->fp16 conversion: leaves
     * headroom for render/capture threads and bounds transient staging. */
    private static final int F16_CONVERT_MAX_WORKERS = 4;

    private void convertFramesToF16() {
        int pending = 0;
        for (int i = 0; i < images.size(); i++) {
            ImageFrame frame = images.get(i);
            if (!frame.fp16 && frame.buffer != null) pending++;
        }
        if (pending == 0) return;
        int workers = Math.min(Math.min(F16_CONVERT_MAX_WORKERS, pending),
                Math.max(1, Runtime.getRuntime().availableProcessors() - 2));
        if (workers <= 1) {
            for (int i = 0; i < images.size(); i++) convertFrameToF16(images.get(i), i);
            return;
        }
        ExecutorService pool = Executors.newFixedThreadPool(workers, r -> new Thread(r, "F16Convert"));
        try {
            ArrayList<Callable<Void>> tasks = new ArrayList<>(pending);
            for (int i = 0; i < images.size(); i++) {
                final int idx = i;
                final ImageFrame frame = images.get(i);
                if (frame.fp16 || frame.buffer == null) continue;
                tasks.add(() -> { convertFrameToF16(frame, idx); return null; });
            }
            for (Future<Void> f : pool.invokeAll(tasks)) f.get();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("f16-convert interrupted", e);
        } catch (ExecutionException e) {
            Throwable cause = e.getCause();
            if (cause instanceof RuntimeException) throw (RuntimeException) cause;
            throw new RuntimeException("f16-convert failed", cause);
        } finally {
            pool.shutdown();
        }
    }

    /**
     * Converts one frame's buffer to normalized fp16 in place. Safe to call
     * from any thread: {@link ImageFrame#upload()}'s staging cache is
     * synchronized (concurrent callers each get an owned buffer) and the
     * native converters keep no global state.
     */
    private void convertFrameToF16(ImageFrame frame, int index) {
        if (frame.fp16 || frame.buffer == null) return;
        ByteBuffer normalized;
        if (frame.packedBits > 0) {
            // Frame arrived as a packed bitstream (burst-memory saving).
            // The common 10-bit depth decodes straight into normalized fp16,
            // skipping the uint16 staging buffer and its memory round trip;
            // other depths keep the unpack-then-normalize path.
            if (frame.packedBits == 10) {
                normalized = Allocator.allocate(parameters.rawSize.x * parameters.rawSize.y * 2);
                if (normalized != null && !Allocator.unpackNormalizeF16TenBit(normalized,
                        frame.buffer, parameters.rawSize.x, parameters.rawSize.y,
                        parameters.whiteLevel, parameters.blackLevel)) {
                    Allocator.free(normalized);
                    normalized = null;
                }
                if (normalized == null) {
                    try (ImageFrame.Upload up = frame.upload()) {
                        normalized = Allocator.createF16(up.buffer,
                                parameters.rawSize.x, parameters.rawSize.y,
                                parameters.whiteLevel, parameters.blackLevel);
                    }
                }
            } else {
                // Unpack to uint16 first — createF16 reads raw sample counts.
                try (ImageFrame.Upload up = frame.upload()) {
                    normalized = Allocator.createF16(up.buffer,
                            parameters.rawSize.x, parameters.rawSize.y,
                            parameters.whiteLevel, parameters.blackLevel);
                }
            }
        } else {
            normalized = Allocator.createF16(frame.buffer,
                    parameters.rawSize.x, parameters.rawSize.y,
                    parameters.whiteLevel, parameters.blackLevel);
        }
        if (normalized == null) {
            throw new IllegalStateException("createF16 failed for frame " + index);
        }
        Allocator.free(frame.buffer);
        frame.buffer = normalized;
        frame.fp16 = true;
        frame.packedBits = 0;
    }

    @Override
    public void Run() {
        com.particlesdevs.photoncamera.settings.TunableInjector.inject(this);
        if (profileGpuSync)
            Log.d("ESD4D", "GPU sync profiling ON: Stage[merge:*] times now include GPU execution");
        Log.d("ESD4D", "Noise multiplier: " + noiseMpy);
            //Log.d("ESD4D", "Optical flow refinement: " + enableFlowRefinement + " maxShift: " + flowRefineMaxDisp);
        glUtils = new GLUtils(glOne.glProcessing);

        // Convert every frame once into white/black-level normalized fp16
        // (RawF16, NEON). The swap is size-neutral (uint16 -> half float), the
        // original raw is freed, and every later upload (merge00 / alignment
        // normalize / flowRGB inputs) feeds FLOAT_16 textures directly - the
        // shaders receive already-normalized floats and skip the
        // whitelevel/blackLevel math. Frames are independent and the native
        // converters keep no global state, so the burst is converted on a
        // small worker pool (bounded to keep transient staging memory tame).
        long f16T = System.currentTimeMillis();
        convertFramesToF16();
        Log.d("ESD4D", "Stage[f16-convert] elapsed:" + (System.currentTimeMillis() - f16T) + " ms");

        // The Halide aligner needs only the normalized frames and parameters,
        // and its kernels are CPU-only, so launch it here: its ~190 ms at
        // 12.6 MP then overlaps the GPU noise-blend / histogram / brightmap
        // passes instead of running after them with the GPU idle. The Result
        // texture is still created on the GL thread at the aligner branch.
        Point alignmentOutputSize = new Point(parameters.alignmentSize.x * parameters.tilesX,
                parameters.alignmentSize.y * ((images.size()-1)/parameters.tilesX + 1));
        Log.d("Alignment", "alignment pipeline size: " + alignmentOutputSize.x + " " + alignmentOutputSize.y);
        if (Objects.equals(alignerSelect, "halide")) {
            halideAlignment = new HalideAlignment(alignmentOutputSize, images);
            halideAlignment.parameters = parameters;
            halideLaunchMs = System.currentTimeMillis();
            halideThread = new Thread(() -> {
                try {
                    halideAlignment.RunCPU();
                } catch (Throwable t) {
                    halideError.compareAndSet(null, t);
                    Log.e("ESD4D", "Halide alignment worker failed", t);
                }
            }, "HalideAlignment");
            halideThread.start();
        }

        float minExp = 1.f;
        int minExpIdx = 0;
        int lowCnt = 0;
        for (int i = 1; i < images.size(); i++) {
            ImageFrame frame = images.get(i);
            float exposure = 1.f/frame.pair.layerMpy;
            Log.d("ESD4D", "exposure: " + exposure);
            if(exposure < 0.95f) {
                lowCnt++;
            }
            if(exposure < minExp) {
                minExpIdx = i;
                minExp = exposure;
            }
        }

        if (parameters.tile != 16) {
            // Custom tile sizes (set upstream) keep their own alignmentSize.
            Log.d("ESD4D", "Alignment tile size: " + parameters.tile
                    + " alignmentSize: " + parameters.alignmentSize.x + "x" + parameters.alignmentSize.y);
        }
        long texT = System.currentTimeMillis();
        Point raw = parameters.rawSize;
        Point rawHalf = new Point(parameters.rawSize.x/2,parameters.rawSize.y/2);
        // merge00 green-normalizes all packed quads for any CFA: the quincunx
        // sub-texel sampler needs the two greens on the anti-diagonal g/b
        // slots. Only GRBG/GBRG carry their greens on the main diagonal - for
        // those, quad origins are shifted back by the red-site offset and the
        // packed grid grows by it. RGGB/BGGR already have greens on the
        // anti-diagonal (R/B merely sit swapped for BGGR, which every merge
        // stage treats channel-agnostically), so they get no shift, no filler
        // and an unchanged grid. Real raw site X lives at packed rel = X +
        // cfaShift; shifted out-of-range sites are edge duplicates, never
        // read back on unpack.
        int cfa = (int) parameters.cfaPattern;
        if (cfa < 0 || cfa > 3) cfa = 0; // quad/monochrome modes: no normalization
        cfaShift = (cfa == 1 || cfa == 2) ? new Point(cfa % 2, cfa / 2) : new Point(0, 0);
        packedSize = new Point(rawHalf.x + cfaShift.x, rawHalf.y + cfaShift.y);
        result = new GLTexture(raw,new GLFormat(GLFormat.DataType.FLOAT_16,1), null, GL_NEAREST, GL_CLAMP_TO_EDGE);
        inputBase = new GLTexture(parameters.rawSize, new GLFormat(GLFormat.DataType.FLOAT_16,1), null, GL_NEAREST, GL_CLAMP_TO_EDGE);
        inputBase.loadRawHalf(images.get(0).buffer);
        // Pyramid diff
        baseDiff = new GLTexture(packedSize,new GLFormat(GLFormat.DataType.FLOAT_16,4),null,GL_LINEAR,GL_CLAMP_TO_EDGE);
        // Temporal result
        base = new GLTexture(packedSize,new GLFormat(GLFormat.DataType.FLOAT_16,4),null,GL_LINEAR,GL_CLAMP_TO_EDGE);
        baseAlter = new GLTexture(packedSize,new GLFormat(GLFormat.DataType.FLOAT_16,4),null,GL_LINEAR,GL_CLAMP_TO_EDGE);
        alter = new GLTexture(packedSize,new GLFormat(GLFormat.DataType.FLOAT_16,4),null,GL_LINEAR,GL_CLAMP_TO_EDGE);
        // Pack 4 horizontal luma samples per rgba16f texel (r16f image formats are
        // rejected by some drivers) -> texture is 4x smaller in x.
        Point brightMapSize = new Point((packedSize.x + 3) / 4, packedSize.y);
        brightMap = new GLTexture(brightMapSize,new GLFormat(GLFormat.DataType.FLOAT_16,4));
        brightMapCPUSize = new Point(brightMapSize.x * 4, brightMapSize.y);
        float[] blackLevel = parameters.blackLevel;
        //float[] blackLevel = new float[]{parameters.blackLevel[0]*0.5f, parameters.blackLevel[1]*0.5f, parameters.blackLevel[2]*0.5f, parameters.blackLevel[3]*0.5f};
        //float bl = Math.max(Math.max(parameters.blackLevel[0], parameters.blackLevel[1]), Math.max(parameters.blackLevel[2], parameters.blackLevel[3]));
        // Black/white-level normalization now happens on the CPU (createF16),
        // so the shaders receive site-normalized fp16 directly; only the
        // max-black estimate for mergeAlign's minLevel noise floor remains.
        glOne.glProgram.setDefine("RAWSIZE",parameters.rawSize);
        glOne.glProgram.setDefine("CFAPATTERN",(int)parameters.cfaPattern);

        float[] analogBalance = new float[4];
        switch (parameters.cfaPattern){
            case 0: // RGGB
                analogBalance[0] = 1.0f/parameters.whitePoint[0];
                analogBalance[1] = 1.0f/parameters.whitePoint[1];
                analogBalance[2] = 1.0f/parameters.whitePoint[1];
                analogBalance[3] = 1.0f/parameters.whitePoint[2];
                break;
            case 1: // GRBG
                analogBalance[0] = 1.0f/parameters.whitePoint[1];
                analogBalance[1] = 1.0f/parameters.whitePoint[0];
                analogBalance[2] = 1.0f/parameters.whitePoint[2];
                analogBalance[3] = 1.0f/parameters.whitePoint[1];
                break;
            case 2: // GBRG
                analogBalance[0] = 1.0f/parameters.whitePoint[1];
                analogBalance[1] = 1.0f/parameters.whitePoint[2];
                analogBalance[2] = 1.0f/parameters.whitePoint[0];
                analogBalance[3] = 1.0f/parameters.whitePoint[1];
                break;
            case 3: // BGGR
                analogBalance[0] = 1.0f/parameters.whitePoint[2];
                analogBalance[1] = 1.0f/parameters.whitePoint[1];
                analogBalance[2] = 1.0f/parameters.whitePoint[1];
                analogBalance[3] = 1.0f/parameters.whitePoint[0];
                break;
        }
        NoiseModeler modeler = parameters.noiseModeler;
        noiseS = modeler.baseModel[0].first.floatValue() +
                modeler.baseModel[1].first.floatValue() +
                modeler.baseModel[2].first.floatValue();
        noiseO = modeler.baseModel[0].second.floatValue() +
                modeler.baseModel[1].second.floatValue() +
                modeler.baseModel[2].second.floatValue();
        noiseS /= 3.f;
        noiseO /= 3.f;
        //GLUtils glUtils = new GLUtils(glOne.glProcessing);
        int tile = 8;
        glProg.setLayout(tile,tile,1);
        glProg.useAssetProgram("merge/merge00",true);
        glProg.setVar("exposure", 1.f/images.get(0).pair.layerMpy);
        glProg.setVar("createDiff", 0);
        glProg.setVar("cfaShift", cfaShift);
        glProg.setVar("analogBalance", analogBalance);
        glProg.setVar("randF", (float)Math.random(), (float)Math.random());
        // Test value if enabled in shader
        //glProg.setVar("noiseS", 0.0013796629f);
        //glProg.setVar("noiseO", 8.3751265E-6f);
        //glProg.setVar("noiseS", 0.05f);
        //glProg.setVar("noiseO", 0.0f);
        glProg.setTexture("inTexture",inputBase);
        glProg.setTextureCompute("outTexture",base, true);
        glProg.computeAuto(new Point(base.mSize.x, base.mSize.y), 1);
        Log.d("ESD4D", "Stage[textures+base] elapsed:" + (System.currentTimeMillis() - texT) + " ms");
        //glUtils.convertVec4(base, "vec4(0.5)", base);
        //var buff = glUtils.GenerateGLImage(base.mSize, 4);
        //Log.d(Name, "Buffer first:" + buff.byteBuffer.get(0) + " " + buff.byteBuffer.get(1));
        //glUtils.Result(base.mSize, "noiseInput", buff.byteBuffer);

        double adaptiveNMpy = 1.0;
        if (enableAdaptiveNoise) {
            // 2D histogram: (brightness_bin * NUM_VARIANCE_BINS + variance_bin) -> count
            // Model: variance = NoiseS * brightness + NoiseO  =>  sigma = sqrt(NoiseS*b + NoiseO)
            final int numBrightnessBins = 64;
            final int numVarianceBins = 64;
            final int noiseScanBins = numBrightnessBins * numVarianceBins; // 4096
            // Estimation input: progressive misaligned Gaussian blend of the
            // burst (see buildNoiseBlendFrame). The blend blurs scene detail
            // - the root cause of the old overestimation - while noise only
            // drops by the calibrated factor varStat below.
            int blendFrames = Math.min(Math.min(noiseBlendMaxFrames, BLEND_GRID.length), images.size());
            final float varStat = NOISE_BLEND_VAR_STAT[blendFrames - 1] * noiseBlendCalMpy;
            // Variance axis anchored so bin 63 = SIGMA_REF for every frame
            // count (the old fixed 64*6 scale wasted most of the range at
            // typical noise levels, quantizing low-ISO fits into 1-3 bins).
            final float varianceScale = (numVarianceBins - 1) / (varStat * NOISE_BLEND_SIGMA_REF);
            final float brightnessScale = 64.0f * (float)Math.sqrt(3.0f);
            float[] spatialKernel = new float[9];
            long blendT = System.currentTimeMillis();
            GLTexture noiseInput = buildNoiseBlendFrame(8, spatialKernel);
            Log.d("ESD4D", "Stage[noiseblend] elapsed:" + (System.currentTimeMillis() - blendT) + " ms");
            GLHistogram noiseHist = new GLHistogram(glProg, noiseScanBins);
            noiseHist.Custom = true;
            noiseHist.Rc = true;
            noiseHist.Gc = false;
            noiseHist.Bc = false;
            noiseHist.Ac = false;
            noiseHist.exposure[0] = 1.0f;
            noiseHist.exposure[1] = 1.0f;
            noiseHist.exposure[2] = 1.0f;
            noiseHist.exposure[3] = 1.0f;
            noiseHist.CustomShader = "merge/noisehist";
            noiseHist.input1 = brightnessScale;
            noiseHist.input2 = varianceScale;
            noiseHist.resize = noiseScanSubsample;
            noiseHist.customKernel = spatialKernel;
            long histT = System.currentTimeMillis();
            int[][] noiseRes = noiseHist.Compute(noiseInput);
            // The accumulator may be the borrowed baseDiff (odd frame counts):
            // it must survive for the merge loop, which fully overwrites it.
            if (noiseInput != baseAlter && noiseInput != baseDiff) noiseInput.close();
            noiseHist.close();
            Log.d("ESD4D", "Stage[noise-histogram] elapsed:" + (System.currentTimeMillis() - histT) + " ms");
            int[] hist = noiseRes[0];
            long fitT = System.currentTimeMillis();
            // Weighted linear regression: variance = NoiseS * brightness + NoiseO,
            // run in two passes. Pass 1 fits all bins kept by the per-row
            // filter; pass 2 (adaptive gate, noiseFitGateMpy) keeps only bins
            // whose implied variance is consistent with the pass-1 noise
            // model - the per-brightness "lower part" that rejects texture
            // and saturation-capped bins without a per-threshold
            // calibration (see tools/noise-blend-calibration pct/gate runs).
            double sumW = 0, sumWb = 0, sumWv = 0, sumWb2 = 0, sumWbv = 0;
            int points = 0;
            int varCnt = 0;
            for (int i = 0; i < noiseScanBins; i++) {
                int count = hist[i];
                var bin = i / numVarianceBins;
                var vin = i % numVarianceBins;
                if(vin == 0) {
                    varCnt = 0;
                }
                if (count <= 0 || bin == numBrightnessBins-1 || (varCnt >= 30 && vin == 63) || varCnt > noiseFitVarBins) continue;
                varCnt++;
                // Fit in the absolute quad-mean brightness domain the
                // consumers use. (The old (b-minBr)/(1-minBr) rescale fit in
                // scene-relative brightness and inflated S by 1/(1-minBr) on
                // any scene without near-black content - snow, sky, low-key.)
                double brightness = ((double)(bin) + 0.5) / ((double)brightnessScale);
                brightness = Math.pow(brightness, 2.0);
                double variance = (vin + 0.5) / varianceScale;
                // The shader's "var" statistic (|center - kernel mean| after
                // the temporal blend) is calibrated end-to-end - see
                // NOISE_BLEND_VAR_STAT and tools/noise-blend-calibration -
                // to varStat * sigma, so squaring and dividing by varStat^2
                // recovers the per-frame variance.
                variance = variance * variance / ((double) varStat * varStat);
                double w = count * 1.0f;
                sumW += w;
                sumWb += w * brightness;
                sumWv += w * variance;
                sumWb2 += w * brightness * brightness;
                sumWbv += w * brightness * variance;
                points++;
            }
            //points = 9;
            if (points >= 1) {
                double denom = sumW * sumWb2 - sumWb * sumWb;
                if (denom > 1e-20) {
                    double passS = (sumW * sumWbv - sumWb * sumWv) / denom;
                    double passO = (sumWv - passS * sumWb) / sumW;
                    double fitS = passS;
                    double fitO = passO;
                    if (noiseFitGateMpy > 0.0f) {
                        // Pass 2: keep only bins whose implied variance is
                        // within the gate multiple of the pass-1 model.
                        double gW = 0, gWb = 0, gWv = 0, gWb2 = 0, gWbv = 0;
                        int gPoints = 0;
                        varCnt = 0;
                        for (int i = 0; i < noiseScanBins; i++) {
                            int count = hist[i];
                            var bin = i / numVarianceBins;
                            var vin = i % numVarianceBins;
                            if(vin == 0) {
                                varCnt = 0;
                            }
                            if (count <= 0 || bin == numBrightnessBins-1 || (varCnt >= 30 && vin == 63) || varCnt > noiseFitVarBins) continue;
                            varCnt++;
                            double brightness = ((double)(bin) + 0.5) / ((double)brightnessScale);
                            brightness = Math.pow(brightness, 2.0);
                            double variance = (vin + 0.5) / varianceScale;
                            variance = variance * variance / ((double) varStat * varStat);
                            double gateVar = noiseFitGateMpy
                                    * (Math.max(passS, 1e-12) * brightness + Math.max(passO, 0.0));
                            if (variance > gateVar) continue;
                            double w = count * 1.0f;
                            gW += w;
                            gWb += w * brightness;
                            gWv += w * variance;
                            gWb2 += w * brightness * brightness;
                            gWbv += w * brightness * variance;
                            gPoints++;
                        }
                        double gDenom = gW * gWb2 - gWb * gWb;
                        if (gPoints >= 1 && gDenom > 1e-20) {
                            fitS = (gW * gWbv - gWb * gWv) / gDenom;
                            fitO = (gWv - fitS * gWb) / gW;
                            Log.d("DynamicNoise", "Gate pass: " + gPoints + " bins kept of " + points);
                        }
                    }
                    fitS = Math.max(fitS, 1e-10);
                    Log.d("DynamicNoise",  "Fit S:" + fitS + " O:" + fitO);
                    // Keep at least 5% of original read noise so we don't collapse to zero on noisy sensors
                    double minO = 0.05 * noiseO;
                    fitO = Math.max(fitO, minO);
                    // Read-noise floor: O=S/7 overstates read noise now that the variance
                    // estimator is unbiased (previously S carried a ~2.2x bias that made
                    // S/7 a sane proxy). S/20 keeps a guard against O collapsing while no
                    // longer dominating realistic sensors (O/S is typically < 0.05).
                    //fitO = Math.max(fitO, fitS/20);
                    //fitS = Math.max(fitS, parameters.noiseModeler.SPlace(parameters.iso));
                    //fitO = Math.max(fitO, parameters.noiseModeler.OPlace(parameters.iso) * noiseOFloorMpy);
                    // Commit the fitted S/O to the multisample noise map, then read
                    // back the blended (moving-average) value. Committing before
                    // reading makes the current estimation participate in the
                    // average, while the store's measurement-list guard skips
                    // duplicate scenes (same exposure/iso) to avoid bias. Using the
                    // blended output smooths per-capture estimator fluctuations.
                    double commitS = fitS;
                    double commitO = fitO;
                    DynamicNoiseStore.NoiseEstimate blended = null;
                    if (enableNoiseStore) {
                        blended = DynamicNoiseStore.dynamicNoiseStore.commitAndGet(
                                parameters.physicalID, parameters.iso,
                                parameters.noiseModeler.AnalogueISO,
                                commitS, commitO, parameters.exposureTime);
                    }
                    if (blended != null) {
                        fitS = blended.s;
                        fitO = blended.o;
                        // Re-apply floors defensively on the blended result.
                        //fitS = Math.max(fitS, parameters.noiseModeler.SPlace(parameters.iso));
                        //fitO = Math.max(fitO, parameters.noiseModeler.OPlace(parameters.iso));
                        Log.d("DynamicNoise", "Blended noise model from store: S=" + fitS
                                + " O=" + fitO + " for iso=" + parameters.iso);
                    }
                    // Legacy correction that compensated the old under-rescaled
                    // fit; off by default now that the blend is calibrated.
                    if (enableFitOCorrection) fitO += fitS*fitS * 3.0/8.0;
                    noiseS = (float) fitS;
                    noiseO = (float) fitO;
                    Log.d("DynamicNoise",  "Fitted noise model: NoiseS=" + noiseS + " NoiseO=" + noiseO + " Half=" + Math.sqrt(noiseS * 0.5 + noiseO) + " (points=" + points + ")");
                    parameters.noiseModeler.baseModel = new Pair[] {
                            new Pair<>((double) noiseS, (double) noiseO),
                            new Pair<>((double) noiseS, (double) noiseO),
                            new Pair<>((double) noiseS, (double) noiseO)};
                }
                adaptiveNMpy = 1.0;
            } else {
                // Fallback: scale original model to match observed at mid-gray (same as before)
                double modelSigmaMid = Math.sqrt(noiseS * 0.5 + noiseO);
                if (modelSigmaMid > 1e-10) {
                    double sumWeightedSigma = 0, sumWeightedCount = 0;
                    for (int i = 0; i < noiseScanBins; i++) {
                        int count = hist[i];
                        if (count <= 0) continue;
                        double sigma = ((i % numVarianceBins + 0.5) / varianceScale) / varStat;
                        sumWeightedSigma += sigma * count;
                        sumWeightedCount += count;
                    }
                    if (sumWeightedCount > 0) {
                        double observedSigma = sumWeightedSigma / sumWeightedCount;
                        adaptiveNMpy = observedSigma / modelSigmaMid;
                        adaptiveNMpy = Math2.clamp(adaptiveNMpy, adaptiveFallbackMin, adaptiveFallbackMax);
                    }
                }
                Log.d("DynamicNoise", "Adaptive Mpy (fallback): " + adaptiveNMpy + " (insufficient points=" + points + ")");
            }
            Log.d("ESD4D", "Stage[noise-fit] elapsed:" + (System.currentTimeMillis() - fitT) + " ms");
        }
        parameters.noiseModeler.setAdaptiveMpy(adaptiveNMpy);
        double noisempy = Math.pow(2.0, PhotonCamera.getSettings().mergeStrength);
        //double noiseMin = 1.0/(double)parameters.whiteLevel;
        double noiseMin = 1e-10;
        kernelSigma = (float) Math.sqrt(noiseS * 0.5 + noiseO);
        // Pre-inflation noise model for the optical-flow significance gate
        // (noiseS/noiseO below are merge-strength inflated).
        float rawNoiseS = noiseS;
        float rawNoiseO = noiseO;
        srBaseNoiseS = rawNoiseS;
        srBaseNoiseO = rawNoiseO;
        noiseS = (float)Math.max(noiseS * noisempy * adaptiveNMpy * adaptiveNMpy,noiseMin);
        noiseO = (float)Math.max(noiseO * noisempy * adaptiveNMpy * adaptiveNMpy,noiseMin);
        if(enableHotPixelCorrection) {
            long hotT = System.currentTimeMillis();
            hotPixels();
            Log.d("ESD4D", "Stage[hotpixels] elapsed:" + (System.currentTimeMillis() - hotT) + " ms");
        }

        long brightT = System.currentTimeMillis();
        glProg.setLayout(tile,tile,1);
        glProg.useAssetProgram("merge/mergeGrayscale",true);
        glProg.setVar("inSize", packedSize);
        glProg.setTextureCompute("inTexture",base, false);
        glProg.setTextureCompute("outTexture",brightMap, true);
        glProg.computeAuto(brightMap.mSize, 1);
        exportBrightMap();
        Log.d("ESD4D", "Stage[brightmap] elapsed:" + (System.currentTimeMillis() - brightT) + " ms");
        // GPU copy consumed (the CPU copy feeds inference from here on):
        // release now instead of AfterRun so it doesn't span alignment +
        // the merge loop. Nulled; AfterRun null-guards it.
        brightMap.close();
        brightMap = null;
        // KernelNet's input derives from the reference frame only, so its
        // inference is independent of the alignment/merge loop below. Run it
        // on a worker thread concurrently with alignment (merge00 / FlowNet /
        // mergeAlign) and collect it just before the first combine pass needs
        // kernelsMap. The inference (and the model load inside it) touches no
        // GL state; the kernelsMap build/upload must rejoin the GL thread.
        final float kernelSigmaArg = kernelSigma * noiseMpy;
        final AtomicReference<KernelNetResult> kernelNetResult = new AtomicReference<>();
        Thread kernelNetThread = new Thread(() -> {
            try {
                kernelNetResult.set(runKernelNetInference(kernelSigmaArg));
            } catch (Throwable t) {
                Log.e("ESD4D", "KernelNet worker failed", t);
            }
        }, "KernelNet-inference");
        kernelNetThread.start();

        // The KernelNet worker joins the already-running Halide worker (launched
        // right after f16-convert) so both overlap this GPU-side block; with a
        // GL/FlowNet/off aligner, alignment itself still runs below with the
        // inference concurrently. Nothing between the worker start and the
        // merge loop consumes alignmentTex, and this block only needs the
        // reference-frame inputs already prepared above.
        // The merge loop uploads each alter frame into inputAlter just before
        // FlowNet consumes it, so create it up-front and share it (plus the
        // already-uploaded inputBase) instead of letting FlowNet allocate and
        // re-upload duplicates. -48 MB VRAM, -8 uploads; bit-exact (same
        // texture object sampled identically).
        inputAlter = new GLTexture(parameters.rawSize, new GLFormat(GLFormat.DataType.FLOAT_16, 1), null, GL_NEAREST, GL_MIRRORED_REPEAT);
        // Second ring slot: see the merge loop. Sharing with FlowNet/Pyramid
        // stays on the first slot (alignment completes before the loop).
        inputAlterAlt = new GLTexture(parameters.rawSize, new GLFormat(GLFormat.DataType.FLOAT_16, 1), null, GL_NEAREST, GL_MIRRORED_REPEAT);
        // Pay the merge programs' first-use setup now, while the GPU is still
        // draining the noise passes and the CPU is otherwise waiting.
        prewarmMergePrograms(tile);
        // Aligner selector: 0 = GL pyramid (disables FlowNet), 1 = FlowNet,
        // 2 = Halide CPU. FlowNet keeps its init fallback to the pyramid.
        if (Objects.equals(alignerSelect, "flownet")) {
            FlowNetAlignment flowNetAlignmentTmp = new FlowNetAlignment(alignmentOutputSize, images, glProg, glUtils, this, minExpIdx);
            flowNetAlignmentTmp.parameters = parameters;
            flowNetAlignmentTmp.shareInputTextures(inputBase, inputAlter);
            long startTime = System.currentTimeMillis();
            boolean useNcnnFlow = flowNetAlignmentTmp.initFlow();
            Log.d("ESD4D", "FlowNet alignment init time: " + (System.currentTimeMillis() - startTime) + "ms");
            if (useNcnnFlow) {
                flowNetAlignment = flowNetAlignmentTmp;
                alignmentTex = flowNetAlignment.flowTex;
            } else {
                flowNetAlignmentTmp.close();
                alignerSelect = "off";
                Log.d("ESD4D", "FlowNet alignment disabled, using identity alignment");
            }
        }
        if (Objects.equals(alignerSelect, "halide")) {
            // CPU/NEON Halide path; identical Result atlas format, so the
            // merge below is unchanged. The CPU half has been running on a
            // worker since the f16-convert; collect it and upload the atlas on
            // this (GL) thread.
            long waitT = System.currentTimeMillis();
            try {
                joinHalideWorker();
                Log.d("ESD4D", "Halide alignment wait: " + (System.currentTimeMillis() - waitT)
                        + "ms total: " + (System.currentTimeMillis() - halideLaunchMs) + "ms");
                halideAlignment.uploadResult();
            } catch (RuntimeException e) {
                halideAlignment.close();
                halideAlignment = null;
                throw e;
            }
            alignmentTex = halideAlignment.Result;
            if (alignDebugCompare == 1) {
                logAlignerCompare(alignmentOutputSize, images);
            }
            halideAlignment.close();
            halideAlignment = null;
        } else if (Objects.equals(alignerSelect, "gl")) {
            PyramidAlignment pyramidAlignment = new PyramidAlignment(alignmentOutputSize, images, glProg, glUtils, this);
            pyramidAlignment.parameters = parameters;
            pyramidAlignment.startLevel = alignmentStartLevel;
            pyramidAlignment.shareInputTextures(inputBase, inputAlter);
            // P2 (H2): lend baseDiff as pyramid scratch. Valid only when the
            // packed grid matches rawHalf exactly (cfaShift==0); otherwise the
            // 1-px size mismatch would diverge border texels.
            if (cfaShift.x == 0 && cfaShift.y == 0) pyramidAlignment.borrowTempTexture(baseDiff);
            long startTime = System.currentTimeMillis();
            pyramidAlignment.Run();
            Log.d("ESD4D", "Alignment time: " + (System.currentTimeMillis() - startTime) + "ms");
            alignmentTex = pyramidAlignment.Result;
            pyramidAlignment.close();
        } else if (Objects.equals(alignerSelect, "off")) {
            alignmentTex = new GLTexture(alignmentOutputSize, new GLFormat(GLFormat.DataType.FLOAT_16, 4),
                    BufferUtils.getFrom(new float[alignmentOutputSize.x * alignmentOutputSize.y * 4]),
                    GL_NEAREST, GL_CLAMP_TO_EDGE);
            Log.d("ESD4D", "Alignment disabled, using identity alignment");
        }

        //Point aSize = new Point(parameters.rawSize.x/(2*parameters.tile) + 1, parameters.rawSize.y/(2*parameters.tile) + 1);
        Point border = new Point(16,16);
        // NOTE: inputAlter is created above the alignment block (shared with FlowNet).
        //alignmentTex = new GLTexture(aSize, new GLFormat(GLFormat.DataType.FLOAT_32, 2), alignment, GL_NEAREST, GL_MIRRORED_REPEAT);

        //counter.put(1.0f,1.0f);
        float cnt1 = 2.0f;

        float cnt2 = 1.0f;
        //Log.d("ESD4D", "alignment size: " + aSize.x + " " + aSize.y);
        Log.d("ESD4D", "alignment size: " + parameters.alignmentSize.x + " " + parameters.alignmentSize.y);
        float maxBlack = Math.max(blackLevel[0], Math.max(blackLevel[1], Math.max(blackLevel[2], blackLevel[3])));
        float minLevel = (float) (1.0/(double)(parameters.whiteLevel-maxBlack));

        // The base frame's pixels are on the GPU now (inputBase upload, shared
        // with the alignment stage; synchronous). The loop below only touches
        // its GPU texture and scalar pair metadata, and the base index is
        // never loaded there, so release the native copy up-front: it would
        // otherwise outlive the whole merge.
        images.get(0).close();

        // getBase() aliases base onto baseAlter from the first iteration,
        // orphaning the original base texture; reclaim it post-loop below.
        final GLTexture mergeBase0 = base;

        long mergeLoopT = System.currentTimeMillis();
        int alterSlot = 0;
        // 3a SR detail layer: motion-compensated residuals accumulate on
        // multi-frame upscales and on explicit 1.0x (enhanced native:
        // detail without resizing; Disabled stays the untouched legacy
        // path). Single frames, native/downscale sizes and over-budget
        // sensors skip allocation entirely and render exactly as before
        // (srGain 0 below).
        srActive = false;
        srAccA = null;
        srAccB = null;
        srAccFinal = null;
        srAccumFrames = 0;
        // Engagement rule for both SR layers: any genuine upscale whose target
        // is at most srMaxExpand times the raw slice - the per-sensor resize
        // factor and a digital-zoom crop both count, since computeResizedTarget
        // already folds the zoom expansion in. The drizzle's sub-pixel
        // diversity (handheld dither plus the synthesized jitter) supports
        // roughly 2x; beyond that the raw samples land several output pixels
        // apart and the fused field degenerates into a smooth interpolation,
        // which the KernelNet reconstructs better. The cap is a tunable
        // (default 4.0): the fused luma's honest band ends at the raw Nyquist
        // (0.5/expand c/px), so it leaves the eye's sensitive range around 4x
        // while the fused's magnified raw grain keeps growing. Downscales have
        // nothing to resolve.
        // Declared here: both the 3a setup and the full-SR gate read them.
        Point srTgt = null;
        float srExpand = 0f;
        boolean srScaleOk = false;
        try {
            float srFactor = parameters != null ? parameters.getActiveUpscaleFactor() : 0f;
            try {
                srTgt = com.particlesdevs.photoncamera.processing.render.Parameters
                        .computeResizedTarget(parameters, parameters != null ? parameters.rawSize : null);
            } catch (Exception ignored) {
            }
            srExpand = (srTgt != null && parameters != null && parameters.rawSize != null
                    && parameters.rawSize.x > 0)
                    ? srTgt.x / (float) parameters.rawSize.x : 0f;
            srScaleOk = srExpand > 1.0f + 1e-4f && srExpand <= srMaxExpand + 1e-4f;
            // The 3a layer accumulates even when the full-SR will run: it is
            // the fallback if the full-SR export fails, and it only costs a
            // packed-size accumulator pair (~50MB). merge2o keeps the merged
            // frame clean while the full-SR ferry exists and the post skips
            // the layer the same way, so a working full-SR is unaffected -
            // but a failed export no longer leaves the shot with nothing.
            boolean wantSR = srDetailEnable && images != null && images.size() > 1
                    && srScaleOk
                    && packedSize != null
                    && packedSize.x > 0 && packedSize.y > 0;
            if (wantSR) {
                long needBytes = 2L * (long) packedSize.x * (long) packedSize.y * 4L * 2L;
                if (needBytes <= (long) srMemoryCapMB * 1024L * 1024L) {
                    srAccA = new GLTexture(packedSize, new GLFormat(GLFormat.DataType.FLOAT_16, 4), null, GL_NEAREST, GL_CLAMP_TO_EDGE);
                    srAccB = new GLTexture(packedSize, new GLFormat(GLFormat.DataType.FLOAT_16, 4), null, GL_NEAREST, GL_CLAMP_TO_EDGE);
                    srActive = true;
                    Log.d("ESD4D", "SR detail active: " + packedSize.x + "x" + packedSize.y
                            + " packed accum x2, factor=" + srFactor);
                } else {
                    Log.d("ESD4D", "SR detail skipped: need " + (needBytes / 1048576)
                            + "MB over cap " + srMemoryCapMB + "MB");
                }
            }
        } catch (Throwable t) {
            Log.e("ESD4D", "SR detail setup failed, disabled", t);
            srActive = false;
            if (srAccA != null) {
                try {
                    srAccA.close();
                } catch (Exception ignored) {
                }
                srAccA = null;
            }
            if (srAccB != null) {
                try {
                    srAccB.close();
                } catch (Exception ignored) {
                }
                srAccB = null;
            }
        }
        // 3b full-SR drizzle: burst frames accumulate onto the output grid
        // (translation + local block motion, same atlas convention as
        // mergeAlign). Active on multi-frame upscales with a real aligner
        // (halide/pyramid; flownet uses per-frame flow fields and off has no
        // motion, both fall back) and aspect-sliced shots off (their output
        // grid differs from the merge grid). Post consumes the accumulation
        // instead of the aniso upscale; any failure falls back to it.
        Point srFullTarget = null;
        GLTexture srDriOut = null;
        float srFullOx = 0f, srFullOy = 0f;
        float srFullW = 0f, srFullH = 0f;
        float[] srRwArr = null, srGwArr = null, srBwArr = null;
        srFullActive = false;
        srDriA = null;
        srDriB = null;
        srFullBase = null;
        srFullCPU = null;
        srFullSize = null;
        try {
            float srFFactor = parameters != null ? parameters.getActiveUpscaleFactor() : 0f;
            boolean srFlowAlign = Objects.equals(alignerSelect, "flownet");
            boolean srOffAlign = Objects.equals(alignerSelect, "off");
            boolean srZoomed = false;
            boolean srAspect = false;
            try {
                srZoomed = com.particlesdevs.photoncamera.app.PhotonCamera.getCaptureController() != null
                        && com.particlesdevs.photoncamera.app.PhotonCamera.getCaptureController().zoomController.isZoomed();
                srAspect = com.particlesdevs.photoncamera.app.PhotonCamera.getSettings().aspect169 && !srZoomed;
            } catch (Exception ignored) {
            }
            // FlowNet is supported: srwarp samples the dense flow directly
            // (srFlowAlign), keeping the continuous displacement where the
            // merge's own warp truncates it to integers. The identity ("off")
            // atlas carries no motion, so the drizzle stays declined there.
            if (srFullEnable && images != null && images.size() > 1
                    && srScaleOk && !srOffAlign && !srAspect
                    && parameters != null && parameters.rawSize != null
                    && parameters.rawSize.x > 0 && parameters.rawSize.y > 0
                    && alignmentTex != null && base != null
                    && srFullFitsMemory()) {
                int[] srMap = srChannelMap((int) parameters.cfaPattern, cfaShift);
                if (srMap == null) {
                    Log.d("ESD4D", "Full-SR skipped: unsupported CFA pattern " + parameters.cfaPattern);
                } else {
                    // Collect the map before the seed dispatch: the reference
                    // frame is one sample among equals, so it must not carry
                    // the bilinear kernel the alters no longer use.
                    ensureSrRefMap();
                    srRwArr = new float[4];
                    srGwArr = new float[4];
                    srBwArr = new float[4];
                    // One-hot selectors (1 at every position of the colour,
                    // including both green positions): merge/srluma reads them
                    // as own-channel tests (> 0.5) and as neighbour weights.
                    // 0.5 at the greens made every green site fail the own
                    // test, so it was rebuilt from its diagonal neighbours
                    // instead of using its own sample - half of all sites and
                    // 71.5% of the luma.
                    srRwArr[srMap[0]] = 1f;
                    srGwArr[srMap[1]] = 1f;
                    srGwArr[srMap[2]] = 1f;
                    srBwArr[srMap[3]] = 1f;
                Point tgt = com.particlesdevs.photoncamera.processing.render.Parameters.computeResizedTarget(parameters, parameters.rawSize);
                // The drizzled image replaces the aniso reconstruction, so it
                // must use the aniso geometry: the crop stretched over the
                // target (crop x factor, zoom expand folded into the target
                // size). The tone stages map output UV onto the crop's
                // gain-map footprint and rescale the sensor bounds by
                // output/rawSize, both of which assume the output covers the
                // crop - not a full-frame canvas with the crop placed at its
                // sensor offset (which also left the map UV outside [0,1]).
                srFullW = parameters.rawSize.x;
                srFullH = parameters.rawSize.y;
                srFullOx = 0f;
                srFullOy = 0f;
                srDriA = new GLTexture(tgt, new GLFormat(GLFormat.DataType.UNSIGNED_32, 1), null, GL_NEAREST, GL_CLAMP_TO_EDGE);
                srDriB = new GLTexture(tgt, new GLFormat(GLFormat.DataType.UNSIGNED_32, 1), null, GL_NEAREST, GL_CLAMP_TO_EDGE);
                srLumaSize = new Point(parameters.rawSize);
                srLumaTex = new GLTexture(srLumaSize,
                        new GLFormat(GLFormat.DataType.FLOAT_16, 4), null, GL_LINEAR, GL_CLAMP_TO_EDGE);
                // Base frame first (exact init, viewport/FBO/blend free:
                // compute dispatches know no viewport). Its per-site luma must
                // exist before the drizzle reads it.
                if (!dispatchSrLuma(base, srRwArr, srGwArr, srBwArr)) {
                    throw new IllegalStateException("SR luma pass failed");
                }
                glProg.setLayout(tile, tile, 1);
                glProg.useAssetProgram("merge/srwarp", true);
                glProg.setTexture("srLumaTex", srLumaTex);
                glProg.setTexture("diffPacked", base);
                glProg.setTexture("basePacked", base);
                glProg.setTexture("alignmentTexture", srAlignTex());
                glProg.setTexture("srRefMap", srRefTex != null ? srRefTex : base);
                glProg.setVar("srRefine", srRefTex != null ? srRefine : 0f);
                glProg.setTextureCompute("srDriIn", srDriB, false);
                glProg.setTextureCompute("srDriOut", srDriA, true);
                glProg.setVar("srShift", 0, 0);
                glProg.setVar("srAlignSize", parameters.alignmentSize);
                glProg.setVar("srFlowAlign", Objects.equals(alignerSelect, "flownet") ? 1 : 0);
                glProg.setVar("srRawHalf", new Point(parameters.rawSize.x / 2, parameters.rawSize.y / 2));
                glProg.setVar("srCfa", cfaShift);
                glProg.setVar("srFullPerOut", srFullW / (float) tgt.x, srFullH / (float) tgt.y);
                glProg.setVar("srOrigin", srFullOx, srFullOy);
                glProg.setVar("srExpose", 1.f / images.get(0).pair.layerMpy);
                glProg.setVar("srZeroMotion", 1f);
                glProg.setVar("srJitter", srJitter);
                glProg.setVar("srFrame", 0);
                bindSrTrust();
                glProg.setVar("srMotionMax", Math.max(srMotionMax, 1f));
                glProg.setVar("srFirst", 1);
                while (android.opengl.GLES30.glGetError() != android.opengl.GLES30.GL_NO_ERROR) {
                }
                glProg.computeAuto(srDriA.mSize, 1);
                if (android.opengl.GLES30.glGetError() != android.opengl.GLES30.GL_NO_ERROR) {
                    throw new IllegalStateException("SR base drizzle dispatch failed");
                }
                srDriOut = srDriA;
                srFullTarget = tgt;
                srFullActive = true;
                Log.d("ESD4D", "Full-SR drizzle active: " + tgt.x + "x" + tgt.y
                        + ", factor=" + srFFactor + ", expand=" + srExpand);
                }
            } else {
                Log.d("ESD4D", "Full-SR declined: enable=" + srFullEnable
                        + " frames=" + (images != null ? images.size() : 0)
                        + " expand=" + srExpand + " factor=" + srFFactor
                        + " flow=" + srFlowAlign + " off=" + srOffAlign + " aspect=" + srAspect
                        + " raw=" + (parameters != null && parameters.rawSize != null
                                ? parameters.rawSize.x + "x" + parameters.rawSize.y : "null")
                        + " cfa=" + (parameters != null ? parameters.cfaPattern : -1));
            }
        } catch (Throwable t) {
            Log.e("ESD4D", "Full-SR setup failed, disabled", t);
            srFullActive = false;
            if (srDriA != null) {
                try {
                    srDriA.close();
                } catch (Exception ignored) {
                }
                srDriA = null;
            }
            if (srDriB != null) {
                try {
                    srDriB.close();
                } catch (Exception ignored) {
                }
                srDriB = null;
            }
        }
        // Bayer drizzle for scaled/SR DNG: same motion warp as the RGB
        // drizzle, accumulated per Bayer site (merge2o channel rule, so every
        // CFA layout the merge supports works) with a consensus clamp for
        // ghost rejection. Active only when the processor requested a DNG
        // save and the factor is explicitly set; the running consensus needs
        // no weight plane and no normalize pass.
        Point srBayerTarget = null;
        float srBayerOx = 0f, srBayerOy = 0f;
        float srBayerW = 0f, srBayerH = 0f;
        GLTexture srBayOut = null;
        srBayerActive = false;
        srBayA = null;
        srBayB = null;
        srBayerBase = null;
        srBayerCPU = null;
        srBayerCPUSize = null;
        try {
            float srBFactor = parameters != null ? parameters.getActiveUpscaleFactor() : 0f;
            boolean wantBay = saveDngWanted && images != null && images.size() > 1
                    && !com.particlesdevs.photoncamera.processing.render.Parameters.isResizeDisabled(srBFactor)
                    && !Objects.equals(alignerSelect, "flownet") && !Objects.equals(alignerSelect, "off")
                    && parameters != null && parameters.rawSize != null
                    && parameters.rawSize.x > 0 && parameters.rawSize.y > 0
                    && alignmentTex != null && base != null;
            int[] srBayMap = wantBay ? srChannelMap((int) parameters.cfaPattern, cfaShift) : null;
            if (wantBay && srBayMap == null) {
                Log.d("ESD4D", "Bayer drizzle skipped: unsupported CFA pattern " + parameters.cfaPattern);
            } else if (wantBay) {
                // Same reasoning as the JPEG seed: the reference frame is
                // reconstructed steered like every alter.
                ensureSrRefMap();
                Point tgt = com.particlesdevs.photoncamera.processing.render.Parameters.computeResizedTarget(parameters, parameters.rawSize);
                boolean sCropped = parameters.isCropped && parameters.fullRawSize != null
                        && parameters.fullRawSize.x > 0 && parameters.fullRawSize.y > 0;
                srBayerW = sCropped ? parameters.fullRawSize.x : parameters.rawSize.x;
                srBayerH = sCropped ? parameters.fullRawSize.y : parameters.rawSize.y;
                srBayerOx = (sCropped && parameters.cropOrigin != null) ? parameters.cropOrigin.x : 0f;
                srBayerOy = (sCropped && parameters.cropOrigin != null) ? parameters.cropOrigin.y : 0f;
                srBayA = new GLTexture(tgt, new GLFormat(GLFormat.DataType.UNSIGNED_32, 1), null, GL_NEAREST, GL_CLAMP_TO_EDGE);
                srBayB = new GLTexture(tgt, new GLFormat(GLFormat.DataType.UNSIGNED_32, 1), null, GL_NEAREST, GL_CLAMP_TO_EDGE);
                glProg.setLayout(tile, tile, 1);
                glProg.useAssetProgram("merge/srbayer", true);
                glProg.setTexture("alterPacked", base);
                glProg.setTexture("alignmentTexture", srAlignTex());
                glProg.setTexture("diffPacked", base);
                glProg.setTexture("srRefMap", srRefTex != null ? srRefTex : base);
                glProg.setVar("srRefine", srRefTex != null ? srRefine : 0f);
                glProg.setTextureCompute("srBayIn", srBayB, false);
                glProg.setTextureCompute("srBayOut", srBayA, true);
                glProg.setVar("srShift", 0, 0);
                glProg.setVar("srAlignSize", parameters.alignmentSize);
                glProg.setVar("srFlowAlign", Objects.equals(alignerSelect, "flownet") ? 1 : 0);
                glProg.setVar("srRawHalf", new Point(parameters.rawSize.x / 2, parameters.rawSize.y / 2));
                glProg.setVar("srCfa", cfaShift);
                glProg.setVar("srFullPerOut", srBayerW / (float) tgt.x, srBayerH / (float) tgt.y);
                glProg.setVar("srOrigin", srBayerOx, srBayerOy);
                glProg.setVar("srExpose", 1.f / images.get(0).pair.layerMpy);
                glProg.setVar("srBaseExpose", 1.f / images.get(0).pair.layerMpy);
                glProg.setTexture("basePacked", base);
                glProg.setVar("srZeroMotion", 1f);
                glProg.setVar("srFirst", 1);
                bindSrTrust();
                glProg.setVar("srMotionMax", Math.max(srMotionMax, 1f));
                while (android.opengl.GLES30.glGetError() != android.opengl.GLES30.GL_NO_ERROR) {
                }
                glProg.computeAuto(srBayA.mSize, 1);
                if (android.opengl.GLES30.glGetError() != android.opengl.GLES30.GL_NO_ERROR) {
                    throw new IllegalStateException("SR Bayer base dispatch failed");
                }
                srBayOut = srBayA;
                srBayerTarget = tgt;
                srBayerActive = true;
                Log.d("ESD4D", "Bayer drizzle active: " + tgt.x + "x" + tgt.y + ", factor=" + srBFactor);
            }
        } catch (Throwable t) {
            Log.e("ESD4D", "Bayer drizzle setup failed, disabled", t);
            srBayerActive = false;
            if (srBayA != null) {
                try {
                    srBayA.close();
                } catch (Exception ignored) {
                }
                srBayA = null;
            }
            if (srBayB != null) {
                try {
                    srBayB.close();
                } catch (Exception ignored) {
                }
                srBayB = null;
            }
        }
        for (int f = 0; f < images.size(); f++) {
            startT();
            if(f == minExpIdx) continue;
            int ind = f;
            if(ind == 0){
                ind = minExpIdx;
            }
            ImageFrame frame = images.get(ind);
            float exposure = 1.f/frame.pair.layerMpy;
            Point shift = PyramidAlignment.alignmentShift(parameters, ind);
            //int f = 1;
            Log.d("ESD4D", "load:"+frame.pair.curlayer.name() + " " + frame.pair.layerMpy);
            GLTexture alterTarget = alterSlot == 0 ? inputAlter : inputAlterAlt;
            waitUploadFence(alterUploadFences, alterSlot);
            long stageT = System.currentTimeMillis();
            alterTarget.loadRawHalf(frame.buffer);
            Log.d("ESD4D", "Stage[merge:upload] elapsed:" + (System.currentTimeMillis() - stageT) + " ms f=" + f);

            GLTexture flowTex = null;
            if(Objects.equals(alignerSelect, "flownet")) {
                // Dense FlowNet optical flow for THIS alter frame, computed just
                // in time (one pair at a time, no stored flow fields). Must run
                // before the mergeAlign program is bound below.
                flowTex = flowNetAlignment.computeFlow(ind);
                // Guided upsample of the dense flow against the running
                // base: the SR drizzles below sample this edge-aware field.
                dispatchSrFlowResample(flowTex);
            }

            // This frame's only native reads are the two synchronous uploads
            // above: merge00's alterTarget (line above) and, on the FlowNet
            // path, inputAlter inside computeFlow. Everything below runs on
            // GPU textures plus scalar pair metadata, so release the f16
            // buffer now instead of after combine: the burst drains one frame
            // earlier, off the merge's peak (all 8 f16 frames are live when
            // the loop starts). close() is idempotent, so HdrxProcessor's
            // post-merge loop stays a safe net.
            frame.close();

            // Convert inputAlter to alter (vec4 format)
            stageT = System.currentTimeMillis();
            glProg.setLayout(tile, tile, 1);
            glProg.useAssetProgram("merge/merge00", true);
            glProg.setVar("exposure", 1.f/images.get(0).pair.layerMpy);
            glProg.setVar("createDiff", 0);
            glProg.setVar("cfaShift", cfaShift);
            glProg.setTexture("inTexture", alterTarget);
            glProg.setTextureCompute("outTexture", alter, true);
            glProg.computeAuto(new Point(alter.mSize.x, alter.mSize.y), 1);
            // merge00 is this ring slot's only reader: signal the GPU point
            // after which the slot may be overwritten, without draining the
            // queue. The next two frames use the other slot in between.
            markUploadFence(alterUploadFences, alterSlot);
            alterSlot ^= 1;

            correctHotPixelsInAlter(hotPixelBuffer, hotPixelCount);
            //alignmentTex.loadData(alignment.position((ind-1)*(aSize.x*aSize.y*4*2)));
            Log.d("ESD4D", "Stage[merge:merge00] elapsed:" + (System.currentTimeMillis() - stageT) + " ms f=" + f);
            glProg.setDefine("TILE_AL", parameters.tile);
            stageT = System.currentTimeMillis();
            glProg.setLayout(tile, tile, 1);
            glProg.useAssetProgram(Objects.equals(alignerSelect, "flownet") ? "merge/mergeAlignFlow" : "merge/mergeAlign", true);
            glProg.setVar("rawHalf", rawHalf);
            glProg.setVar("whitePoint", parameters.whitePoint);
            // Red-site origin shift for mergeAlign's noise-model repack
            // (harmless no-op uniform for mergeAlignFlow).
            glProg.setVar("cfaShift", cfaShift);
            glProg.setVar("minLevel",minLevel);
            glProg.setVar("exposure", exposure);
            glProg.setVar("analogBalance", analogBalance);
            if(exposure >= 0.95f) {
                if(lowCnt > 1)
                    glProg.setVar("exposureLow", minExp - 0.05f);
                else {
                    glProg.setVar("exposureLow", 0.0f);
                }
            } else {
                glProg.setVar("exposureLow", 0.0f);
            }
            glProg.setVar("createDiff", 1);
            glProg.setVar("noiseS", noiseS);
            glProg.setVar("noiseO", noiseO);
            glProg.setVar("border", border);
            if(Objects.equals(alignerSelect, "flownet")) {
                glProg.setTexture("alignmentTexture", flowTex);
            } else {
                glProg.setVar("shift", shift);
                glProg.setVar("alignmentSize", parameters.alignmentSize);
                glProg.setTexture("alignmentTexture", alignmentTex);
            }
            glProg.setTexture("inTexture", inputBase);
            glProg.setTextureCompute("baseTexture",base, false);
            glProg.setTextureCompute("alterTexture", alter, false);
            glProg.setTextureCompute("outTexture", baseDiff, true);
            glProg.computeAuto(baseDiff.mSize, 1);
            gpuSyncProfile();
            Log.d("ESD4D", "Stage[merge:mergeAlign] elapsed:" + (System.currentTimeMillis() - stageT) + " ms f=" + f);

            // Sub-pixel alignment refinement for the SR drizzles: a bounded
            // Lucas-Kanade step against the running base, per alignment cell,
            // dispatched before the drizzles so both paths share one pass.
            // baseDiff is fresh here. Failure just drops the correction.
            if (srRefTex != null && srRefine > 0f && (srFullActive || srBayerActive)
                    && baseDiff != null && base != null) {
                try {
                    glProg.setLayout(8, 8, 1);
                    glProg.useAssetProgram("merge/srrefine", true);
                    glProg.setTexture("diffPacked", baseDiff);
                    glProg.setTexture("basePacked", base);
                    // Half the atlas cell: the refine map runs at 2x its
                    // resolution, so the cell in packed texels is tile/4.
                    glProg.setVar("srRefCell", Math.max(1, parameters.tile / 4));
                    glProg.setTextureCompute("srRefOut", srRefTex, true);
                    glProg.computeAuto(srRefSize, 1);
                } catch (Throwable t) {
                    Log.e("ESD4D", "SR refine pass failed, disabled", t);
                    try {
                        srRefTex.close();
                    } catch (Exception ignored) {
                    }
                    srRefTex = null;
                    srRefSize = null;
                }
            }

            // SR detail: fold this frame's motion-compensated residual into
            // the ping-pong accumulator (dispatched before the kernelnet join
            // below so GPU work overlaps the CPU wait). baseDiff is fresh
            // here; combine only reads it. Any failure disables the layer and
            // the shot continues exactly as without it.
            if (srActive && srAccA != null && srAccB != null && baseDiff != null) {
                try {
                    GLTexture srIn = (srAccumFrames % 2 == 0) ? srAccA : srAccB;
                    GLTexture srOut = (srAccumFrames % 2 == 0) ? srAccB : srAccA;
                    glProg.setLayout(tile, tile, 1);
                    glProg.useAssetProgram("merge/sradd", true);
                    glProg.setTextureCompute("srAccIn", srIn, false);
                    glProg.setTextureCompute("srDiffIn", baseDiff, false);
                    glProg.setTextureCompute("srAccOut", srOut, true);
                    glProg.setVar("srClamp", srDetailClamp);
                    glProg.setVar("srFirst", srAccumFrames == 0 ? 1 : 0);
                    glProg.computeAuto(srOut.mSize, 1);
                    srAccFinal = srOut;
                    srAccumFrames++;
                } catch (Throwable t) {
                    Log.e("ESD4D", "SR accum failed, disabling", t);
                    srActive = false;
                }
            }

            // 3b full-SR drizzle: warp this frame onto the output grid,
            // accumulated via ping-pong compute (viewport/FBO/blend free by
            // construction). baseDiff is fresh here (robustness weight
            // source); combine only reads it. Any failure disables the layer
            // for a clean fallback.
            if (srFullActive && srDriA != null && srDriB != null && srDriOut != null
                    && srFullTarget != null
                    && alter != null && baseDiff != null && alignmentTex != null) {
                try {
                    GLTexture srDriIn = srDriOut;
                    GLTexture srDriNext = (srDriOut == srDriA) ? srDriB : srDriA;
                    // Rebuild this frame's luma FIRST, then load srwarp: the
                    // luma pass loads merge/srluma, and setTextureCompute
                    // resolves the image layout against the *active* program,
                    // so binding srDriIn/srDriOut before it would look them up
                    // in srluma's map, silently skip glBindImageTexture, and
                    // leave the dispatch reading/writing whatever units were
                    // bound before (the luma texture, baseDiff). Same order as
                    // the base seed below.
                    if (!dispatchSrLuma(alter, srRwArr, srGwArr, srBwArr)) {
                        throw new IllegalStateException("SR luma pass failed");
                    }
                    glProg.setLayout(tile, tile, 1);
                    glProg.useAssetProgram("merge/srwarp", true);
                    glProg.setTexture("srLumaTex", srLumaTex);
                    glProg.setTexture("diffPacked", baseDiff);
                    glProg.setTexture("basePacked", base);
                    glProg.setTexture("alignmentTexture", srAlignTex());
                    glProg.setTexture("srRefMap", srRefTex != null ? srRefTex : base);
                    glProg.setVar("srRefine", srRefTex != null ? srRefine : 0f);
                    glProg.setTextureCompute("srDriIn", srDriIn, false);
                    glProg.setTextureCompute("srDriOut", srDriNext, true);
                    glProg.setVar("srShift", shift);
                    glProg.setVar("srAlignSize", parameters.alignmentSize);
                    glProg.setVar("srFlowAlign", Objects.equals(alignerSelect, "flownet") ? 1 : 0);
                    glProg.setVar("srRawHalf", new Point(parameters.rawSize.x / 2, parameters.rawSize.y / 2));
                    glProg.setVar("srCfa", cfaShift);
                    glProg.setVar("srFullPerOut", srFullW / (float) srFullTarget.x, srFullH / (float) srFullTarget.y);
                    glProg.setVar("srOrigin", srFullOx, srFullOy);
                    glProg.setVar("srExpose", exposure);
                    glProg.setVar("srZeroMotion", 0f);
                    glProg.setVar("srJitter", srJitter);
                    glProg.setVar("srFrame", f);
                    bindSrTrust();
                    glProg.setVar("srMotionMax", Math.max(srMotionMax, 1f));
                    glProg.setVar("srFirst", 0);
                    while (android.opengl.GLES30.glGetError() != android.opengl.GLES30.GL_NO_ERROR) {
                    }
                    glProg.computeAuto(srDriNext.mSize, 1);
                    if (android.opengl.GLES30.glGetError() != android.opengl.GLES30.GL_NO_ERROR) {
                        throw new IllegalStateException("SR drizzle dispatch failed");
                    }
                    srDriOut = srDriNext;
                } catch (Throwable t) {
                    Log.e("ESD4D", "Full-SR drizzle failed, disabling", t);
                    srFullActive = false;
                }
            }

            // Bayer drizzle for scaled/SR DNG: same warp, per-site channel,
            // consensus-clamped running average (no weight plane, no normalize
            // pass). Any failure disables the layer for the legacy DNG path.
            if (srBayerActive && srBayA != null && srBayB != null && srBayOut != null
                    && srBayerTarget != null
                    && alter != null && alignmentTex != null) {
                try {
                    GLTexture srBayIn = srBayOut;
                    GLTexture srBayNext = (srBayOut == srBayA) ? srBayB : srBayA;
                    glProg.setLayout(tile, tile, 1);
                    glProg.useAssetProgram("merge/srbayer", true);
                    glProg.setTexture("alterPacked", alter);
                    glProg.setTexture("alignmentTexture", srAlignTex());
                    glProg.setTexture("diffPacked", baseDiff);
                    glProg.setTexture("srRefMap", srRefTex != null ? srRefTex : base);
                    glProg.setVar("srRefine", srRefTex != null ? srRefine : 0f);
                    glProg.setTextureCompute("srBayIn", srBayIn, false);
                    glProg.setTextureCompute("srBayOut", srBayNext, true);
                    glProg.setVar("srShift", shift);
                    glProg.setVar("srAlignSize", parameters.alignmentSize);
                    glProg.setVar("srFlowAlign", Objects.equals(alignerSelect, "flownet") ? 1 : 0);
                    glProg.setVar("srRawHalf", new Point(parameters.rawSize.x / 2, parameters.rawSize.y / 2));
                    glProg.setVar("srCfa", cfaShift);
                    glProg.setVar("srFullPerOut", srBayerW / (float) srBayerTarget.x, srBayerH / (float) srBayerTarget.y);
                    glProg.setVar("srOrigin", srBayerOx, srBayerOy);
                    glProg.setVar("srExpose", exposure);
                    glProg.setVar("srBaseExpose", 1.f / images.get(0).pair.layerMpy);
                    glProg.setTexture("basePacked", mergeBase0);
                    glProg.setVar("srZeroMotion", 0f);
                    glProg.setVar("srFirst", 0);
                    bindSrTrust();
                    glProg.setVar("srMotionMax", Math.max(srMotionMax, 1f));
                    while (android.opengl.GLES30.glGetError() != android.opengl.GLES30.GL_NO_ERROR) {
                    }
                    glProg.computeAuto(srBayNext.mSize, 1);
                    if (android.opengl.GLES30.glGetError() != android.opengl.GLES30.GL_NO_ERROR) {
                        throw new IllegalStateException("SR Bayer drizzle dispatch failed");
                    }
                    srBayOut = srBayNext;
                } catch (Throwable t) {
                    Log.e("ESD4D", "Bayer drizzle failed, disabling", t);
                    srBayerActive = false;
                }
            }

            if (PhotonCamera.DEBUG)
                Log.d("ESD4D", "create diff");

            // First combine pass: collect the KernelNet result that has been
            // running concurrently with alignment and this frame's merge00 /
            // mergeAlign work. Waits only for any inference remainder; the
            // texture build below needs the GL thread anyway. SR paths may
            // have collected it already (steered seed), in which case this
            // is a no-op.
            stageT = System.currentTimeMillis();
            kernelNetThread = joinKernelNet(kernelNetThread, kernelNetResult, "f=" + f);

            glProg.setLayout(tile, tile, 1);
            stageT = System.currentTimeMillis();
            glProg.useAssetProgram("merge/mergeCombineWeight1", true);
            glProg.setVar("cfaPattern", parameters.cfaPattern);
            glProg.setTexture("inTex", inputBase);
            glProg.setTexture("kernelsMap", kernelsMap);
            // Optical flow refinement: brute-force diagonal candidate wins
            // only when it beats the zero offset beyond the shader's gates.
            //glProg.setVar("enableFlow", enableFlowRefinement ? 1 : 0);
            glProg.setVar("flowNoiseS", rawNoiseS);
            glProg.setVar("flowNoiseO", rawNoiseO);
            glProg.setTextureCompute("inTexture", base, false);
            glProg.setTextureCompute("diffTexture", baseDiff, false);
            base = getBase();
            glProg.setTextureCompute("outTexture", base, true);
            glProg.setVar("noiseS", noiseS);
            glProg.setVar("noiseO", noiseO);
            glProg.setVar("analogBalance", analogBalance);
            glProg.setVar("exposure", exposure);
            if(exposure >= 0.95f){
                glProg.setVar("weight", 1.0f/cnt1);
                //glProg.setVar("exposure", minExp);
                cnt1+=1.0f;
            } else {
                glProg.setVar("weight", 1.0f/cnt2);
                //glProg.setVar("exposure", 1.0f);
                cnt2+=1.0f;
            }
            //glProg.setVar("exposure", exposure);
            //glProg.setVar("weight",  1.0f);
            glProg.computeAuto(base.mSize, 1);
            gpuSyncProfile();
            Log.d("ESD4D", "Stage[merge:combine] elapsed:" + (System.currentTimeMillis() - stageT) + " ms f=" + f);
            endT();
        }
        Log.d("ESD4D", "Stage[merge-loop] elapsed:" + (System.currentTimeMillis() - mergeLoopT) + " ms");

        // Temporal temporaries are dead past the loop: merge2o below reads
        // only base + alignmentTex, and no finalize/export below touches
        // them. Release (~530 MB at 64 MP, plus upload fences) BEFORE the
        // readback mallocs instead of after, so peak pressure doesn't fail
        // them. Fields are nulled and AfterRun null-guards them, so a stale
        // close can never delete a recycled ID.
        if (mergeBase0 != base) mergeBase0.close();
        baseDiff.close(); baseDiff = null;
        alter.close(); alter = null;
        inputAlter.close(); inputAlter = null;
        if (inputAlterAlt != null) { inputAlterAlt.close(); inputAlterAlt = null; }
        deleteUploadFences(alterUploadFences);
        inputBase.close(); inputBase = null;

        // SR detail finalize: normalized highpass of the accumulator into
        // srHP (bound by merge2o below and exported for the post-upscale
        // apply node), then release the accumulators well before the output
        // readback. Export failure only drops the post node; merge2o still
        // applies the texture.
        srHP = null;
        if (srActive && srAccFinal != null && srAccumFrames > 0) {
            try {
                srHP = new GLTexture(packedSize, new GLFormat(GLFormat.DataType.FLOAT_16, 4), null, GL_NEAREST, GL_CLAMP_TO_EDGE);
                glProg.setLayout(tile, tile, 1);
                glProg.useAssetProgram("merge/srhp", true);
                glProg.setTextureCompute("srHpIn", srAccFinal, false);
                glProg.setTextureCompute("srHpOut", srHP, true);
                glProg.setVar("srNorm", 1.0f / srAccumFrames);
                float srT1 = Math.max(srCoring1 * kernelSigma, srCoring0 * kernelSigma + 1e-4f);
                glProg.setVar("srT0", srCoring0 * kernelSigma);
                glProg.setVar("srT1", srT1);
                glProg.computeAuto(srHP.mSize, 1);
                gpuSyncProfile();
                if (gpuHandoff) {
                    // Shared-group handoff (see the full-SR export): the detail
                    // texture crosses by name; no readback, no ferry.
                    srDetailTexID = srHP.mTextureID;
                    srDetailCPUSize = new Point(packedSize);
                    srDetailShift = cfaShift != null ? new Point(cfaShift) : null;
                } else {
                    // BufferLoad (not BindBuffer): mBuffer is 0 until Bufferize
                    // runs, and binding FBO 0 would attach this RGBA16F texture to
                    // the default framebuffer - the half-float read then hits an
                    // 8-bit buffer and fails with GL_INVALID_OPERATION (0x502).
                    srHP.BufferLoad();
                    ByteBuffer srRead = srHP.textureBufferHalfFloatNative();
                    if (srRead != null) {
                        srRead.order(java.nio.ByteOrder.nativeOrder());
                        srRead.rewind();
                        srDetailBase = srRead;
                        srDetailCPU = srRead.asShortBuffer();
                        srDetailCPUSize = new Point(packedSize);
                        srDetailShift = cfaShift != null ? new Point(cfaShift) : null;
                    } else {
                        Log.e("ESD4D", "SR detail export failed, post node will pass through");
                    }
                }
            } catch (Throwable t) {
                Log.e("ESD4D", "SR detail finalize failed, disabled", t);
                srActive = false;
                srDetailTexID = 0;
                if (srHP != null) {
                    try {
                        srHP.close();
                    } catch (Exception ignored) {
                    }
                    srHP = null;
                }
            }
        }
        if (srAccA != null) {
            try {
                srAccA.close();
            } catch (Exception ignored) {
            }
            srAccA = null;
        }
        if (srAccB != null) {
            try {
                srAccB.close();
            } catch (Exception ignored) {
            }
            srAccB = null;
        }
        srAccFinal = null;

        // Bayer drizzle export for scaled/SR DNG: read back the consensus
        // (value in the low half of each packed R32UI word, read downstream)
        // with the same non-finite sanity gate. Export failure only drops DNG
        // scaling.
        if (srBayerActive && srBayOut != null && srBayerTarget != null) {
            try {
                srBayOut.BufferLoad();
                ByteBuffer srBRead = srBayOut.textureBufferUintNative();
                boolean srBOk = false;
                if (srBRead != null) {
                    // Direct-buffer short views default to BIG_ENDIAN; the
                    // GPU wrote native halves (codebase-wide convention: set
                    // native order before reading shorts/floats).
                    srBRead.order(java.nio.ByteOrder.nativeOrder());
                    srBRead.rewind();
                    java.nio.ShortBuffer scan = srBRead.asShortBuffer();
                    // Dense probe on the leading rows (full texel coverage,
                    // both packed halves), then the same diagnostic split: the
                    // strided whole-buffer scan read only one channel and
                    // misread the pack layout, inflating phantom counts.
                    int probeRows = Math.min(srBayerTarget.y, 16);
                    int probeTexels = probeRows * srBayerTarget.x;
                    int badNaN = 0;
                    int badInf = 0;
                    int checked = 0;
                    int bandRows = 256;
                    int bands = Math.max(1, (srBayerTarget.y + bandRows - 1) / bandRows);
                    int[] bandBad = new int[bands];
                    for (int t = 0; t < probeTexels; t++) {
                        for (int cC = 0; cC < 2; cC++) {
                            int v = scan.get(t * 2 + cC) & 0xFFFF;
                            checked++;
                            if ((v & 0x7C00) == 0x7C00) {
                                if ((v & 0x03FF) == 0) {
                                    badInf++;
                                } else {
                                    badNaN++;
                                }
                                int band = (t / srBayerTarget.x) / bandRows;
                                if (band >= 0 && band < bands) bandBad[band]++;
                            }
                        }
                    }
                    long badFrac = checked > 0 ? ((long) (badNaN + badInf) * 100L) / checked : 0L;
                    StringBuilder bb = new StringBuilder();
                    for (int bi = 0; bi < bands; bi++) {
                        if (bi > 0) bb.append('/');
                        bb.append(bandBad[bi]);
                    }
                    Log.d("ESD4D", "Bayer drizzle probe: " + badNaN + " NaN + " + badInf
                            + " Inf / " + checked + " (" + badFrac + "%) bands[" + bb + "]");
                    if (badFrac < 25L) {
                        // Sparse poison is healed at compact (neighbor fill);
                        // dense faults above the fraction still fall back.
                        srBRead.rewind();
                        srBayerBase = srBRead;
                        srBayerCPU = srBRead.asShortBuffer();
                        srBayerCPUSize = new Point(srBayerTarget);
                        srBOk = true;
                    } else {
                        Log.e("ESD4D", "Bayer drizzle export discarded: " + badNaN
                                + " NaN + " + badInf + " Inf / " + checked
                                + " (" + badFrac + "%) bands[" + bb + "]");
                        com.particlesdevs.photoncamera.util.Allocator.free(srBRead);
                    }
                } else {
                    Log.e("ESD4D", "Bayer drizzle export failed, legacy DNG path");
                }
                if (!srBOk) srBayerActive = false;
            } catch (Throwable t) {
                Log.e("ESD4D", "Bayer drizzle export failed, disabled", t);
                srBayerActive = false;
            }
            try {
                if (srBayA != null) srBayA.close();
            } catch (Exception ignored) {
            }
            srBayA = null;
            try {
                if (srBayB != null) srBayB.close();
            } catch (Exception ignored) {
            }
            srBayB = null;
            srBayOut = null;
        }

        // 3b export: read back the drizzled accumulation for post (which
        // normalizes by the weight plane), then release both GL textures well
        // before the output readback. A strided sanity scan for non-finite
        // halves discards corruption into a clean fallback. Export failure
        // only drops post consumption.
        if (srFullActive && srDriOut != null && srFullTarget != null) {
            try {
                if (gpuHandoff) {
                    // Shared-group handoff: the accumulator texture crosses to
                    // the post by name (this context stays alive until after
                    // the post runs), so there is no readback and no ferry.
                    srFullTexID = srDriOut.mTextureID;
                    srFullSize = new Point(srFullTarget);
                } else {
                    // Both accumulators stay alive until after the readback:
                    // releasing the idle one first saved ~400MB but put a
                    // glDeleteFramebuffers/glDeleteTextures in the readback's
                    // path, and any state it unbinds turns the export into a
                    // silent passthrough. The memory estimate accounts for the
                    // full peak instead.
                    srDriOut.BufferLoad();
                    ByteBuffer srFRead = srDriOut.textureBufferUintNative();
                    boolean srFOk = false;
                    if (srFRead != null) {
                        srFRead.order(java.nio.ByteOrder.nativeOrder());
                        srFRead.rewind();
                        java.nio.ShortBuffer scan = srFRead.asShortBuffer();
                        int n = scan.remaining();
                        boolean finite = true;
                        for (int i = 0; i < n; i += 256) {
                            int v = scan.get(i) & 0xFFFF;
                            if ((v & 0x7C00) == 0x7C00) {
                                finite = false;
                                break;
                            }
                        }
                        if (finite) {
                            srFRead.rewind();
                            srFullBase = srFRead;
                            srFullCPU = srFRead.asShortBuffer();
                            srFullSize = new Point(srFullTarget);
                            srFOk = true;
                        } else {
                            Log.e("ESD4D", "Full-SR export discarded: non-finite halves");
                            com.particlesdevs.photoncamera.util.Allocator.free(srFRead);
                        }
                    } else {
                        Log.e("ESD4D", "Full-SR export failed, post will pass through");
                    }
                    if (!srFOk) srFullActive = false;
                }
            } catch (Throwable t) {
                Log.e("ESD4D", "Full-SR export failed, disabled", t);
                srFullActive = false;
                srFullTexID = 0;
            }
            try {
                if (srDriA != null && srDriA.mTextureID != srFullTexID) srDriA.close();
            } catch (Exception ignored) {
            }
            srDriA = null;
            try {
                if (srDriB != null && srDriB.mTextureID != srFullTexID) srDriB.close();
            } catch (Exception ignored) {
            }
            srDriB = null;
            srDriOut = null;
        }

        // The merge result stays normalized fp16 end-to-end: merge2o unpacks
        // the packed quads straight into the R16F output buffer (no uint16
        // re-encode); PostPipeline consumes it as-is and the uint16 DNG save
        // re-encodes on the CPU (Allocator.createU16FromF16).
        // Loop temporaries were already released above, ahead of the
        // finalize/export readbacks.
        long mergeOutT = System.currentTimeMillis();

        glProg.setLayout(tile,tile,1);
        glProg.useAssetProgram("merge/merge2o");
        glProg.setVar("cfaShift", cfaShift); // uniform: GLProg clears defines after each load
        glProg.setTexture("inTexture",base);
        glProg.setTexture("alignmentTexture", alignmentTex);
        // SR detail: precomputed normalized highpass (srGain = strength;
        // normalization lives in srhp), bound for merge2o and exported for
        // post. Falls back to base so the sampler always has a binding
        // (untouched while the gain is 0).
        // When the full-SR drizzle *and its export* both succeeded
        // (srFullActive survives to here), the merged frame must stay exactly
        // as the non-SR path: the resolve replaces the band downstream. If
        // the export failed, srFullActive is already false and the 3a layer
        // is the only SR detail left, so apply it here as before.
        GLTexture srBind = (srActive && srHP != null) ? srHP : base;
        float srGain = (srActive && srHP != null && srAccumFrames > 0 && !srFullActive) ? srDetailStrength : 0f;
        glProg.setVar("srGain", srGain);
        glProg.setTexture("srDetail", srBind);
        result.BufferLoad();
        // Issue every block's draw + readback into the PBO ring, then tear
        // down CPU-side resources (AfterRun) while the transfers are in
        // flight, and only then collect. The first block's readback still
        // waits for the merge loop's queued GPU work (that drain is this
        // stage's true cost), but the transfer and teardown now overlap.
        glOne.glProcessing.beginBlocksToOutputAsync();
        Log.d("ESD4D", "Stage[merge2o+submit] elapsed:" + (System.currentTimeMillis() - mergeOutT) + " ms");
        long teardownT = System.currentTimeMillis();
        AfterRun();
        Log.d("ESD4D", "Stage[afterrun] elapsed:" + (System.currentTimeMillis() - teardownT) + " ms");
        glOne.glProcessing.finishBlocksToOutputAsync();
        Output = glOne.glProcessing.mOutBuffer;
        Log.d("ESD4D", "Stage[merge2o+readback] elapsed:" + (System.currentTimeMillis() - mergeOutT) + " ms");
        com.particlesdevs.photoncamera.util.Allocator.logStage("ESD4D", "post-merge");
    }

    /**
     * Reads brightMap back to CPU. The packed rgba16f texels decode directly to
     * row-major grayscale luma (4 x-samples per texel), so reading the RGBA floats
     * in order already yields the full-width buffer. Must be called while the GL
     * context is current and before AfterRun() closes brightMap.
     */
    public FloatBuffer exportBrightMap() {
        if (brightMap == null) return null;
        brightMap.BufferLoad();
        ByteBuffer raw = brightMap.textureBuffer(new GLFormat(GLFormat.DataType.FLOAT_32, 4), true);
        raw.order(ByteOrder.nativeOrder());
        brightMapCPU = raw.asFloatBuffer();
        return brightMapCPU;
    }

    /**
     * Runs the KernelNet parameter model on the previously exported {@link #brightMapCPU}
     * (call {@link #exportBrightMap()} first). Returns half-resolution kernel params
     * as RGBA-interleaved (s1, s2, rho, 1) floats, or null if the model isn't available.
     * Takes ~40-170ms at high res; Run() calls this on a worker thread in
     * parallel with the alignment loop and collects the result before the
     * first mergeCombineWeight0 pass. Touches no GL state, so it is safe to
     * call off the GL thread.
     */
    public KernelNetResult runKernelNetInference(float sigma) {
        if (brightMapCPU == null || brightMapCPUSize == null) return null;
        var ctx = PhotonCamera.getAppContext();
        if (ctx == null) return null;
        // Process-wide shared instance (kept warm across shots): do NOT
        // close it here. runInference blocks until the model is ready.
        KernelNetNcnnProcessor processor = KernelNetNcnnProcessor.start(ctx);
        if (!processor.isReady()) return null;
        return processor.runInference(brightMapCPU, brightMapCPUSize.x, brightMapCPUSize.y, sigma);
    }

    /**
     * Uploads the KernelNet parameter map as an RGBA16F texture for the
     * anisotropic Gaussian filter: texel = (s1, s2, rho, 1). The inference
     * result already comes back as RGBA-interleaved fp16 halves in the exact
     * GL_RGBA16F layout (see {@link KernelNetNcnnProcessor}), so it is
     * uploaded with GL_HALF_FLOAT — no repack, no driver FLOAT->HALF
     * conversion. Also publishes the same buffer as {@link #kernelsMapCPU}
     * for the post pipeline. The texture is left open for downstream use;
     * caller owns it.
     */
    public GLTexture createKernelsMap(KernelNetResult result) {
        if (result == null) return null;
        int w = result.width();
        int h = result.height();
        GLTexture map = new GLTexture(new Point(w, h), new GLFormat(GLFormat.DataType.FLOAT_16, 4), null);
        // The inference result already comes back RGBA-interleaved fp16 (see
        // {@link KernelNetNcnnProcessor}), so the buffer is uploaded as-is.
        // Also publishes the same interleaved buffer as {@link #kernelsMapCPU}
        // for the post pipeline; the malloc'd base buffer rides along via
        // kernelsMapBase so the post pipeline can free it deterministically
        // once uploaded (views don't free). The texture is left open for
        // downstream use; caller owns it.
        ByteBuffer halves = result.params();
        halves.position(0);
        try {
            map.loadRawHalf(halves);
        } catch (Throwable t) {
            // Upload failed: the malloc'd result has no other owner yet.
            com.particlesdevs.photoncamera.util.Allocator.free(result.params());
            throw t;
        }
        kernelsMapCPU = halves.asShortBuffer();
        kernelsMapCPUSize = new Point(w, h);
        kernelsMapBase = halves;
        return map;
    }

    @Override
    public void close() {
        // Safety net for error paths that skip the aligner branch: never let
        // the CPU worker outlive the script against buffers the caller may
        // free (or a GL context about to be torn down). No-op after the
        // normal join in Run.
        try {
            joinHalideWorker();
        } catch (Throwable t) {
            Log.e("ESD4D", "Halide alignment worker failed during close", t);
        }
        deleteUploadFences(alterUploadFences);
        deleteUploadFences(noiseBlendUploadFences);
        super.close();
    }

    /** This script's EGL context (null before setup). */
    public GLContext getGLContext() {
        return glOne != null ? glOne.glProcessing : null;
    }

    /**
     * Probes whether a context created in this context's EGL share group can
     * see its textures - the condition for handing the drizzle accumulator and
     * the 3a detail texture to the post as texture names instead of CPU
     * ferries. Creates a probe context, checks a probe texture's name there,
     * then restores this context. Sets {@link #gpuHandoff}; false keeps the
     * ferry paths.
     */
    public boolean probeGpuHandoff() {
        gpuHandoff = false;
        int[] tex = new int[1];
        GLContext probe = null;
        try {
            GLContext self = getGLContext();
            if (self == null || self.getEGLContext() == null) return false;
            android.opengl.GLES30.glGenTextures(1, tex, 0);
            if (tex[0] == 0) return false;
            android.opengl.GLES30.glBindTexture(android.opengl.GLES30.GL_TEXTURE_2D, tex[0]);
            android.opengl.GLES30.glTexStorage2D(android.opengl.GLES30.GL_TEXTURE_2D, 1,
                    android.opengl.GLES30.GL_RGBA16F, 4, 4);
            probe = new GLContext(1, 1, self.getEGLContext());
            gpuHandoff = android.opengl.GLES30.glIsTexture(tex[0]);
        } catch (Throwable t) {
            Log.e("ESD4D", "SR GPU handoff probe failed", t);
            gpuHandoff = false;
        } finally {
            if (probe != null) {
                try {
                    probe.close();
                } catch (Exception ignored) {
                }
            }
            GLContext self = getGLContext();
            if (self != null) self.makeCurrent();
            if (tex[0] != 0) {
                android.opengl.GLES30.glDeleteTextures(1, tex, 0);
            }
        }
        Log.d("ESD4D", "SR GPU handoff: " + gpuHandoff);
        return gpuHandoff;
    }

    @Override
    public void AfterRun() {
        // The unpack staging cache is only needed while frames are uploaded
        // (this phase). Release it here so post-merge MemStage baselines are
        // unchanged vs per-upload malloc/free.
        ImageFrame.releaseUploadStaging();
        if(hotPixelBuffer != null) hotPixelBuffer.close();
        // baseDiff/alter/inputAlter/inputBase/brightMap may already be
        // released post-loop (nulled there); guard so a stale close can
        // never delete a recycled texture ID.
        if (inputAlter != null) inputAlter.close();
        if (inputAlterAlt != null) inputAlterAlt.close();
        deleteUploadFences(alterUploadFences);
        if (srAccA != null) {
            srAccA.close();
            srAccA = null;
        }
        if (srAccB != null) {
            srAccB.close();
            srAccB = null;
        }
        srAccFinal = null;
        if (srHP != null) {
            if (srHP.mTextureID != srDetailTexID) {
                srHP.close();
            }
            srHP = null;
        }
        if (srDriA != null) {
            srDriA.close();
            srDriA = null;
        }
        if (srDriB != null) {
            srDriB.close();
            srDriB = null;
        }
        if (srBayA != null) {
            srBayA.close();
            srBayA = null;
        }
        if (srBayB != null) {
            srBayB.close();
            srBayB = null;
        }
        if (srRefTex != null) {
            srRefTex.close();
            srRefTex = null;
        }
        srRefSize = null;
        if (srFlowGuided != null) {
            srFlowGuided.close();
            srFlowGuided = null;
        }
        srFlowGuidedSize = null;
        if (srLumaTex != null) {
            srLumaTex.close();
            srLumaTex = null;
        }
        srLumaSize = null;
        if (alter != null) alter.close();
        if (inputBase != null) inputBase.close();
        if (baseDiff != null) baseDiff.close();
        base.close();
        baseAlter.close();
        if (brightMap != null) brightMap.close();
        // The kernel-params texture is only read by the merge passes above;
        // downstream uses the fp32 CPU view. Free the GPU copy with the rest
        // instead of leaving it registered until context teardown.
        if (kernelsMap != null) {
            kernelsMap.close();
            kernelsMap = null;
        }
        result.close();
        if(Objects.equals(alignerSelect, "flownet") && flowNetAlignment != null) {
            // Closes flowTex (== alignmentTex), so drop the reference to avoid
            // a double close below.
            flowNetAlignment.close();
            flowNetAlignment = null;
            alignmentTex = null;
        } else {
            alignmentTex.close();
        }
        if(kernelsMap != null) kernelsMap.close();
        GLTexture.notClosed();
    }

    /**
     * Debug: runs the GL block-pyramid aligner in addition to the already
     * computed Halide atlas (alignmentTex) and logs per-frame differences
     * between the two vector fields. A constant offset reveals a packing or
     * sign-convention bug; scene-scaled differences reveal a grid/phase
     * mismatch; random differences are matcher noise.
     */
    private void logAlignerCompare(Point atlasSize, ArrayList<ImageFrame> images) {
        try {
            long t0 = System.currentTimeMillis();
            PyramidAlignment pyramidAlignment = new PyramidAlignment(atlasSize, images, glProg, glUtils, this);
            pyramidAlignment.parameters = parameters;
            pyramidAlignment.startLevel = alignmentStartLevel;
            pyramidAlignment.Run();
            Log.d("AlignerCompare", "GL pyramid reference computed in "
                    + (System.currentTimeMillis() - t0) + "ms");

            Point rawHalf = new Point(parameters.rawSize.x / 2, parameters.rawSize.y / 2);
            float[] halide = readAtlasFloats(alignmentTex, atlasSize);
            float[] glpyr = readAtlasFloats(pyramidAlignment.Result, atlasSize);
            pyramidAlignment.close();
            if (halide == null || glpyr == null) return;

            for (int f = 1; f < images.size(); f++) {
                Point shift = PyramidAlignment.alignmentShift(parameters, f);
                ArrayList<Float> dxs = new ArrayList<>();
                ArrayList<Float> dys = new ArrayList<>();
                float sdx = 0, sdy = 0;
                for (int ty = 0; ty < parameters.alignmentSize.y; ty++) {
                    for (int tx = 0; tx < parameters.alignmentSize.x; tx++) {
                        int o = ((shift.y + ty) * atlasSize.x + shift.x + tx) * 4;
                        if (o + 3 >= glpyr.length) continue;
                        float hdx = decodeDx(halide, o, rawHalf.x), hdy = decodeDy(halide, o, rawHalf.y);
                        float gdx = decodeDx(glpyr, o, rawHalf.x), gdy = decodeDy(glpyr, o, rawHalf.y);
                        dxs.add(hdx - gdx);
                        dys.add(hdy - gdy);
                        sdx += hdx - gdx;
                        sdy += hdy - gdy;
                    }
                }
                int n = dxs.size();
                // A few sample cells: center + four corners of the grid.
                StringBuilder samples = new StringBuilder();
                int[][] probe = {
                        {parameters.alignmentSize.x / 2, parameters.alignmentSize.y / 2},
                        {2, 2}, {parameters.alignmentSize.x - 3, 2},
                        {2, parameters.alignmentSize.y - 3},
                        {parameters.alignmentSize.x - 3, parameters.alignmentSize.y - 3}};
                for (int[] pc : probe) {
                    int o = ((shift.y + pc[1]) * atlasSize.x + shift.x + pc[0]) * 4;
                    if (o + 3 >= glpyr.length) continue;
                    samples.append(String.format(" [(%d,%d) h(%+.1f,%+.1f) g(%+.1f,%+.1f)]",
                            pc[0], pc[1],
                            decodeDx(halide, o, rawHalf.x), decodeDy(halide, o, rawHalf.y),
                            decodeDx(glpyr, o, rawHalf.x), decodeDy(glpyr, o, rawHalf.y)));
                }
                Log.d("AlignerCompare", String.format(
                        "frame %d: mean diff (%+.2f,%+.2f) rawHalf texels over %d cells%s",
                        f, sdx / n, sdy / n, n, samples));
            }
        } catch (Throwable tr) {
            Log.w("AlignerCompare", "comparison failed", tr);
        }
    }

    private static float decodeDx(float[] atlas, int o, float rawHalf) {
        return (float) Math.floor(atlas[o] * rawHalf + 0.5f) + atlas[o + 2];
    }

    private static float decodeDy(float[] atlas, int o, float rawHalf) {
        return (float) Math.floor(atlas[o + 1] * rawHalf + 0.5f) + atlas[o + 3];
    }

    private static float[] readAtlasFloats(GLTexture tex, Point size) {
        java.nio.ByteBuffer buf = tex.textureBuffer(tex.mFormat);
        buf.order(java.nio.ByteOrder.nativeOrder());
        java.nio.FloatBuffer fb = buf.asFloatBuffer();
        float[] out = new float[size.x * size.y * 4];
        fb.get(out);
        return out;
    }
}
