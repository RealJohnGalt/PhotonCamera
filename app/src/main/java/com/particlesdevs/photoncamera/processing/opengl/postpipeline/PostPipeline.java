package com.particlesdevs.photoncamera.processing.opengl.postpipeline;

import android.graphics.Bitmap;
import android.graphics.Point;
import android.opengl.GLES30;
import com.particlesdevs.photoncamera.util.Log;

import static android.opengl.GLES31.GL_CLAMP_TO_EDGE;
import static android.opengl.GLES31.GL_LINEAR;

import com.particlesdevs.photoncamera.api.CameraMode;
import com.particlesdevs.photoncamera.app.PhotonCamera;
import com.particlesdevs.photoncamera.processing.ImageFrame;
import com.particlesdevs.photoncamera.processing.opengl.GLBasePipeline;
import com.particlesdevs.photoncamera.processing.opengl.GLCoreBlockProcessing;
import com.particlesdevs.photoncamera.processing.opengl.GLDrawParams;
import com.particlesdevs.photoncamera.processing.opengl.GLFormat;
import com.particlesdevs.photoncamera.processing.opengl.GLImage;
import com.particlesdevs.photoncamera.processing.opengl.GLInterface;
import com.particlesdevs.photoncamera.processing.opengl.GLProg;
import com.particlesdevs.photoncamera.processing.opengl.GLTexture;
import com.particlesdevs.photoncamera.processing.opengl.nodes.Node;
import com.particlesdevs.photoncamera.processing.opengl.postpipeline.RotateWatermark;
import com.particlesdevs.photoncamera.processing.parameters.ResolutionSolution;
import com.particlesdevs.photoncamera.processing.render.NoiseModeler;
import com.particlesdevs.photoncamera.processing.render.Parameters;
import com.particlesdevs.photoncamera.processing.ultrahdr.GainMapComputer;
import com.particlesdevs.photoncamera.settings.annotations.Tunable;
import com.particlesdevs.photoncamera.util.Allocator;
import com.particlesdevs.photoncamera.util.BufferUtils;

import java.nio.ByteBuffer;
import java.util.ArrayList;

public class PostPipeline extends GLBasePipeline {
    public ByteBuffer stackFrame;
    public ByteBuffer lowFrame;
    public ByteBuffer highFrame;
    public GLTexture FusionMap;
    public GLTexture GainMap;
    public ArrayList<Bitmap> debugData = new ArrayList<>();
    public ArrayList<ImageFrame> SAGAIN;
    public Point cropSize;
    public float[] analyzedBL = new float[]{0.f, 0.f, 0.f};
    float regenerationSense = 1.f;
    float totalGain = 1.f;
    float AecCorr = 1.f;
    float fusionGain = 1.f;
    float softLight = 1.f;

    // Ultra HDR (cheap second pass) state.
    public boolean hdrOutput = false;
    /** When true, the first run captures the linear (post-demosaic) buffer so the
     *  second pass can re-run only color + tone. */
    public boolean captureDemosaic = false;
    private boolean mCaptured = false;
    /** CPU copy of the post-demosaic linear buffer (survives GLTexture.closeAll). */
    public ByteBuffer demosaicLinear;
    public Point demosaicLinearSize;
    /** CPU copies of the per-image GainMap / FusionMap used by Initial. */
    public ByteBuffer gainMapBuf;
    public ByteBuffer fusionMapBuf;
    public Point gainMapSize;
    public Point fusionMapSize;

    public PostPipeline() {
        super("PostPipeline");
    }

    public int getRotation() {
        int rotation = mParameters.cameraRotation;
        String TAG = "ParseExif";
        Log.d(TAG, "Gravity rotation:" + PhotonCamera.getGravity().getRotation());
        Log.d(TAG, "Sensor rotation:" + PhotonCamera.getCaptureController().mSensorOrientation);
        return rotation;
    }

    @SuppressWarnings("SuspiciousNameCombination")
    private Point getRotatedCoords(Point in) {
        switch (getRotation()) {
            case 0:
            case 180:
                return in;
            case 90:
            case 270:
                return new Point(in.y, in.x);
        }
        return in;
    }

    float constShift = 0.0f;

    @Tunable(
        title = "Demosaicing Method",
        description = "0 = Demosaic (compatibility mode), 1 = Demosaic3 (better quality)",
        category = "Demosaic",
        min = 0.0f,
        max = 1.0f,
        defaultValue = 1.0f,
        step = 1.0f
    )
    int demosaicingMethod = 1;

    private void computeNoise(Parameters parameters) {
        NoiseModeler modeler = parameters.noiseModeler;
        noiseS = modeler.computeModel[0].first.floatValue() +
                modeler.computeModel[1].first.floatValue() +
                modeler.computeModel[2].first.floatValue();
        noiseO = modeler.computeModel[0].second.floatValue() +
                modeler.computeModel[1].second.floatValue() +
                modeler.computeModel[2].second.floatValue();
        noiseS /= 3.f;
        noiseO /= 3.f;
        double noisempy = Math.pow(2.0, mSettings.noiseRstr + constShift);
        Log.d("PostPipeline", "noisempy:" + noisempy);
        noiseS *= noisempy;
        noiseO *= noisempy;
        Log.d("PostPipeline", "NoiseS:" + noiseS + "\n" + "NoiseO:" + noiseO);
        noiseO = Math.max(noiseO, 1.0f/4096.0f);
        noiseS = Math.max(noiseS, Float.MIN_NORMAL);
    }

    public Bitmap Run(ByteBuffer inBuffer, Parameters parameters) {
        mParameters = parameters;
        mSettings = PhotonCamera.getSettings();
        workSize = new Point(mParameters.rawSize.x, mParameters.rawSize.y);
        computeNoise(parameters);
        captureDemosaic = mSettings.ultraHdr;
        mCaptured = false;
        Point rawSliced = parameters.rawSize;
        cropSize = new Point(parameters.rawSize);
        if (PhotonCamera.getSettings().aspect169) {
            if (rawSliced.x > rawSliced.y) {
                rawSliced = new Point(rawSliced.x, rawSliced.x * 9 / 16);
            } else {
                rawSliced = new Point(rawSliced.y * 9 / 16, rawSliced.y);
            }
            cropSize =  new Point(rawSliced);
        }
        Point rotatedSize = getRotatedCoords(rawSliced);
        if (PhotonCamera.getSettings().energySaving || mParameters.rawSize.x * mParameters.rawSize.y < ResolutionSolution.smallRes) {
            GLDrawParams.TileSize = 8;
        } else {
            GLDrawParams.TileSize = 256;
        }
        GLFormat format = new GLFormat(GLFormat.DataType.SIMPLE_8, 4);
        GLImage output = new GLImage(rotatedSize, format, false);
        GLCoreBlockProcessing glproc = new GLCoreBlockProcessing(rotatedSize, output, format, GLDrawParams.Allocate.Direct);
        glint = new GLInterface(glproc);
        stackFrame = inBuffer;
        glint.parameters = parameters;

        // Inject tunable values for PostPipeline (since it doesn't extend Node)
        com.particlesdevs.photoncamera.settings.TunableInjector.inject(this);

        BuildDefaultPipeline();
        GLImage resImg = runAll();
        Bitmap res = resImg.getBufferedImage();
        Allocator.free(resImg.byteBuffer);

        // Retain the GainMap / FusionMap (and the linear buffer is already
        // captured inside Initial) so the cheap HDR pass can reuse them.
        if (captureDemosaic) {
            retainGainMap();
            retainFusionMap();
        }

        GLTexture.closeAll();
        return res;
    }

    private void retainGainMap() {
        if (GainMap == null) return;
        GainMap.BindBuffer();
        gainMapBuf = GainMap.textureBuffer(new GLFormat(GLFormat.DataType.FLOAT_16, 4), true);
        gainMapBuf.rewind();
        gainMapSize = new Point(GainMap.mSize.x, GainMap.mSize.y);
    }

    private void retainFusionMap() {
        if (FusionMap == null) return;
        FusionMap.BindBuffer();
        fusionMapBuf = FusionMap.textureBuffer(new GLFormat(GLFormat.DataType.FLOAT_16, 4), true);
        fusionMapBuf.rewind();
        fusionMapSize = new Point(FusionMap.mSize.x, FusionMap.mSize.y);
    }

    /** Called from Initial.Run (first pass) to keep the linear scene buffer. */
    public void captureDemosaicLinear(GLTexture tex) {
        if (mCaptured || demosaicLinear != null) return;
        tex.BindBuffer();
        demosaicLinear = tex.textureBuffer(new GLFormat(GLFormat.DataType.FLOAT_16, 4), true);
        demosaicLinear.rewind();
        demosaicLinearSize = new Point(tex.mSize.x, tex.mSize.y);
        mCaptured = true;
    }

    /**
     * Cheap second run: reuses the already-demosaiced linear buffer (captured in
     * {@link #Run}) plus the GainMap/FusionMap, and re-runs only the color + tone
     * stages (Initial in HDR mode, AutoExposure, rotate) to produce a linear HDR
     * rendition. The expensive Bayer2Float / fusion / demosaic / denoise / ABLC
     * stages are skipped entirely.
     *
     * @param sdr  the SDR (display-encoded) base rendition, already computed by {@link #Run}
     * @param down downsample factor per axis for the gain map (gain map is 1/down^2 the size)
     * @param scale total log2 range the gain-map shader covers; passed through to {@link GainMapComputer}
     * @return the raw RGBA8 gain map plus its dimensions and the scale used
     */
    public GainMapRaw RunHDRGainMap(ByteBuffer inBuffer, Parameters parameters, Bitmap sdr, int down, float scale) {
        if (demosaicLinear == null) {
            throw new IllegalStateException("No demosaiced linear buffer captured; Run() must run first with ultraHdr enabled");
        }
        mParameters = parameters;
        mSettings = PhotonCamera.getSettings();
        workSize = new Point(mParameters.rawSize.x, mParameters.rawSize.y);
        computeNoise(parameters);
        hdrOutput = true;
        captureDemosaic = false;
        Point rawSliced = parameters.rawSize;
        cropSize = new Point(parameters.rawSize);
        if (PhotonCamera.getSettings().aspect169) {
            if (rawSliced.x > rawSliced.y) {
                rawSliced = new Point(rawSliced.x, rawSliced.x * 9 / 16);
            } else {
                rawSliced = new Point(rawSliced.y * 9 / 16, rawSliced.y);
            }
            cropSize =  new Point(rawSliced);
        }
        Point rotatedSize = getRotatedCoords(rawSliced);
        // The gain-map shader samples the SDR base and the HDR render at identical
        // texel coordinates. Any size/orientation mismatch displaces the boost
        // field from the scene; fail loudly -> caller falls back to SDR JPEG.
        if (sdr.getWidth() != rotatedSize.x || sdr.getHeight() != rotatedSize.y) {
            throw new IllegalStateException("SDR/HDR size mismatch: sdr="
                    + sdr.getWidth() + "x" + sdr.getHeight()
                    + " hdr=" + rotatedSize.x + "x" + rotatedSize.y);
        }
        GLFormat format = new GLFormat(GLFormat.DataType.FLOAT_32, 4);
        GLImage output = new GLImage(rotatedSize, format, false);
        GLCoreBlockProcessing glproc = new GLCoreBlockProcessing(rotatedSize, output, format, GLDrawParams.Allocate.Direct);
        glint = new GLInterface(glproc);
        stackFrame = inBuffer;
        glint.parameters = parameters;

        com.particlesdevs.photoncamera.settings.TunableInjector.inject(this);

        // Defensive: the replayed linear buffer must match the pipeline input size,
        // otherwise the passthrough samples out of bounds and the HDR rendition is
        // garbage.
        if (demosaicLinearSize == null
                || demosaicLinearSize.x != workSize.x || demosaicLinearSize.y != workSize.y) {
            throw new IllegalStateException("Linear buffer size " + demosaicLinearSize
                    + " does not match workSize " + workSize
                    + "; cannot replay cheap HDR pass");
        }

        try {
            // Restore the GainMap / FusionMap that Bayer2Float / ExposureFusionBayer2
            // would have built, otherwise Initial's tonemap diverges and ~35% of the
            // HDR rendition is invalid (see restoreHdrMaps()).
            restoreHdrMaps();
            BuildHdrPipeline();
            GLImage hdrOut = runAll();
            // runAll() returns the full-frame linear HDR rendition. Read it back as
            // a texture (not from the scratch renderbuffer) so the gain map sees
            // the complete image.
            GLTexture hdrTex = new GLTexture(hdrOut);
            GLImage sdrImage = new GLImage(sdr);
            GLTexture sdrTex = new GLTexture(sdrImage);

            int gw = Math.max(1, rotatedSize.x / down);
            int gh = Math.max(1, rotatedSize.y / down);
            GLTexture outTex = new GLTexture(new Point(gw, gh), new GLFormat(GLFormat.DataType.SIMPLE_8, 4));
            outTex.BufferLoad();

            GLProg prog = glint.glProgram;
            prog.useAssetProgram("ultrahdr/gainmap");
            prog.setTexture("InputBuffer", sdrTex);
            prog.setTexture("HDRBuffer", hdrTex);
            prog.setVar("uDown", down);
            prog.setVar("uScale", scale);
            prog.setVar("uEps", GainMapComputer.epsilon());
            GLES30.glDisable(GLES30.GL_SCISSOR_TEST);
            GLES30.glBindFramebuffer(GLES30.GL_FRAMEBUFFER, outTex.mBuffer);
            GLES30.glViewport(0, 0, gw, gh);
            prog.draw();
            int drawErr = GLES30.glGetError();
            if (drawErr != GLES30.GL_NO_ERROR) {
                Log.e("PostPipeline", "GL error 0x" + Integer.toHexString(drawErr) + " in gain map pass");
            }
            ByteBuffer gm = outTex.textureBuffer(new GLFormat(GLFormat.DataType.SIMPLE_8, 4), false);
            gm.position(0);

            sdrTex.close();
            hdrTex.close();
            outTex.close();
            return new GainMapRaw(gm, gw, gh, scale);
        } finally {
            // Ensure subsequent SDR runs see hdrOutput=false.
            hdrOutput = false;
            GLTexture.closeAll();
        }
    }

    /**
     * Rebuilds the {@code GainMap}/{@code FusionMap} that the cheap HDR pass would
     * otherwise be missing: those textures are created only by Bayer2Float and
     * ExposureFusionBayer2, which BuildHdrPipeline skips. closeAll() deleted the
     * first pass's GL textures but left the Java references dangling, so they are
     * nulled first - otherwise Initial binds a deleted texture and its tonemap
     * diverges, leaving ~35% of the HDR rendition invalid.
     */
    private void restoreHdrMaps() {
        GainMap = null;
        FusionMap = null;
        // GainMap is exactly mParameters.gainMap uploaded (see Bayer2Float) - the
        // canonical source, so it matches the full pass byte-for-byte.
        if (mParameters.gainMap != null && mParameters.mapSize != null) {
            GainMap = new GLTexture(mParameters.mapSize,
                    new GLFormat(GLFormat.DataType.FLOAT_16, 4),
                    BufferUtils.getFrom(mParameters.gainMap), GL_LINEAR, GL_CLAMP_TO_EDGE);
        }
        // FusionMap is computed from fusion internals with no Parameters source, so
        // replay the texture captured from the first pass. Keying off the captured
        // buffer mirrors Initial, which defines FUSION solely when
        // basePipeline.FusionMap != null.
        if (fusionMapBuf != null && fusionMapSize != null) {
            fusionMapBuf.rewind();
            FusionMap = new GLTexture(fusionMapSize,
                    new GLFormat(GLFormat.DataType.FLOAT_16, 4),
                    fusionMapBuf, GL_LINEAR, GL_CLAMP_TO_EDGE);
        }
    }

    public static class GainMapRaw {
        public final ByteBuffer buffer;
        public final int w;
        public final int h;
        public final float scale;
        GainMapRaw(ByteBuffer buffer, int w, int h, float scale) {
            this.buffer = buffer;
            this.w = w;
            this.h = h;
            this.scale = scale;
        }
    }

    private void BuildDefaultPipeline() {
        boolean nightMode = PhotonCamera.getSettings().selectedMode == CameraMode.NIGHT;
        add(new Bayer2Float());
        add(new ExposureFusionBayer2());
        switch (PhotonCamera.getSettings().cfaPattern) {
            case -2: {
                add(new DemosaicQUAD());
                break;
            }
            case 4: {
                add(new MonoDemosaic());
                break;
            }
            default: {
                if (PhotonCamera.getSettings().hdrxNR) {
                }

                if (mSettings.alignAlgorithm != 2) {
                    switch (demosaicingMethod) {
                        case 0:
                            add(new Demosaic());
                            break;
                        default:
                            add(new Demosaic3());
                            break;
                    }
                }
                if (PhotonCamera.getSettings().hdrxNR) {
                    add(new ESD3D2(true));
                }
                break;
            }
        }
        add(new ABLC());
        add(new Initial());
        add(new AutoExposure());
        add(new CaptureSharpening());
        add(new CorrectingFlow());
        add(new Sharpen2());
        add(new RotateWatermark(getRotation()));
    }

    /**
     * Cheap second pass for Ultra HDR: replays the already-captured post-demosaic
     * linear buffer ({@link #demosaicLinear}) through color + tone (Initial in HDR
     * mode) and the same post-Initial chain as the full pipeline, ending at
     * rotation. The expensive Bayer / demosaic / fusion / denoise / ABLC stages
     * are skipped entirely.
     *
     * <p>{@code basePipeline.main1/main2/main3} were deleted (but not nulled) by
     * {@link GLTexture#closeAll()} at the end of the first {@link #Run}, so they
     * must be recreated unconditionally here - a null-check would reuse the dead
     * GL textures and yield a garbage HDR rendition.
     */
    private void BuildHdrPipeline() {
        add(new DemosaicSourceNode());
        add(new Initial());
        add(new AutoExposure());
        add(new CaptureSharpening());
        add(new CorrectingFlow());
        add(new Sharpen2());
        add(new RotateWatermark(getRotation()));
    }

    /** First node of the cheap HDR pipeline: feeds the captured linear buffer forward. */
    private class DemosaicSourceNode extends Node {
        DemosaicSourceNode() {
            super("ultrahdr/source", "DemosaicSource");
        }

        @Override
        public void Run() {
            // Recreate the main ping-pong textures unconditionally (see BuildHdrPipeline).
            basePipeline.main1 = new GLTexture(basePipeline.workSize,
                    new GLFormat(GLFormat.DataType.FLOAT_16, GLDrawParams.WorkDim),
                    null, GL_LINEAR, GL_CLAMP_TO_EDGE);
            basePipeline.main2 = new GLTexture(basePipeline.workSize,
                    new GLFormat(GLFormat.DataType.FLOAT_16, GLDrawParams.WorkDim),
                    null, GL_LINEAR, GL_CLAMP_TO_EDGE);
            basePipeline.main3 = new GLTexture(basePipeline.workSize,
                    new GLFormat(GLFormat.DataType.FLOAT_16, GLDrawParams.WorkDim),
                    null, GL_LINEAR, GL_CLAMP_TO_EDGE);
            GLTexture src = new GLTexture(demosaicLinearSize,
                    new GLFormat(GLFormat.DataType.FLOAT_16, 4), demosaicLinear);
            glProg.setTexture("InputBuffer", src);
            WorkingTexture = basePipeline.getMain();
            glProg.drawBlocks(WorkingTexture);
            src.close();
            glProg.closed = true;
        }
    }
}
