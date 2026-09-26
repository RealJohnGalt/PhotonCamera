package com.particlesdevs.photoncamera.processing.opengl.postpipeline;

import android.graphics.Point;

import com.particlesdevs.photoncamera.processing.opengl.GLFormat;
import com.particlesdevs.photoncamera.processing.opengl.GLTexture;
import com.particlesdevs.photoncamera.processing.opengl.nodes.Node;
import com.particlesdevs.photoncamera.processing.render.Parameters;
import com.particlesdevs.photoncamera.settings.annotations.Tunable;
import com.particlesdevs.photoncamera.util.Allocator;
import com.particlesdevs.photoncamera.util.Log;

import java.nio.ByteBuffer;
import java.nio.ShortBuffer;

import static android.opengl.GLES20.GL_CLAMP_TO_EDGE;
import static android.opengl.GLES20.GL_LINEAR;

/**
 * Applies the merge-stage SR detail highpass at output size, right after the
 * guided reconstruction. The base injection inside {@code merge2o} is
 * smoothed again by the upscale itself; this top-up survives by construction
 * (smoothly upsampled map, per-site add in linear pre-tonemap light).
 *
 * <p>Null ferry (single frame, no upscale, export failure) renders as a
 * passthrough and frees the ferry exactly once, mirroring the KernelNet
 * params lifecycle in {@link UpscaleCrop}.</p>
 */
public final class SRDetailApply extends Node {

    @Tunable(title = "SR post-upscale detail", category = "Upscale", description = "Gain of the post-upscale SR detail top-up (base injection in merge2o stays on its own strength)", min = 0.0f, max = 2.0f, step = 0.05f, defaultValue = 1.0f)
    float srPostStrength;

    private GLTexture detailTex;

    public SRDetailApply() {
        super("", "SRDetailApply");
    }

    @Override
    public void Compile() {
    }

    @Override
    public int halo() {
        // Bilinear detail fetch reads neighbor packed texels across band
        // edges; the driver skirts tiles per this contract.
        return 1;
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

    private void releaseFerry(PostPipeline pp) {
        if (pp == null) return;
        freeBase(pp.srDetailBase);
        pp.srDetailBase = null;
        pp.srDetail = null;
        pp.srDetailSize = null;
    }

    @Override
    public void Run() {
        GLTexture input = previousNode.WorkingTexture;
        PostPipeline pp = (PostPipeline) basePipeline;
        ShortBuffer params = pp.srDetail;
        Point paramsSize = pp.srDetailSize;
        ByteBuffer ownedBase = pp.srDetailBase;
        Parameters p = basePipeline.mParameters;

        boolean ok = input != null && params != null && paramsSize != null && ownedBase != null
                && paramsSize.x > 0 && paramsSize.y > 0 && input.mSize.x > 0 && input.mSize.y > 0
                && srPostStrength > 0f && p != null && p.rawSize != null
                && p.rawSize.x > 0 && p.rawSize.y > 0;
        if (!ok) {
            releaseFerry(pp);
            WorkingTexture = input;
            return;
        }

        // Full-frame raw domain reference: cropped shots map through the
        // crop origin, uncropped shots are identity. Matches the merge00
        // packing convention (packed = (cropRaw + cfaShift) / 2).
        float fullW = (p.isCropped && p.fullRawSize != null && p.fullRawSize.x > 0)
                ? p.fullRawSize.x : p.rawSize.x;
        float fullH = (p.isCropped && p.fullRawSize != null && p.fullRawSize.y > 0)
                ? p.fullRawSize.y : p.rawSize.y;
        float ox = (p.isCropped && p.cropOrigin != null) ? p.cropOrigin.x : 0f;
        float oy = (p.isCropped && p.cropOrigin != null) ? p.cropOrigin.y : 0f;
        // Output spans the full frame (zoom expand) or the raw buffer, so
        // raw-per-output is the full size over the input size. Rotation is
        // handled downstream; both domains here are unrotated.
        float rawPerOutX = fullW / input.mSize.x;
        float rawPerOutY = fullH / input.mSize.y;

        detailTex = new GLTexture(paramsSize,
                new GLFormat(GLFormat.DataType.FLOAT_16, 4), null, GL_LINEAR, GL_CLAMP_TO_EDGE);
        try {
            ownedBase.position(0);
            detailTex.loadRawHalf(ownedBase);
        } catch (Throwable t) {
            Log.e(Name, "SR detail upload failed, passing through", t);
            try {
                detailTex.close();
            } catch (Exception ignored) {
            }
            detailTex = null;
            releaseFerry(pp);
            WorkingTexture = input;
            return;
        }
        // CPU copy served (now on GPU): release so it doesn't ride the render.
        freeBase(pp.srDetailBase);
        pp.srDetailBase = null;
        pp.srDetail = null;
        pp.srDetailSize = null;

        // Packed-texel bounds of the base domain, for the shader's edge clamp
        // reference only (sampling itself clamps to the map).
        glProg.useAssetProgram("srdetail/apply");
        glProg.setTexture("InputBuffer", input);
        glProg.setTexture("DetailMap", detailTex);
        glProg.setVar("srRawPerOut", rawPerOutX, rawPerOutY);
        glProg.setVar("srOrigin", ox, oy);
        // Exact packing shift from the merge that built the map (never
        // re-derived: quad/mono layouts share the same ferry contract).
        android.graphics.Point shift = pp.srDetailShift;
        if (shift != null) {
            glProg.setVar("srCfa", shift.x, shift.y);
        } else {
            glProg.setVar("srCfa", 0, 0);
        }
        glProg.setVar("srStrength", srPostStrength);
        glProg.setVar("u_tileOrigin", 0, tileActive() ? tileY0 : 0);
        WorkingTexture = tileActive() ? tileOut : basePipeline.getMain();
        glProg.drawBlocks(WorkingTexture);
        glProg.closed = true;
    }

    @Override
    public void AfterRun() {
        if (detailTex != null) {
            try {
                detailTex.close();
            } catch (Exception ignored) {
            }
            detailTex = null;
        }
    }
}
