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

    @Tunable(title = "SR post-upscale detail", category = "Upscale", description = "Gain of the post-upscale SR detail top-up (base injection in merge2o stays on its own strength)", min = 0.0f, max = 2.0f, step = 0.05f, defaultValue = 0.6f)
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
        pp.srDetailTexID = 0;
    }

    // Program bound by prepare() for the head driver's per-band phase switch.
    int tileProgram = 0;
    // Frozen band-body uniforms (set in prepare, re-issued by renderTile).
    private float rawPerOutX, rawPerOutY, srOx, srOy;
    private int srCfaX, srCfaY;

    @Override
    public void Run() {
        GLTexture input = previousNode.WorkingTexture;
        PostPipeline pp = (PostPipeline) basePipeline;
        if (pp.headFused) {
            // Head produce already streamed this node's bands into the head
            // output main (see TileDriver.runHeadProduce).
            WorkingTexture = input;
            glProg.closed = true;
            return;
        }
        if (!prepare(input != null ? input.mSize : null)) {
            return; // passthrough; prepare set WorkingTexture
        }
        renderTile(input, tileActive() ? tileOut : basePipeline.getMain(), 0,
                tileActive() ? tileY0 : 0);
        glProg.closed = true;
    }

    /**
     * One-time setup shared by the legacy Run and the head driver: adopts or
     * uploads the detail map, binds the program (capturing {@link #tileProgram})
     * and freezes the band-body uniforms. {@code outSize} is the full output
     * size the raw-per-output ratio is measured against - the driver calls
     * this before any band exists, so it cannot read it off the input.
     * Returns false when the node passes through, with WorkingTexture already
     * set to the input. Idempotent: a second call reuses the detail texture.
     */
    boolean prepare(Point outSize) {
        if (detailTex != null) {
            return true;
        }
        PostPipeline pp = (PostPipeline) basePipeline;
        ShortBuffer params = pp.srDetail;
        Point paramsSize = pp.srDetailSize;
        ByteBuffer ownedBase = pp.srDetailBase;
        int sharedDetail = pp.srDetailTexID;
        Parameters p = basePipeline.mParameters;

        boolean ok = outSize != null && outSize.x > 0 && outSize.y > 0
                && paramsSize != null
                && (sharedDetail != 0 || (params != null && ownedBase != null))
                && paramsSize.x > 0 && paramsSize.y > 0
                && srPostStrength > 0f && p != null && p.rawSize != null
                && p.rawSize.x > 0 && p.rawSize.y > 0
                // Full-SR path replaces this content downstream: stand down
                // (releasing our own ferry) instead of rendering discarded work.
                && pp.srFullCPU == null && pp.srFullTexID == 0;
        if (!ok) {
            releaseFerry(pp);
            WorkingTexture = previousNode.WorkingTexture;
            return false;
        }

        // Full-frame raw domain reference: cropped shots map through the
        // crop origin, uncropped shots are identity. Matches the merge00
        // packing convention (packed = (cropRaw + cfaShift) / 2).
        float fullW = (p.isCropped && p.fullRawSize != null && p.fullRawSize.x > 0)
                ? p.fullRawSize.x : p.rawSize.x;
        float fullH = (p.isCropped && p.fullRawSize != null && p.fullRawSize.y > 0)
                ? p.fullRawSize.y : p.rawSize.y;
        srOx = (p.isCropped && p.cropOrigin != null) ? p.cropOrigin.x : 0f;
        srOy = (p.isCropped && p.cropOrigin != null) ? p.cropOrigin.y : 0f;
        // Output spans the full frame (zoom expand) or the raw buffer, so
        // raw-per-output is the full size over the output size. Rotation is
        // handled downstream; both domains here are unrotated.
        rawPerOutX = fullW / outSize.x;
        rawPerOutY = fullH / outSize.y;

        if (sharedDetail != 0) {
            if (!android.opengl.GLES30.glIsTexture(sharedDetail)) {
                Log.e(Name, "SR detail: shared map " + sharedDetail + " invisible, passing through");
                releaseFerry(pp);
                WorkingTexture = previousNode.WorkingTexture;
                return false;
            }
            // Shared-group handoff: adopt the merge's detail texture; no
            // upload, no ferry. closeAll()/AfterRun delete it here.
            detailTex = new GLTexture(sharedDetail, paramsSize,
                    new GLFormat(GLFormat.DataType.FLOAT_16, 4));
        } else {
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
                WorkingTexture = previousNode.WorkingTexture;
                return false;
            }
        }
        // CPU copy served (now on GPU): release so it doesn't ride the render.
        freeBase(pp.srDetailBase);
        pp.srDetailBase = null;
        pp.srDetail = null;
        pp.srDetailSize = null;

        // Exact packing shift from the merge that built the map (never
        // re-derived: quad/mono layouts share the same ferry contract).
        android.graphics.Point shift = pp.srDetailShift;
        srCfaX = shift != null ? shift.x : 0;
        srCfaY = shift != null ? shift.y : 0;

        // Packed-texel bounds of the base domain, for the shader's edge clamp
        // reference only (sampling itself clamps to the map).
        glProg.useAssetProgram("srdetail/apply");
        tileProgram = glProg.mCurrentProgramActive;
        return true;
    }

    /**
     * Band body shared by the legacy Run and the head driver: rebinds the
     * program, re-issues every uniform/texture (a rebind clears unit
     * assignments) and draws the output band. {@code inOriginY} is the
     * absolute output row the input texture's row 0 maps to;
     * {@code outOriginY} the same for the draw target.
     */
    void renderTile(GLTexture inTile, GLTexture outTile, int inOriginY, int outOriginY) {
        glProg.rebindProgram(tileProgram);
        glProg.setTexture("InputBuffer", inTile);
        glProg.setTexture("DetailMap", detailTex);
        glProg.setVar("srRawPerOut", rawPerOutX, rawPerOutY);
        glProg.setVar("srOrigin", srOx, srOy);
        glProg.setVar("srCfa", srCfaX, srCfaY);
        glProg.setVar("srStrength", srPostStrength);
        glProg.setVar("u_inOrigin", 0, inOriginY);
        glProg.setVar("u_tileOrigin", 0, outOriginY);
        WorkingTexture = outTile;
        glProg.drawBlocks(outTile);
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
