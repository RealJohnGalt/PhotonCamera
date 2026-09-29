package com.particlesdevs.photoncamera.processing.opengl.postpipeline;

import android.graphics.Point;

import com.particlesdevs.photoncamera.processing.opengl.GLFormat;
import com.particlesdevs.photoncamera.processing.opengl.GLTexture;
import com.particlesdevs.photoncamera.processing.opengl.nodes.Node;
import com.particlesdevs.photoncamera.settings.annotations.Tunable;
import com.particlesdevs.photoncamera.util.Allocator;
import com.particlesdevs.photoncamera.util.Log;

import java.nio.ByteBuffer;
import java.nio.ShortBuffer;

import static android.opengl.GLES20.GL_CLAMP_TO_EDGE;
import static android.opengl.GLES20.GL_NEAREST;

/**
 * Resolves the merge-stage full-SR drizzle into the chain.
 *
 * <p>The drizzle fuses the burst's per-frame cross-channel (demosaiced) luma
 * onto the output grid. The aniso reconstruction that UpscaleCrop rendered as
 * this node's input carries the clean color and resolves the scene's low
 * band; the drizzle carries the upper band its interpolation rolls off, plus
 * the burst's sub-pixel detail. This node therefore injects only that band -
 * {@code ref + blend * hp(yFused) * shrink} with a 5x5 box high-pass - so the
 * shrinkage acts where the signal-to-noise is high instead of fighting the
 * low band, where the two estimates agree and only noise differs.</p>
 *
 * <p>Null ferry renders as a passthrough and frees defensively.</p>
 */
public final class SRResolve extends Node {

    @Tunable(title = "SR resolve blend", category = "Upscale", description = "Weight of the drizzle detail injected into the demosaiced aniso reconstruction (0 = pure aniso, 1 = full)", min = 0.0f, max = 1.0f, step = 0.05f, defaultValue = 1.0f)
    float srResolveBlend = 1.0f;

    @Tunable(title = "SR resolve acutance", category = "Upscale", description = "How much of the aniso's edge-gated acutance term is added to the resolved output. This is the legitimate half of the Disabled render's edge sharpening (applied on elongated-kernel edges, ~0 on flats and isotropic textures), handed over in the reference's alpha so the SR output matches the native render's edge crispness without inheriting the KernelNet band - and the quantization mesh that lives in it", min = 0.0f, max = 2.0f, step = 0.05f, defaultValue = 1.0f)
    float srResolveAcutance = 1.0f;

    @Tunable(title = "SR resolve detail scale", category = "Upscale", description = "Injected high-band detail is Wiener-shrunk toward zero below this multiple of the modelled fused-noise sigma; raise to suppress noise, lower for more faint texture, 0 = no shrinkage. Lowered from 3.0 after measuring that a high threshold passes the large |d| at high-contrast edges (where aliasing lives) while suppressing the small |d| of real fine texture - i.e. it selected the wrong content. Simulated at the shipped kernel: 3.0 -> 2.0 improves the edge error 0.0176 -> 0.0164, cuts the staircase signature 0.641 -> 0.327 and raises fine detail, for noise 0.00024 -> 0.00131 (0.33 levels)", min = 0.0f, max = 10.0f, step = 0.1f, defaultValue = 0.6f)
    float srResolveDetail = 0.6f;

    private GLTexture accumTex;

    public SRResolve() {
        super("", "SRResolve");
    }

    @Override
    public void Compile() {
    }

    @Override
    public int halo() {
        // The mesh-proof fallback reads a 3x3 stencil of both the accumulator
        // and the reference (the accumulator is a full texture; the reference
        // is read at the tile's own texels plus one).
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
        freeBase(pp.srFullBase);
        pp.srFullBase = null;
        pp.srFullCPU = null;
        pp.srFullSize = null;
        pp.srFullTexID = 0;
    }

    @Override
    public void Run() {
        GLTexture input = previousNode.WorkingTexture;
        PostPipeline pp = (PostPipeline) basePipeline;
        ShortBuffer acc = pp.srFullCPU;
        Point accSize = pp.srFullSize;
        ByteBuffer ownedBase = pp.srFullBase;
        int sharedAcc = pp.srFullTexID;

        // The aniso input is the color/level reference: without it (or without
        // the drizzle) there is nothing to resolve.
        boolean ok = input != null && accSize != null
                && accSize.x > 0 && accSize.y > 0
                && (sharedAcc != 0 || (acc != null && ownedBase != null));
        if (!ok) {
            Log.d(Name, "SR resolve passthrough: acc=" + (acc != null)
                    + " shared=" + sharedAcc
                    + " size=" + accSize + " base=" + (ownedBase != null)
                    + " input=" + (input != null));
            releaseFerry(pp);
            WorkingTexture = input;
            return;
        }
        if (sharedAcc != 0 && !android.opengl.GLES30.glIsTexture(sharedAcc)) {
            // The shared name is not visible here (the EGL group silently fell
            // back to unshared): skip the SR rather than read garbage.
            Log.e(Name, "SR resolve: shared accumulator " + sharedAcc + " invisible, skipping");
            releaseFerry(pp);
            WorkingTexture = input;
            return;
        }

        if (sharedAcc != 0) {
            // Shared-group handoff: adopt the merge's accumulator texture; no
            // upload, no ferry. closeAll() deletes it in this context.
            accumTex = new GLTexture(sharedAcc, accSize,
                    new GLFormat(GLFormat.DataType.UNSIGNED_32, 1));
        } else {
            accumTex = new GLTexture(accSize,
                    new GLFormat(GLFormat.DataType.UNSIGNED_32, 1), null, GL_NEAREST, GL_CLAMP_TO_EDGE);
            try {
                ownedBase.position(0);
                accumTex.loadRawUint(ownedBase);
            } catch (Throwable t) {
                Log.e(Name, "SR accum upload failed, passing through", t);
                releaseFerry(pp);
                closeTextures();
                WorkingTexture = input;
                return;
            }
        }
        // CPU copies served (now on GPU): release so they don't ride the render.
        releaseFerry(pp);
        Log.d(Name, "SR resolve active: " + accSize.x + "x" + accSize.y);

        glProg.useAssetProgram("srresolve/normalize");
        glProg.setTexture("srAccum", accumTex);
        glProg.setTexture("InputBuffer", input);
        glProg.setVar("u_tileOrigin", 0, tileActive() ? tileY0 : 0);
        glProg.setVar("srBlend", srResolveBlend);
        // Pre-inflation model: the fused luma's noise does not scale with the
        // denoise slider, so the shrinkage threshold must not either.
        float nS = pp.noiseS0 > 0f ? pp.noiseS0 : basePipeline.noiseS;
        float nO = pp.noiseO0 > 0f ? pp.noiseO0 : basePipeline.noiseO;
        glProg.setVar("srNoiseS", nS);
        glProg.setVar("srNoiseO", nO);
        glProg.setVar("srDetail", srResolveDetail);
        glProg.setVar("srAcutance", srResolveAcutance);
        float[] ablc = pp.ablcBlack;
        glProg.setVar("srBlack", ablc != null && ablc.length >= 3
                ? new float[]{ablc[0], ablc[1], ablc[2]}
                : new float[]{0f, 0f, 0f});
        WorkingTexture = tileActive() ? tileOut : basePipeline.getMain();
        glProg.drawBlocks(WorkingTexture);
        glProg.closed = true;
        pp.srResolved = true;
        // The accumulator feeds this draw alone. Release it here instead of at
        // closeAll so the tone/Laplacian/tail stages don't carry its
        // full-frame bytes (0.4-0.6 GB at 64 MP upscales) through the peak -
        // the Laplacian's pyramid stage is where the OOM lands. Tiled runs
        // execute this node once per band, so only the full-frame path may
        // drop it (a band would need the accumulator for every other band).
        if (!tileActive()) {
            closeTextures();
        }
    }

    private void closeTextures() {
        if (accumTex != null) {
            try {
                accumTex.close();
            } catch (Exception ignored) {
            }
            accumTex = null;
        }
    }

    @Override
    public void AfterRun() {
        closeTextures();
    }
}
