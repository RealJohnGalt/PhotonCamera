package com.particlesdevs.photoncamera.processing.ultrahdr;

import android.graphics.Bitmap;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Normalises the GPU-produced, fixed-range gain map (RGBA8, single channel) into
 * the final Ultra HDR gain map and derives its {@code GainMapMin}/{@code GainMapMax}
 * metadata.
 *
 * <p>The GPU shader stores, per (downsampled) pixel, a value {@code v = logBoost /
 * scale} clamped to [0,1] where {@code logBoost = log2(hdrLinear/sdrLinear)} and
 * {@code scale} is the total log2 range covered by the pass. Here we recover the
 * actual {@code logBoost} range present in the image and rescale the 8-bit values
 * to fill the [0,255] range, giving the gain-map {@code GainMapMin}/{@code GainMapMax}
 * (in log2). The decode formula then reconstructs the HDR image from the SDR base
 * for any display headroom.
 */
public final class GainMapComputer {

    // Linear-space epsilon used both when computing the ratio and as the
    // gain-map OffsetSDR/OffsetHDR values (1/64, matching the reference).
    static final float EPS = 1.0f / 64.0f;
    static final float GAMMA = 1.0f;
    // Total log2 range covered by the GPU pass (e.g. 6 stops -> up to 64x boost).
    public static final float SCALE = 12.0f;
    // Gain-map downsample factor per axis (gain map is 1/SCALE_DOWN^2 the pixels).
    public static final int SCALE_DOWN = 1;

    private GainMapComputer() {}

    public static class Result {
        /** Single-channel gain map, packed as R=G=B=value, A=255 (ARGB_8888). */
        public final Bitmap gainMap;
        public final float gainMapMin;
        public final float gainMapMax;
        public final float hdrCapacityMax;
        public final int gainW;
        public final int gainH;

        Result(Bitmap gainMap, float gainMapMin, float gainMapMax) {
            this.gainMap = gainMap;
            this.gainMapMin = gainMapMin;
            this.gainMapMax = gainMapMax;
            // HDRCapacityMax must be greater than HDRCapacityMin (0); otherwise
            // the viewer's headroom weight divides by zero/negative and every
            // viewer clamps it differently.
            this.hdrCapacityMax = Math.max(gainMapMax, 1e-3f);
            this.gainW = gainMap.getWidth();
            this.gainH = gainMap.getHeight();
        }
    }

    /**
     * @param rgba8 GPU-produced gain map (RGBA8, R=G=B), row-major
     * @param gw    gain map width in pixels
     * @param gh    gain map height in pixels
     * @param scale the log2 range the GPU pass used (== {@link #SCALE})
     */
    public static Result compute(ByteBuffer rgba8, int gw, int gh, float scale) {
        rgba8.order(ByteOrder.LITTLE_ENDIAN);
        final int n = gw * gh;
        final float[] logBoost = new float[n];
        float minBoost = Float.POSITIVE_INFINITY;
        float maxBoost = Float.NEGATIVE_INFINITY;
        int invalid = 0;

        for (int i = 0; i < n; i++) {
            final int r = rgba8.get(i * 4) & 0xFF;
            final float v = r / 255.0f;
            // GPU encodes logBoost directly in [0, scale].
            final float lb = v * scale;
            // Clamp any residual negative/garbage to identity.
            logBoost[i] = Math.max(lb, 0f);
            if (lb < 0f) invalid++;
            if (lb < minBoost) minBoost = lb;
            if (lb > maxBoost) maxBoost = lb;
        }

        // A small fraction of negative boosts is expected (near-black quantization
        // noise) and harmless; only a large fraction indicates a genuinely broken
        // HDR buffer (e.g. an empty/garbage rendition). Fail loudly then so the
        // caller falls back to plain SDR instead of shipping a silent artifact.
        if (invalid > n / 5) {
            throw new IllegalStateException(
                    "HDR rendition invalid: " + invalid + "/" + n
                            + " gain map pixels below 0 stops");
        }

        if (!Float.isFinite(minBoost)) minBoost = 0f;
        if (!Float.isFinite(maxBoost)) maxBoost = 0f;
        // The HDR rendition is SDR * headroom with headroom >= 1, so boosts are
        // non-negative by construction; keep the metadata range in [0, ...].
        if (minBoost < 0f) minBoost = 0f;
        if (maxBoost < minBoost) maxBoost = minBoost;
        if (maxBoost - minBoost < 1e-3f) maxBoost = minBoost + 1e-3f;

        // Pad slightly so extreme pixels don't sit exactly on the endpoints.
        final float pad = (maxBoost - minBoost) * 0.02f + 1e-4f;

        // Anchor the bottom of the metadata range at exactly 0: identity gain for
        // midtones/shadows. gMax tracks the maximum boost (plus padding).
        final float gMin = 0f;
        final float gMax = maxBoost + pad;
        final float range = gMax - gMin;

        final Bitmap gainMap = Bitmap.createBitmap(gw, gh, Bitmap.Config.ARGB_8888);
        final int[] out = new int[n];
        for (int i = 0; i < n; i++) {
            // Normalize logBoost into [0,1] across [gMin,gMax], with gMin==0 for identity.
            float vNorm = (logBoost[i] - gMin) / range;
            if (vNorm < 0f) vNorm = 0f;
            else if (vNorm > 1f) vNorm = 1f;
            final int byteVal = Math.round(vNorm * 255.0f);
            out[i] = (0xFF << 24) | (byteVal << 16) | (byteVal << 8) | byteVal;
        }
        gainMap.setPixels(out, 0, gw, 0, 0, gw, gh);

        return new Result(gainMap, gMin, gMax);
    }

    static float gamma() {
        return GAMMA;
    }

    public static float epsilon() {
        return EPS;
    }

    static float scale() {
        return SCALE;
    }
}
