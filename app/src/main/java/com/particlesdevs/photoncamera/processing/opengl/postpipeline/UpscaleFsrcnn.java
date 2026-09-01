package com.particlesdevs.photoncamera.processing.opengl.postpipeline;

import android.content.Context;
import android.graphics.Point;

import com.particlesdevs.photoncamera.app.PhotonCamera;
import com.particlesdevs.photoncamera.processing.ml.FsrcnnNcnnProcessor;
import com.particlesdevs.photoncamera.processing.opengl.GLFormat;
import com.particlesdevs.photoncamera.processing.opengl.GLTexture;
import com.particlesdevs.photoncamera.processing.opengl.nodes.Node;
import com.particlesdevs.photoncamera.util.Log;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

/**
 * Upscales a cropped capture toward its full-sensor output size.
 *
 * <p>The FSRCNN model is a one-channel Y/luma x4 model. It does not accept or
 * return RGB data. This node therefore:</p>
 *
 * <ol>
 *     <li>Reads the prior texture as float B,G,R,A pixels.</li>
 *     <li>Calculates normalized display-domain luma Y.</li>
 *     <li>Runs FSRCNN on the Y plane only.</li>
 *     <li>Bilinearly upsamples source chroma residuals.</li>
 *     <li>Reconstructs B,G,R from learned luma plus conventional chroma.</li>
 *     <li>Falls back to GPU interpolation on any model/allocation failure.</li>
 * </ol>
 *
 * <p>The project convention used by the existing ncnn FlowNet bridge is B,G,R,A
 * for float texture readback. This class deliberately preserves that ordering.
 * Do not change the component order unless the GL texture transfer convention
 * changes throughout the project.</p>
 */
public final class UpscaleFsrcnn extends Node {
    private static final String TAG = "UpscaleFsrcnn";

    private static final int SCALE = 4;
    private static final int CHANNEL_BLUE = 0;
    private static final int CHANNEL_GREEN = 1;
    private static final int CHANNEL_RED = 2;
    private static final int CHANNEL_ALPHA = 3;

    /*
     * Rec. 709 / sRGB luma coefficients. The upstream FSRCNN checkpoint was
     * trained on normalized one-channel image luma data, not independent RGB
     * planes.
     */
    private static final float LUMA_RED = 0.2126f;
    private static final float LUMA_GREEN = 0.7152f;
    private static final float LUMA_BLUE = 0.0722f;

    public UpscaleFsrcnn() {
        super("", "UpscaleFsrcnn");
    }

    @Override
    public void Compile() {
        // This node performs CPU-side ncnn inference and texture upload.
    }

    private static float clamp01(float value) {
        return Math.max(0.0f, Math.min(1.0f, value));
    }

    /**
     * Bilinearly samples one BGR(A) channel from a source float buffer.
     *
     * <p>The mapping uses pixel-center coordinates. For a 4x output, output
     * pixel center x maps to {@code ((x + 0.5) / 4) - 0.5} in source space.</p>
     */
    private static float sampleChannelBilinear(
            FloatBuffer source,
            int width,
            int height,
            float sourceX,
            float sourceY,
            int channel) {
        float clampedX = Math.max(0.0f, Math.min(sourceX, width - 1.0f));
        float clampedY = Math.max(0.0f, Math.min(sourceY, height - 1.0f));

        int x0 = (int) clampedX;
        int y0 = (int) clampedY;
        int x1 = Math.min(x0 + 1, width - 1);
        int y1 = Math.min(y0 + 1, height - 1);

        float fractionX = clampedX - x0;
        float fractionY = clampedY - y0;

        float topLeft = source.get((y0 * width + x0) * 4 + channel);
        float topRight = source.get((y0 * width + x1) * 4 + channel);
        float bottomLeft = source.get((y1 * width + x0) * 4 + channel);
        float bottomRight = source.get((y1 * width + x1) * 4 + channel);

        float top = topLeft + (topRight - topLeft) * fractionX;
        float bottom = bottomLeft + (bottomRight - bottomLeft) * fractionX;

        return top + (bottom - top) * fractionY;
    }

    private static float lumaFromBgr(float blue, float green, float red) {
        return LUMA_RED * red + LUMA_GREEN * green + LUMA_BLUE * blue;
    }

    /**
     * Produces a regular GPU interpolation output and updates pipeline size
     * bookkeeping. This is used for non-cropped captures, invalid parameters,
     * native model failures, direct-buffer allocation failures, and texture
     * readback failures.
     */
    private GLTexture fallbackInterpolation(GLTexture input, Point target) {
        GLTexture output = glUtils.interpolate(input, target);

        basePipeline.workSize = new Point(target);
        ((PostPipeline) basePipeline).cropSize = new Point(target);

        return output;
    }

    @Override
    public void Run() {
        GLTexture input = previousNode.WorkingTexture;

        if (input == null) {
            WorkingTexture = null;
            return;
        }

        if (basePipeline.mParameters.fullRawSize == null ||
                !basePipeline.mParameters.isCropped) {
            WorkingTexture = input;
            return;
        }

        Point full = basePipeline.mParameters.fullRawSize;

        if (full.x <= 0 || full.y <= 0 ||
                input.mSize.x <= 0 || input.mSize.y <= 0) {
            WorkingTexture = input;
            return;
        }

        /*
         * Use the actual previous-node texture size rather than rawSize. Nodes
         * earlier in the pipeline may have applied aspect-ratio cropping or
         * another size-changing operation.
         */
        Point sourceSize = new Point(input.mSize.x, input.mSize.y);

        Point target = new Point(full.x & ~3, full.y & ~3);

        if (target.x < 4) {
            target.x = 4;
        }

        if (target.y < 4) {
            target.y = 4;
        }

        if (target.equals(sourceSize)) {
            WorkingTexture = input;
            return;
        }

        long upscaledWidthLong = (long) sourceSize.x * SCALE;
        long upscaledHeightLong = (long) sourceSize.y * SCALE;

        long sourcePixels =
                (long) sourceSize.x * (long) sourceSize.y;

        long upscaledPixels =
                upscaledWidthLong * upscaledHeightLong;

        long lumaInputBytes =
                sourcePixels * Float.BYTES;

        long reconstructedBgrBytes =
                upscaledPixels * 4L * Float.BYTES;

        /*
         * ByteBuffer capacities and GLTexture dimensions use int-sized values.
         * Reject impossible allocations and use the regular GPU path instead.
         */
        if (upscaledWidthLong > Integer.MAX_VALUE ||
                upscaledHeightLong > Integer.MAX_VALUE ||
                lumaInputBytes > Integer.MAX_VALUE ||
                reconstructedBgrBytes > Integer.MAX_VALUE) {
            Log.e(
                    TAG,
                    "FSRCNN dimensions/allocation exceed limits; using interpolation: " +
                            sourceSize.x + "x" + sourceSize.y +
                            " -> " + target.x + "x" + target.y);

            WorkingTexture = fallbackInterpolation(input, target);
            return;
        }

        Point fsrcnnSize = new Point(
                (int) upscaledWidthLong,
                (int) upscaledHeightLong);

        GLTexture fsrcnnTexture = null;
        FsrcnnNcnnProcessor processor = null;

        try {
            Context context = PhotonCamera.getAppContext();

            if (context == null) {
                throw new IllegalStateException("Application context is unavailable");
            }

            ByteBuffer sourceBytes = input.textureBuffer(
                    new GLFormat(GLFormat.DataType.FLOAT_32, 4),
                    true);

            if (sourceBytes == null) {
                throw new IllegalStateException("Could not read source GL texture");
            }

            sourceBytes.order(ByteOrder.nativeOrder());
            sourceBytes.rewind();

            FloatBuffer source = sourceBytes.asFloatBuffer();

            if (source.capacity() < sourcePixels * 4L) {
                throw new IllegalStateException(
                        "Source texture readback is smaller than expected");
            }

            /*
             * The native FSRCNN layer expects exactly one direct float32 plane.
             * The model source was trained with luma normalized to [0, 1].
             */
            ByteBuffer lumaBytes = ByteBuffer.allocateDirect(
                    (int) lumaInputBytes).order(ByteOrder.nativeOrder());

            FloatBuffer luma = lumaBytes.asFloatBuffer();

            for (int pixel = 0; pixel < (int) sourcePixels; ++pixel) {
                int offset = pixel * 4;

                float blue = source.get(offset + CHANNEL_BLUE);
                float green = source.get(offset + CHANNEL_GREEN);
                float red = source.get(offset + CHANNEL_RED);

                /*
                 * Values outside [0, 1] are not part of the source model's
                 * trained domain. Clamp only the luma sent into FSRCNN; retain
                 * source color values for chroma reconstruction below.
                 */
                luma.put(
                        pixel,
                        clamp01(lumaFromBgr(blue, green, red)));
            }

            processor = new FsrcnnNcnnProcessor();

            if (!processor.create(context)) {
                throw new IllegalStateException(
                        "Could not load FSRCNN ncnn assets");
            }

            FsrcnnNcnnProcessor.Result result =
                    processor.runLumaInference(
                            lumaBytes,
                            sourceSize.x,
                            sourceSize.y);

            if (result == null) {
                throw new IllegalStateException("FSRCNN inference failed");
            }

            if (result.width != fsrcnnSize.x ||
                    result.height != fsrcnnSize.y) {
                throw new IllegalStateException(
                        "FSRCNN returned " +
                                result.width + "x" + result.height +
                                "; expected " +
                                fsrcnnSize.x + "x" + fsrcnnSize.y);
            }

            FloatBuffer learnedLuma = result.asFloatBuffer();

            if (learnedLuma.capacity() < upscaledPixels) {
                throw new IllegalStateException(
                        "FSRCNN output buffer is smaller than expected");
            }

            ByteBuffer reconstructedBytes = ByteBuffer.allocateDirect(
                    (int) reconstructedBgrBytes).order(ByteOrder.nativeOrder());

            FloatBuffer reconstructed = reconstructedBytes.asFloatBuffer();

            /*
             * FSRCNN modifies luma detail only. Reconstruct color by retaining
             * conventional bilinear chroma residuals:
             *
             *   Y = 0.2126R + 0.7152G + 0.0722B
             *   Cr = R - Y
             *   Cb = B - Y
             *
             *   R = Y_fsrcnn + Cr_upscaled
             *   B = Y_fsrcnn + Cb_upscaled
             *   G = (Y_fsrcnn - 0.2126R - 0.0722B) / 0.7152
             *
             * The result is written in B,G,R,A order, matching the project
             * texture-readback convention used by its existing ncnn bridge.
             */
            for (int y = 0; y < fsrcnnSize.y; ++y) {
                float sourceY =
                        ((y + 0.5f) * sourceSize.y / fsrcnnSize.y) - 0.5f;

                for (int x = 0; x < fsrcnnSize.x; ++x) {
                    float sourceX =
                            ((x + 0.5f) * sourceSize.x / fsrcnnSize.x) - 0.5f;

                    int outputPixel = y * fsrcnnSize.x + x;

                    float blue = sampleChannelBilinear(
                            source,
                            sourceSize.x,
                            sourceSize.y,
                            sourceX,
                            sourceY,
                            CHANNEL_BLUE);

                    float green = sampleChannelBilinear(
                            source,
                            sourceSize.x,
                            sourceSize.y,
                            sourceX,
                            sourceY,
                            CHANNEL_GREEN);

                    float red = sampleChannelBilinear(
                            source,
                            sourceSize.x,
                            sourceSize.y,
                            sourceX,
                            sourceY,
                            CHANNEL_RED);

                    float sourceLuma = lumaFromBgr(blue, green, red);

                    float enhancedLuma =
                            clamp01(learnedLuma.get(outputPixel));

                    float reconstructedRed =
                            enhancedLuma + (red - sourceLuma);

                    float reconstructedBlue =
                            enhancedLuma + (blue - sourceLuma);

                    float reconstructedGreen =
                            (enhancedLuma -
                                    LUMA_RED * reconstructedRed -
                                    LUMA_BLUE * reconstructedBlue) /
                            LUMA_GREEN;

                    int destination = outputPixel * 4;

                    reconstructed.put(
                            destination + CHANNEL_BLUE,
                            clamp01(reconstructedBlue));

                    reconstructed.put(
                            destination + CHANNEL_GREEN,
                            clamp01(reconstructedGreen));

                    reconstructed.put(
                            destination + CHANNEL_RED,
                            clamp01(reconstructedRed));

                    reconstructed.put(
                            destination + CHANNEL_ALPHA,
                            1.0f);
                }
            }

            reconstructedBytes.rewind();

            fsrcnnTexture = new GLTexture(
                    new Point(fsrcnnSize),
                    new GLFormat(GLFormat.DataType.FLOAT_32, 4));

            fsrcnnTexture.loadData(reconstructedBytes);
        } catch (OutOfMemoryError error) {
            Log.e(
                    TAG,
                    "FSRCNN memory allocation failed; using interpolation: " +
                            error.getMessage());

            if (fsrcnnTexture != null) {
                try {
                    fsrcnnTexture.close();
                } catch (Exception ignored) {
                    // Best-effort cleanup before fallback.
                }
                fsrcnnTexture = null;
            }
        } catch (Exception error) {
            Log.e(
                    TAG,
                    "FSRCNN failed; using interpolation: " +
                            error.getClass().getSimpleName() +
                            ": " +
                            error.getMessage());

            if (fsrcnnTexture != null) {
                try {
                    fsrcnnTexture.close();
                } catch (Exception ignored) {
                    // Best-effort cleanup before fallback.
                }
                fsrcnnTexture = null;
            }
        } finally {
            if (processor != null) {
                processor.close();
            }
        }

        if (fsrcnnTexture == null) {
            /*
             * The model is unavailable or inference failed. Still perform the
             * intended crop-to-full-size expansion without leaving a broken
             * texture in the post-processing pipeline.
             */
            WorkingTexture = fallbackInterpolation(input, target);
            return;
        }

        GLTexture output;

        if (fsrcnnSize.equals(target)) {
            output = fsrcnnTexture;
        } else {
            output = glUtils.interpolate(fsrcnnTexture, target);
            fsrcnnTexture.close();
        }

        WorkingTexture = output;
        basePipeline.workSize = new Point(target);
        ((PostPipeline) basePipeline).cropSize = new Point(target);
    }
}
