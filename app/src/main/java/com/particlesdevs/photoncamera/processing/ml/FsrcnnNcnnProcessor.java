package com.particlesdevs.photoncamera.processing.ml;

import android.content.Context;
import android.content.res.AssetManager;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

/**
 * JNI wrapper for the Y-only FSRCNN x4 ncnn model.
 *
 * <p>Model contract:</p>
 * <ul>
 *     <li>Input blob: {@code in0}</li>
 *     <li>Output blob: {@code out0}</li>
 *     <li>Input: one float32 Y/luma plane, {@code width * height} samples</li>
 *     <li>Output: one float32 Y/luma plane,
 *         {@code (width * 4) * (height * 4)} samples</li>
 *     <li>Expected model-domain range: approximately {@code [0.0, 1.0]}</li>
 * </ul>
 *
 * <p>This is not an RGB model. Callers must convert source BGR/RGB pixels to
 * luma before inference, then combine the returned upscaled luma with
 * conventionally upscaled chroma.</p>
 */
public final class FsrcnnNcnnProcessor implements AutoCloseable {
    static {
        System.loadLibrary("ncnnMl");
    }

    private static final String MODEL_PARAM =
            "models/fsrcnn_small_4x_y.ncnn.param";

    private static final int SCALE = 4;

    private long handle;

    /**
     * A direct native-order buffer containing one channel of float32 luma data.
     * The buffer starts at position zero and contains {@code width * height}
     * output samples.
     */
    public static final class Result {
        public final ByteBuffer buffer;
        public final int width;
        public final int height;

        private Result(ByteBuffer buffer, int width, int height) {
            this.buffer = buffer;
            this.width = width;
            this.height = height;
        }

        /**
         * Returns a native-order duplicate of the luma byte buffer positioned
         * at zero. The returned buffer has one float32 sample per output pixel.
         */
        public ByteBuffer asByteBuffer() {
            ByteBuffer copy = buffer.duplicate().order(ByteOrder.nativeOrder());
            copy.rewind();
            return copy;
        }

        /**
         * Returns a FloatBuffer view with one luma sample per output pixel.
         * Its position starts at zero and its limit is width * height.
         */
        public FloatBuffer asFloatBuffer() {
            ByteBuffer copy = buffer.duplicate().order(ByteOrder.nativeOrder());
            copy.rewind();
            return copy.asFloatBuffer();
        }
    }

    /**
     * Loads the matched FSRCNN .param and .bin assets.
     *
     * @return true only when native model creation and asset loading succeed.
     */
    public synchronized boolean create(Context context) {
        close();

        if (context == null) {
            return false;
        }

        AssetManager assetManager = context.getAssets();
        if (assetManager == null) {
            return false;
        }

        handle = nativeCreate(assetManager, MODEL_PARAM);
        return handle != 0L;
    }

    /**
     * Runs the one-channel FSRCNN x4 model.
     *
     * @param yIn a direct, native-order ByteBuffer containing width * height
     *            float32 luma values in the model's expected value range
     * @param width source width in pixels
     * @param height source height in pixels
     * @return one-channel 4x luma output, or null when validation, allocation,
     *         native inference, or output-shape validation fails
     */
    public synchronized Result runLumaInference(
            ByteBuffer yIn,
            int width,
            int height) {
        if (handle == 0L ||
                yIn == null ||
                !yIn.isDirect() ||
                width <= 0 ||
                height <= 0) {
            return null;
        }

        long inputFloats = (long) width * (long) height;
        long inputBytes = inputFloats * Float.BYTES;

        long outputWidthLong = (long) width * SCALE;
        long outputHeightLong = (long) height * SCALE;
        long outputFloats = outputWidthLong * outputHeightLong;
        long outputBytes = outputFloats * Float.BYTES;

        if (inputBytes > Integer.MAX_VALUE ||
                outputWidthLong > Integer.MAX_VALUE ||
                outputHeightLong > Integer.MAX_VALUE ||
                outputBytes > Integer.MAX_VALUE ||
                yIn.capacity() < inputBytes) {
            return null;
        }

        final ByteBuffer output;

        try {
            output = ByteBuffer.allocateDirect((int) outputBytes)
                    .order(ByteOrder.nativeOrder());
        } catch (OutOfMemoryError error) {
            return null;
        }

        /*
         * Native code obtains the raw start address with
         * GetDirectBufferAddress(), so inference always reads from the buffer
         * base. Rewind prevents caller position state from being misleading.
         */
        yIn.rewind();
        output.rewind();

        boolean ok = nativeRun(
                handle,
                yIn,
                width,
                height,
                output);

        if (!ok) {
            return null;
        }

        output.rewind();

        return new Result(
                output,
                (int) outputWidthLong,
                (int) outputHeightLong);
    }

    /**
     * Releases the native ncnn model and its allocations. This method is safe
     * to call repeatedly.
     */
    @Override
    public synchronized void close() {
        if (handle != 0L) {
            nativeDestroy(handle);
            handle = 0L;
        }
    }

    private static native long nativeCreate(
            AssetManager assetManager,
            String paramPath);

    private static native boolean nativeRun(
            long handle,
            ByteBuffer yIn,
            int width,
            int height,
            ByteBuffer yOut);

    private static native void nativeDestroy(long handle);
}
