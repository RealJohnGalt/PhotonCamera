package com.particlesdevs.photoncamera.processing.processor;

import android.annotation.SuppressLint;
import android.graphics.ImageFormat;
import android.graphics.Point;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.CaptureResult;
import android.media.Image;

import com.particlesdevs.photoncamera.api.ParseExif;
import com.particlesdevs.photoncamera.app.PhotonCamera;
import com.particlesdevs.photoncamera.processing.DngCreator;
import com.particlesdevs.photoncamera.processing.ImageSaver;
import com.particlesdevs.photoncamera.processing.ProcessingEventsListener;
import com.particlesdevs.photoncamera.settings.PreferenceKeys;
import com.particlesdevs.photoncamera.util.Allocator;
import com.particlesdevs.photoncamera.util.FlacAudioRecorder;
import com.particlesdevs.photoncamera.util.Log;
import com.particlesdevs.photoncamera.util.SimpleStorageHelper;
import com.particlesdevs.photoncamera.processing.render.Parameters;

import android.os.StatFs;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

public class RawVideoProcessor extends ProcessorBase {
    private static final String TAG = "RawVideoProcessor";
    public static int videoCounter = 1;

    private final FlacAudioRecorder rawAudioRecorder = new FlacAudioRecorder();

    private volatile boolean fillParams = false;
    private Path outputFolder;
    private int writeBufferSize = 16;
    private int writeBufferCounter = 0;
    private final AtomicInteger pendingWrites = new AtomicInteger(0);
    private volatile ByteBuffer[] dngBuffers = null;
    private volatile ByteBuffer[] rawBuffers = null;
    private DngCreator dngCreator = null;
    private ExecutorService writeExecutor = null;
    private boolean isRecording = false;

    private long recordingStartMs = 0;    private long availableBytesAtStart = 0;
    private int frameWidth = 0;
    private int frameHeight = 0;
    private static final int BITS_PER_SAMPLE = 10;
    private Parameters parameters;

    public static class RawVideoStats {
        public final int pendingWrites;
        public final long elapsedMs;
        public final long estimatedBytes;
        public final long availableBytes;

        public RawVideoStats(int pendingWrites, long elapsedMs, long estimatedBytes, long availableBytes) {
            this.pendingWrites = pendingWrites;
            this.elapsedMs = elapsedMs;
            this.estimatedBytes = estimatedBytes;
            this.availableBytes = availableBytes;
        }
    }

    public RawVideoProcessor(ProcessingEventsListener processingEventsListener) {
        super(processingEventsListener);
    }

    @SuppressLint("DefaultLocale")
    public void videoStart(Path outputFolder, ParseExif.ExifData exifData,
                           CameraCharacteristics characteristics,
                           CaptureResult captureResult,
                           CaptureRequest captureRequest,
                           int cameraRotation,
                           ProcessingCallback callback) {
        this.outputFolder = outputFolder;
        this.exifData = exifData;
        this.characteristics = characteristics;
        this.captureResult = captureResult;
        this.cameraRotation = cameraRotation;
        this.captureRequest = captureRequest;
        videoCounter = 0;
        fillParams = false;
        recordingStartMs = System.currentTimeMillis();
        frameWidth = 0;
        frameHeight = 0;
        availableBytesAtStart = 0;
        try {
            StatFs statFs = new StatFs(outputFolder.getParent().toString());
            availableBytesAtStart = statFs.getAvailableBlocksLong() * statFs.getBlockSizeLong();
        } catch (Exception ignored) {}
        this.callback = callback;
        // Create output folder if not exists
        try {
            Files.createDirectories(outputFolder);
            if(!PreferenceKeys.isRawVideoWriteZip())
                Files.createDirectories(outputFolder.resolve(".dng"));
        } catch (IOException e) {
            Log.d(TAG, "Failed to create output directory: " + outputFolder + ", error: " + Log.getStackTraceString(e));
        }
        dngBuffers = new ByteBuffer[writeBufferSize];
        writeBufferCounter = 0;
        if (writeExecutor != null) {
            writeExecutor.shutdown();
        }
        writeExecutor = Executors.newSingleThreadExecutor();
        Thread th = new Thread( () -> {
            // Always record the rear (away-from-user) microphone
            rawAudioRecorder.start(
                    outputFolder.resolve("RAW_MIC.flac").toString());
        });
        th.start();
        parameters = new Parameters();
        PhotonCamera.getGyro().startVideoRecording(outputFolder, resolveFrameRate());
    }
    int shift = 0;
    @SuppressLint("DefaultLocale")
    public void videoCycle(Image image) {
        int format = image.getFormat();
        int startCounter = videoCounter;

        if(!fillParams){
            // Sync gyroflow timestamps to this first camera frame.
            PhotonCamera.getGyro().syncFirstFrame(image.getTimestamp());
            Log.d(TAG, "videoCycle: " + this + " " + image + " " + startCounter);
            int width = image.getWidth();
            int height = image.getHeight();
            if(format == ImageFormat.RAW_SENSOR){
                width = image.getPlanes()[0].getRowStride() / image.getPlanes()[0].getPixelStride();
                // Crop to 16:9
                if(PreferenceKeys.isRawVideoCrop169()) {
                    height = width * 9 / 16;
                    if (ImageSaver.SETTINGS.cropType) {
                        shift = 0;
                    } else {
                        shift = (image.getHeight() - height) / 2;
                        shift -= shift % 2;
                    }
                    shift *= image.getPlanes()[0].getRowStride();
                } else {
                    height = image.getHeight();
                }
            }
            if(format == ImageFormat.RAW10){
                width = image.getPlanes()[0].getRowStride() * 8 / 10; // Include padding pixels into output DNG
                height = image.getHeight();
            }

            frameWidth = width;
            frameHeight = height;

            parameters.rawSize = new Point(width, height);
            parameters.FillConstParameters(characteristics, parameters.rawSize);
            parameters.FillDynamicParameters(captureResult, captureRequest, 100);
            parameters.cameraRotation = this.cameraRotation;
            ParseExif.syncWithParameters(exifData, parameters);
            fillParams = true;
            dngCreator = new DngCreator();
            dngCreator.setParameters(parameters);
            dngCreator.setBinning(PreferenceKeys.isRawVideoDownscale4x());
            dngCreator.setFrameRate(resolveFrameRate());
            dngCreator.setCompression(false);
            if(PreferenceKeys.isRawVideoWriteZip()) {
                String archivePath = outputFolder.resolve("dng.zip").toString();
                int archiveFd = SimpleStorageHelper.openFdForWrite(archivePath);
                if (archiveFd >= 0) {
                    dngCreator.openArchiveByFd(archiveFd);
                } else {
                    dngCreator.openArchive(archivePath);
                }
            }
            /*if(format == ImageFormat.RAW_SENSOR) {
                dngCreator.setBitsPerSample(16);
            }
            if (format == ImageFormat.RAW10) {
                dngCreator.setBitsPerSample(10);
            }*/
            dngCreator.setBitsPerSample(10);
            dngBuffers[0] = dngCreator.dngBuffer(image.getPlanes()[0].getBuffer(), parameters.rawSize.x, parameters.rawSize.y);
            for (int i = 1; i < writeBufferSize; i++) {
                dngBuffers[i] = Allocator.allocateAndCopy(dngBuffers[0].capacity(), dngBuffers[0], 0);
                dngBuffers[i].put(dngBuffers[0]);
                dngBuffers[i].position(0);
            }
            ByteBuffer firstPlane = image.getPlanes()[0].getBuffer();
            int rawBufferCapacity = firstPlane.remaining();
            rawBuffers = new ByteBuffer[writeBufferSize];
            for (int i = 0; i < writeBufferSize; i++) {
                rawBuffers[i] = ByteBuffer.allocateDirect(rawBufferCapacity);
            }
            Log.d(TAG, "DNG buffer allocated, size: " + dngBuffers[0].capacity());
            image.close();
            isRecording = true;
        } else {
            if (!isRecording || writeExecutor == null || dngBuffers[0] == null || rawBuffers == null) {
                image.close();
                return;
            }
            if (pendingWrites.get() >= writeBufferSize) {
                image.close();
                Log.d(TAG, "Dropped frame");
                writeBufferCounter++;
                videoCounter++;
                processingEventsListener.onProcessingChanged(buildStats());
                return;
            }
            final int slot = writeBufferCounter % writeBufferSize;
            @SuppressLint("DefaultLocale")
            String path = outputFolder.resolve(String.format(".dng/RAW_%05d.dng", startCounter)).toString();
            if(PreferenceKeys.isRawVideoWriteZip()) {
                path = String.format("RAW_%05d.dng", startCounter);
            }
            ByteBuffer rawSlot = rawBuffers[slot];
            ByteBuffer planeBuffer = image.getPlanes()[0].getBuffer();
            planeBuffer.rewind();
            rawSlot.clear();
            rawSlot.put(planeBuffer);
            rawSlot.flip();
            image.close();
            pendingWrites.incrementAndGet();
            final String selectedPath = path;
            writeExecutor.execute(() -> {
                try {
                    if (fillParams && dngCreator != null) {
                        dngCreator.writeFile(dngBuffers[slot], rawBuffers[slot], selectedPath, shift);
                    }
                } finally {
                    pendingWrites.decrementAndGet();
                }
            });
            writeBufferCounter++;
        }

        videoCounter++;
        processingEventsListener.onProcessingChanged(buildStats());
    }

    private RawVideoStats buildStats() {
        long elapsedMs = System.currentTimeMillis() - recordingStartMs;
        long bytesPerFrame = (long) frameWidth * frameHeight * BITS_PER_SAMPLE / 8;
        long estimatedBytes = bytesPerFrame * videoCounter;
        return new RawVideoStats(pendingWrites.get(), elapsedMs, estimatedBytes, availableBytesAtStart);
    }

    private double resolveFrameRate() {
        switch (PreferenceKeys.getFpsMode()) {
            case 1: return 24.0;
            case 2: return 30.0;
            case 3: return 60.0;
            default: return 30.0; // auto
        }
    }

    private void processVideo() {

    }

    public void videoEnd() {
        isRecording = false;

        PhotonCamera.getGyro().stopVideoRecording();
        rawAudioRecorder.stop();
        if (writeExecutor != null) {
            writeExecutor.shutdown();
            try {
                // Pending writeFile tasks consume the ring buffers; wait for
                // them before tearing the archive and buffers down.
                writeExecutor.awaitTermination(10, java.util.concurrent.TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            dngCreator.closeArchive();
            writeExecutor = null;
        }
        releaseDngRing();
    }

    private void releaseDngRing() {
        if (dngBuffers != null) {
            for (int i = 0; i < dngBuffers.length; i++) {
                if (dngBuffers[i] == null) continue;
                if (i == 0 && dngCreator != null) {
                    // Slot 0 is the native tinydngwriter allocation.
                    dngCreator.freeDngBuffer(dngBuffers[i]);
                } else {
                    Allocator.free(dngBuffers[i]);
                }
                dngBuffers[i] = null;
            }
            dngBuffers = null;
        }
        rawBuffers = null;
        if (dngCreator != null) {
            dngCreator.close();
            dngCreator = null;
        }
    }
}