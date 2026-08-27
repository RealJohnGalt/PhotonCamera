package com.particlesdevs.photoncamera.processing.opengl;

import android.graphics.Bitmap;
import android.graphics.Point;
import android.opengl.GLES30;
import android.opengl.GLUtils;
import com.particlesdevs.photoncamera.util.Log;

import java.nio.ByteBuffer;

import static android.opengl.GLES20.GL_COLOR_ATTACHMENT0;
import static android.opengl.GLES20.GL_FRAMEBUFFER;
import static android.opengl.GLES20.GL_NO_ERROR;
import static android.opengl.GLES20.GL_RENDERBUFFER;
import static android.opengl.GLES20.glBindFramebuffer;
import static android.opengl.GLES20.glBindRenderbuffer;
import static android.opengl.GLES20.glFramebufferRenderbuffer;
import static android.opengl.GLES20.glGenFramebuffers;
import static android.opengl.GLES20.glGenRenderbuffers;
import static android.opengl.GLES20.glGetError;
import static android.opengl.GLES20.glRenderbufferStorage;
import static android.opengl.GLES30.GL_DRAW_FRAMEBUFFER;
import static android.opengl.GLES30.GL_RGBA8;
import static android.opengl.GLES30.glReadPixels;
import static android.opengl.GLES30.glViewport;

import com.particlesdevs.photoncamera.util.Allocator;

public class GLCoreBlockProcessing extends GLContext implements AutoCloseable {
    private static String TAG = "GLCoreBlockProcessing";
    public GLImage mOut = null;
    public Point shift = new Point(0,0);
    private final int mOutWidth, mOutHeight;
    public ByteBuffer mBlockBuffer;
    public ByteBuffer mOutBuffer;
    private final GLFormat mglFormat;

    public GLDrawParams.Allocate allocation = GLDrawParams.Allocate.Heap;

    public static void checkEglError(String op) {
        int error = GLES30.glGetError();
        if (error != GLES30.GL_NO_ERROR) {
            String msg = op + ": glError: " + GLUtils.getEGLErrorString(error) + " (" + Integer.toHexString(error) + ")";
            String TAG = "GLCoreBlockProcessing";
            Log.v(TAG, msg);
        }
    }
    public GLCoreBlockProcessing(Point size, GLImage out, GLFormat glFormat, GLDrawParams.Allocate alloc) {
        this(size, glFormat,alloc);
        allocation = alloc;
        mOut = out;
    }

    public GLCoreBlockProcessing(Point size, GLImage out, GLFormat glFormat, GLDrawParams.Allocate alloc, android.opengl.EGLContext shareContext) {
        this(size, glFormat, alloc, shareContext);
        allocation = alloc;
        mOut = out;
    }
    public GLCoreBlockProcessing(Point size, GLImage out, GLFormat glFormat) {
        this(size, glFormat, GLDrawParams.Allocate.Direct);
        mOut = out;
    }
    public GLCoreBlockProcessing(Point size, GLFormat glFormat) {
        this(size,glFormat, GLDrawParams.Allocate.Direct);
    }
    public GLCoreBlockProcessing(Point size, GLFormat glFormat, GLDrawParams.Allocate alloc) {
        super(size.x, GLDrawParams.TileSize);
        allocation = alloc;
        mglFormat = glFormat;
        mOutWidth = size.x;
        mOutHeight = size.y;
        mBlockBuffer = ByteBuffer.allocateDirect(mOutWidth * GLDrawParams.TileSize * mglFormat.mFormat.mSize * mglFormat.mChannels);
        glGenFramebuffers(1,bindFB,0);
        glGenRenderbuffers(1,bindRB,0);
        glBindRenderbuffer(GL_RENDERBUFFER,bindRB[0]);
        // Only TileSize-row windows are ever rendered and read back at once
        // (see drawBlocksToOutput), so full-height storage would waste
        // hundreds of MB at high resolutions.
        glRenderbufferStorage(GL_RENDERBUFFER, glFormat.getGLFormatInternal(), size.x, Math.min(size.y, GLDrawParams.TileSize));
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER,bindFB[0]);
        glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, bindRB[0]);
        final int capacity = mOutWidth * mOutHeight * mglFormat.mFormat.mSize * mglFormat.mChannels;
        if(alloc == GLDrawParams.Allocate.None) return;
        if(alloc == GLDrawParams.Allocate.Direct) mOutBuffer = Allocator.allocate(capacity);
        else {
            // Heap previously used ByteBuffer.allocate (Java heap) which at 64MP
            // is ~256 MB (SIMPLE_8x4) / ~512 MB (FLOAT_16x4) on the heap and
            // triggers GC/OOM. Use off-heap direct buffer (GC-managed, no
            // native leak accounting) to keep peak off the Java heap.
            // Example: 16MP ARGB_8888 64 MB heap -> 0 heap, 64 MP 256 MB -> 0 heap.
            mOutBuffer = ByteBuffer.allocateDirect(capacity);
        }
    }

    /** Shared-context constructor for Ultra HDR GPU retention (shares textures with {@code shareContext}). */
    public GLCoreBlockProcessing(Point size, GLFormat glFormat, GLDrawParams.Allocate alloc, android.opengl.EGLContext shareContext) {
        super(size.x, GLDrawParams.TileSize, shareContext);
        allocation = alloc;
        mglFormat = glFormat;
        mOutWidth = size.x;
        mOutHeight = size.y;
        mBlockBuffer = ByteBuffer.allocateDirect(mOutWidth * GLDrawParams.TileSize * mglFormat.mFormat.mSize * mglFormat.mChannels);
        glGenFramebuffers(1,bindFB,0);
        glGenRenderbuffers(1,bindRB,0);
        glBindRenderbuffer(GL_RENDERBUFFER,bindRB[0]);
        glRenderbufferStorage(GL_RENDERBUFFER, glFormat.getGLFormatInternal(), size.x, Math.min(size.y, GLDrawParams.TileSize));
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER,bindFB[0]);
        glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, bindRB[0]);
        final int capacity = mOutWidth * mOutHeight * mglFormat.mFormat.mSize * mglFormat.mChannels;
        if(alloc == GLDrawParams.Allocate.None) return;
        if(alloc == GLDrawParams.Allocate.Direct) mOutBuffer = Allocator.allocate(capacity);
        else {
            mOutBuffer = ByteBuffer.allocateDirect(capacity);
        }
    }
    public GLCoreBlockProcessing(Point size, GLImage out, GLFormat glFormat,ByteBuffer output) {
        super(size.x, GLDrawParams.TileSize);
        output.position(0);
        mglFormat = glFormat;
        mOutWidth = size.x;
        mOutHeight = size.y;
        mBlockBuffer = ByteBuffer.allocate(mOutWidth * GLDrawParams.TileSize * mglFormat.mFormat.mSize * mglFormat.mChannels);
        glGenFramebuffers(1,bindFB,0);
        glGenRenderbuffers(1,bindRB,0);
        glBindRenderbuffer(GL_RENDERBUFFER,bindRB[0]);
        // See the tile-sizing note in the main constructor.
        glRenderbufferStorage(GL_RENDERBUFFER, glFormat.getGLFormatInternal(), size.x, Math.min(size.y, GLDrawParams.TileSize));
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER,bindFB[0]);
        glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, bindRB[0]);
        mOutBuffer = output;
        mOut = out;
    }

    public void drawBlocksToOutput() {
        glBindFramebuffer(GL_FRAMEBUFFER, bindFB[0]);
        GLProg program = super.mProgram;
        GLBlockDivider divider = new GLBlockDivider(mOutHeight, GLDrawParams.TileSize);
        int[] row = new int[2];
        mOutBuffer.position(0);
        mBlockBuffer.position(0);
        while (divider.nextBlock(row)) {
            int y = row[0];
            int height = row[1];
            glViewport(0, 0, mOutWidth, height);
            checkEglError("glViewport");
            program.setVar("yOffset", y);
            program.draw();
            checkEglError("program");
            mBlockBuffer.position(0);
            glReadPixels(0, 0, mOutWidth, height, mglFormat.getGLFormatExternal(), mglFormat.getGLType(), mBlockBuffer);
            checkEglError("glReadPixels");
            // Avoid per-tile heap copy (new byte[] + get/put) that at 64MP would
            // churn GC for the last partial tile. Instead slice mBlockBuffer to
            // exact valid bytes and bulk-put without intermediate array.
            // 16MP (4096x3840) full tiles: 0 copies; 64MP (9248x6936) partial tile
            // ~9248*8*4=296KB previously via byte[] -> now zero-copy via slice.
            int validBytes = mOutWidth * height * mglFormat.mFormat.mSize * mglFormat.mChannels;
            int oldLimit = mBlockBuffer.limit();
            mBlockBuffer.position(0).limit(validBytes);
            mOutBuffer.put(mBlockBuffer);
            mBlockBuffer.limit(oldLimit).position(0);
            // Keep mOutBuffer position contiguous; mBlockBuffer reused next iter.
            // Full-tile path previously did put(mBlockBuffer) with stale limit
            // (TileSize bytes); slice method is exact for both.
        }
        mOutBuffer.position(0);
        mBlockBuffer = null;
        if (mOut != null) mOut.byteBuffer = mOutBuffer;
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    /**
     * Streams the rendered tiles directly into {@code sink}'s pixel memory
     * (a software ARGB_8888 bitmap wrapped via {@link Allocator#wrapBitmap}),
     * skipping every intermediate full-frame buffer. The per-tile program
     * replay (viewport + yOffset) is identical to {@link #drawBlocksToOutput()}.
     */
    public void drawBlocksToOutput(Bitmap sink) {
        ByteBuffer wrapped = Allocator.wrapBitmap(sink);
        if (wrapped == null) {
            throw new IllegalStateException("Failed to lock bitmap pixels for direct output");
        }
        try {
            glBindFramebuffer(GL_FRAMEBUFFER, bindFB[0]);
            GLProg program = super.mProgram;
            GLBlockDivider divider = new GLBlockDivider(mOutHeight, GLDrawParams.TileSize);
            int[] row = new int[2];
            while (divider.nextBlock(row)) {
                int y = row[0];
                int height = row[1];
                glViewport(0, 0, mOutWidth, height);
                checkEglError("glViewport");
                program.setVar("yOffset", y);
                program.draw();
                checkEglError("program");
                wrapped.position(y * mOutWidth * 4);
                wrapped.limit((y + height) * mOutWidth * 4);
                glReadPixels(0, 0, mOutWidth, height, mglFormat.getGLFormatExternal(), mglFormat.getGLType(), wrapped);
                checkEglError("glReadPixels");
            }
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
        } finally {
            Allocator.unlockBitmap(sink);
        }
    }


    public ByteBuffer drawBlocksToOutput(Point size, GLFormat glFormat) {
        return drawBlocksToOutput(size,glFormat, GLDrawParams.Allocate.Heap);
    }

    public ByteBuffer drawBlocksToOutput(Point size, GLFormat glFormat,GLDrawParams.Allocate alloc) {
        ByteBuffer mOutBuffer;
        allocation = alloc;
        if(alloc == GLDrawParams.Allocate.Direct) mOutBuffer = Allocator.allocate(size.x * size.y * glFormat.mFormat.mSize * glFormat.mChannels);
        else if(alloc == GLDrawParams.Allocate.Heap)
            // Avoid Java heap OOM: 16MP *4B=64MB / 64MP *4B=256MB off-heap.
            mOutBuffer = ByteBuffer.allocateDirect(size.x * size.y * glFormat.mFormat.mSize * glFormat.mChannels);
        else
            mOutBuffer = ByteBuffer.allocate(size.x * size.y * glFormat.mFormat.mSize * glFormat.mChannels);
        return drawBlocksToOutput(size,glFormat,mOutBuffer);
    }

    public ByteBuffer drawBlocksToOutput(Point size, GLFormat glFormat,ByteBuffer mOutBuffer) {
        glBindFramebuffer(GL_FRAMEBUFFER, bindFB[0]);
        checkEglError("glBindFramebuffer");
        GLProg program = super.mProgram;
        GLBlockDivider divider = new GLBlockDivider(size.y, GLDrawParams.TileSize);
        int[] row = new int[2];
        ByteBuffer mBlockBuffert = mBlockBuffer;
        mOutBuffer.position(0);
        mBlockBuffert.position(0);
        while (divider.nextBlock(row)) {
            int y = row[0];
            int height = row[1];
            glViewport(0, 0, size.x, height);
            checkEglError("glViewport");
            program.setVar("yOffset", y);
            program.draw();
            checkEglError("program");
            mBlockBuffert.position(0);
            glReadPixels(0, 0, size.x, height, glFormat.getGLFormatExternal(), glFormat.getGLType(), mBlockBuffert);
            checkEglError("glReadPixels");
            // Unified zero-copy slice for both full and partial tiles — avoids
            // per-tile new byte[] (would be ~296KB @64MP partial, 64KB @16MP).
            int validBytes = size.x * height * glFormat.mFormat.mSize * glFormat.mChannels;
            int oldLim = mBlockBuffert.limit();
            mBlockBuffert.position(0).limit(validBytes);
            mOutBuffer.put(mBlockBuffert);
            mBlockBuffert.limit(oldLim).position(0);
        }
        mOutBuffer.position(0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        return mOutBuffer;
    }

    @Override
    public void close() {
        // Ensure GPU work is complete before tearing down EGL state.
        try {
            GLES30.glFinish();
        } catch (Exception ignored) {}
        try {
            super.close();
        } catch (Exception ignored) {}
        if (mOut != null) {
            try {
                mOut.close();
            } catch (Exception ignored) {}
            mOut = null;
        }
        if (mBlockBuffer != null) {
            try {
                mBlockBuffer.clear();
            } catch (Exception ignored) {}
            mBlockBuffer = null;
        }
        if (mOutBuffer != null) {
            try {
                if (allocation == GLDrawParams.Allocate.Direct) {
                    // Intentionally temporarily leaked: must hold the malloc
                } else {
                    mOutBuffer.clear();
                }
            } catch (Exception ignored) {}
            mOutBuffer = null;
        }
        // FBO/RBO are owned by this context; delete while context was current.
        // They are recreated per-pipeline, so stale IDs must not survive eglTerminate.
        try {
            if (bindFB[0] != 0) GLES30.glDeleteFramebuffers(1, bindFB, 0);
        } catch (Exception ignored) {}
        try {
            if (bindRB[0] != 0) GLES30.glDeleteRenderbuffers(1, bindRB, 0);
        } catch (Exception ignored) {}
    }
}
