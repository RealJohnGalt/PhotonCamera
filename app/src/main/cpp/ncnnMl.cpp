//
// Created by eszdman on 16.08.2026.
//
// Single JNI wrapper around ncnn (Vulkan backend) for the ML models used by
// PhotonCamera:
//
//   * FlowNet-v2 dense optical flow
//   * KernelNet anisotropic parameter model
//   * FSRCNN-small x4 Y/luma upscaler
//

#include <jni.h>
#include <android/log.h>
#include <android/asset_manager_jni.h>
#include <android/asset_manager.h>

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <sys/time.h>
#include <vector>

#include "net.h"
#include "mat.h"

#if NCNN_VULKAN
#include "gpu.h"
#endif

#include "flownet_register.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#define LOG_TAG "NcnnML"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

static void pinOpenMPThreads(int num_threads) {
#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(num_threads);
    LOGI("openmp: linked statically, pinned to %d threads", omp_get_max_threads());
#else
    (void) num_threads;
#endif
}

static int64_t nowMs() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return (int64_t) tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

static int64_t nowUs() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return (int64_t) tv.tv_sec * 1000000 + tv.tv_usec;
}

// "models/foo.ncnn.param" -> "models/foo.ncnn.bin"
static std::string paramToBinPath(const std::string& paramPath) {
    std::string binPath = paramPath;
    size_t pos = binPath.rfind(".param");
    if (pos != std::string::npos) {
        binPath.replace(pos, 6, ".bin");
    }
    return binPath;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_particlesdevs_photoncamera_processing_ml_NcnnMl_nativeEnsureInit(
        JNIEnv*, jclass) {
    return JNI_TRUE;
}

// ===========================================================================
// FlowNet-v2
// ===========================================================================

struct FlowNetCtx {
    ncnn::Net net;
};

extern "C" JNIEXPORT jlong JNICALL
Java_com_particlesdevs_photoncamera_processing_ml_FlowNetNcnnProcessor_nativeCreate(
        JNIEnv* env,
        jclass,
        jobject assetManager,
        jstring paramPath) {
    int64_t t0 = nowMs();

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    if (mgr == nullptr) {
        LOGE("AAssetManager_fromJava failed");
        return 0;
    }

    const char* path = env->GetStringUTFChars(paramPath, nullptr);
    if (path == nullptr) {
        LOGE("paramPath null");
        return 0;
    }

    std::string paramStr = path;
    env->ReleaseStringUTFChars(paramPath, path);

    auto* ctx = new (std::nothrow) FlowNetCtx();
    if (ctx == nullptr) {
        LOGE("OOM allocating FlowNetCtx");
        return 0;
    }

    ctx->net.opt.use_vulkan_compute = true;
    ctx->net.opt.use_fp16_packed = true;
    ctx->net.opt.use_fp16_storage = true;
    ctx->net.opt.use_fp16_arithmetic = true;
    ctx->net.opt.use_bf16_storage = false;
    ctx->net.opt.use_subgroup_ops = false;
    ctx->net.opt.num_threads = 4;
    ctx->net.opt.lightmode = false;

    pinOpenMPThreads(ctx->net.opt.num_threads);

    if (getenv("FLOWNET_CPU") && getenv("FLOWNET_CPU")[0] == '1') {
        ctx->net.opt.use_vulkan_compute = false;
        LOGI("flownet: Vulkan disabled by FLOWNET_CPU=1, using CPU");
    }

    flownet_register_custom_layers(ctx->net);

    const std::string binPath = paramToBinPath(paramStr);

    if (ctx->net.load_param(mgr, paramStr.c_str()) != 0) {
        LOGE("flownet load_param(%s) failed", paramStr.c_str());
        delete ctx;
        return 0;
    }

    if (ctx->net.load_model(mgr, binPath.c_str()) != 0) {
        LOGE("flownet load_model(%s) failed", binPath.c_str());
        delete ctx;
        return 0;
    }

    LOGI(
            "flownet init took %lldms (vulkan=%d)",
            (long long) (nowMs() - t0),
            ctx->net.opt.use_vulkan_compute);

    return reinterpret_cast<jlong>(ctx);
}

// baseRgba/alterRgba: interleaved float B,G,R,A values in [0,255].
// flowOut: interleaved float [flowX, flowY].
extern "C" JNIEXPORT jboolean JNICALL
Java_com_particlesdevs_photoncamera_processing_ml_FlowNetNcnnProcessor_nativeRun(
        JNIEnv* env,
        jclass,
        jlong handle,
        jobject baseRgba,
        jobject alterRgba,
        jint width,
        jint height,
        jobject flowOut) {
    auto* ctx = reinterpret_cast<FlowNetCtx*>(handle);

    if (ctx == nullptr || width <= 0 || height <= 0) {
        return JNI_FALSE;
    }

    pinOpenMPThreads(ctx->net.opt.num_threads);

    const float* basePtr =
            static_cast<const float*>(env->GetDirectBufferAddress(baseRgba));
    const float* alterPtr =
            static_cast<const float*>(env->GetDirectBufferAddress(alterRgba));
    float* outPtr =
            static_cast<float*>(env->GetDirectBufferAddress(flowOut));

    if (basePtr == nullptr || alterPtr == nullptr || outPtr == nullptr) {
        LOGE("flownet GetDirectBufferAddress failed");
        return JNI_FALSE;
    }

    const int64_t planeLong = (int64_t) width * height;
    const int64_t inputBytes = planeLong * 4LL * (int64_t) sizeof(float);
    const int64_t outputBytes = planeLong * 2LL * (int64_t) sizeof(float);

    if (env->GetDirectBufferCapacity(baseRgba) < inputBytes ||
        env->GetDirectBufferCapacity(alterRgba) < inputBytes ||
        env->GetDirectBufferCapacity(flowOut) < outputBytes) {
        LOGE("flownet direct buffer too small");
        return JNI_FALSE;
    }

    const int plane = width * height;

    ncnn::Mat in0(width, height, 3);
    ncnn::Mat in1(width, height, 3);

    if (in0.empty() || in1.empty()) {
        LOGE("flownet input allocation failed");
        return JNI_FALSE;
    }

    float* c0 = static_cast<float*>(in0.channel(0));
    float* c1 = static_cast<float*>(in0.channel(1));
    float* c2 = static_cast<float*>(in0.channel(2));

    float* d0 = static_cast<float*>(in1.channel(0));
    float* d1 = static_cast<float*>(in1.channel(1));
    float* d2 = static_cast<float*>(in1.channel(2));

    for (int i = 0; i < plane; ++i) {
        c0[i] = basePtr[i * 4 + 0];
        c1[i] = basePtr[i * 4 + 1];
        c2[i] = basePtr[i * 4 + 2];

        d0[i] = alterPtr[i * 4 + 0];
        d1[i] = alterPtr[i * 4 + 1];
        d2[i] = alterPtr[i * 4 + 2];
    }

    int64_t tStart = nowMs();

    ncnn::Extractor ex = ctx->net.create_extractor();

    if (ex.input("in0", in0) != 0 || ex.input("in1", in1) != 0) {
        LOGE("flownet input failed");
        return JNI_FALSE;
    }

    ncnn::Mat out;
    const int ret = ex.extract("out0", out);

    if (ret != 0) {
        LOGE("flownet extract failed ret=%d", ret);
        return JNI_FALSE;
    }

    LOGI(
            "flownet forward %dx%d took %lld ms",
            width,
            height,
            (long long) (nowMs() - tStart));

    if (out.c < 2 || out.w != width || out.h != height) {
        LOGE(
                "unexpected flownet output dims=%d w=%d h=%d c=%d",
                out.dims,
                out.w,
                out.h,
                out.c);
        return JNI_FALSE;
    }

    const float* flowX = static_cast<const float*>(out.channel(0));
    const float* flowY = static_cast<const float*>(out.channel(1));

    for (int i = 0; i < plane; ++i) {
        outPtr[i * 2] = flowX[i];
        outPtr[i * 2 + 1] = flowY[i];
    }

    return JNI_TRUE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_particlesdevs_photoncamera_processing_ml_FlowNetNcnnProcessor_nativeDestroy(
        JNIEnv*,
        jclass,
        jlong handle) {
    auto* ctx = reinterpret_cast<FlowNetCtx*>(handle);
    delete ctx;
}

// ===========================================================================
// KernelNet
// ===========================================================================

struct KernelNetCtx {
    ncnn::Net net;

    int tileCore = 1024;
    int tileBorder = 32;
    bool stageTiming = false;

    ncnn::Mat grayTile;
    ncnn::Mat sigmaTile;
};

static jboolean kernelnetRunFull(
        KernelNetCtx* ctx,
        const float* grayPtr,
        int width,
        int height,
        float sigma,
        float* outPtr);

static jboolean kernelnetRunTiled(
        KernelNetCtx* ctx,
        const float* grayPtr,
        int width,
        int height,
        float sigma,
        float* outPtr);

extern "C" JNIEXPORT jlong JNICALL
Java_com_particlesdevs_photoncamera_processing_ml_KernelNetNcnnProcessor_nativeCreate(
        JNIEnv* env,
        jclass,
        jobject assetManager,
        jstring paramPath) {
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    if (mgr == nullptr) {
        LOGE("kernelnet AAssetManager_fromJava failed");
        return 0;
    }

    const char* path = env->GetStringUTFChars(paramPath, nullptr);
    if (path == nullptr) {
        LOGE("kernelnet paramPath null");
        return 0;
    }

    std::string paramStr = path;
    env->ReleaseStringUTFChars(paramPath, path);

    auto* ctx = new (std::nothrow) KernelNetCtx();
    if (ctx == nullptr) {
        LOGE("OOM allocating KernelNetCtx");
        return 0;
    }

    ctx->net.opt.use_vulkan_compute =
            getenv("KN_GPU") && getenv("KN_GPU")[0] == '1';
    ctx->net.opt.use_fp16_packed = true;
    ctx->net.opt.use_fp16_storage = true;
    ctx->net.opt.use_fp16_arithmetic = true;
    ctx->net.opt.use_bf16_storage = false;
    ctx->net.opt.use_subgroup_ops = false;
    ctx->net.opt.num_threads = 4;
    ctx->net.opt.lightmode = false;

    pinOpenMPThreads(ctx->net.opt.num_threads);

    if (getenv("KN_CPU") && getenv("KN_CPU")[0] == '1') {
        ctx->net.opt.use_vulkan_compute = false;
        LOGI("kernelnet: Vulkan disabled by KN_CPU=1");
    }

    if (getenv("KN_FP32") && getenv("KN_FP32")[0] == '1') {
        ctx->net.opt.use_fp16_packed = false;
        ctx->net.opt.use_fp16_storage = false;
        ctx->net.opt.use_fp16_arithmetic = false;
        LOGI("kernelnet: fp32 selected by KN_FP32=1");
    }

    const std::string binPath = paramToBinPath(paramStr);

    if (ctx->net.load_param(mgr, paramStr.c_str()) != 0) {
        LOGE("kernelnet load_param(%s) failed", paramStr.c_str());
        delete ctx;
        return 0;
    }

    if (ctx->net.load_model(mgr, binPath.c_str()) != 0) {
        LOGE("kernelnet load_model(%s) failed", binPath.c_str());
        delete ctx;
        return 0;
    }

    if (const char* tile = getenv("KN_TILE")) {
        const int value = atoi(tile);
        if (value >= 64) {
            ctx->tileCore = (value + 15) / 16 * 16;
        }
    }

    if (const char* border = getenv("KN_BORDER")) {
        const int value = atoi(border);
        if (value >= 8) {
            ctx->tileBorder = (value + 1) / 2 * 2;
        }
    }

    if (const char* timing = getenv("KN_STAGETIMING")) {
        ctx->stageTiming = strcmp(timing, "0") != 0;
    }

#if NCNN_VULKAN
    if (ctx->net.opt.use_vulkan_compute &&
        !(getenv("KN_NOALLOC") && getenv("KN_NOALLOC")[0] == '1')) {
        const ncnn::VulkanDevice* vkdev = ctx->net.vulkan_device();

        if (vkdev != nullptr) {
            ncnn::VkAllocator* blobPool = vkdev->acquire_blob_allocator();
            ctx->net.opt.blob_vkallocator = blobPool;
            ctx->net.opt.workspace_vkallocator = blobPool;
            ctx->net.opt.staging_vkallocator =
                    vkdev->acquire_staging_allocator();

            LOGI("kernelnet: persistent Vulkan allocators attached");
        } else {
            LOGE("kernelnet: Vulkan device unavailable");
        }
    }
#endif

    LOGI(
            "kernelnet model loaded (tile core=%d border=%d)",
            ctx->tileCore,
            ctx->tileBorder);

    return reinterpret_cast<jlong>(ctx);
}

// gray: one normalized luma plane, [width * height] floats.
// output: channel-major [s1][s2][rho].
extern "C" JNIEXPORT jboolean JNICALL
Java_com_particlesdevs_photoncamera_processing_ml_KernelNetNcnnProcessor_nativeRun(
        JNIEnv* env,
        jclass,
        jlong handle,
        jobject grayBuffer,
        jint width,
        jint height,
        jfloat sigma,
        jobject outBuffer) {
    auto* ctx = reinterpret_cast<KernelNetCtx*>(handle);

    if (ctx == nullptr || width <= 0 || height <= 0) {
        return JNI_FALSE;
    }

    pinOpenMPThreads(ctx->net.opt.num_threads);

    const float* grayPtr =
            static_cast<const float*>(env->GetDirectBufferAddress(grayBuffer));
    float* outPtr =
            static_cast<float*>(env->GetDirectBufferAddress(outBuffer));

    if (grayPtr == nullptr || outPtr == nullptr) {
        LOGE("kernelnet GetDirectBufferAddress failed");
        return JNI_FALSE;
    }

    if (getenv("KN_NOTILE") && getenv("KN_NOTILE")[0] == '1') {
        return kernelnetRunFull(
                ctx,
                grayPtr,
                width,
                height,
                sigma,
                outPtr);
    }

    return kernelnetRunTiled(
            ctx,
            grayPtr,
            width,
            height,
            sigma,
            outPtr);
}

static jboolean kernelnetRunFull(
        KernelNetCtx* ctx,
        const float* grayPtr,
        int width,
        int height,
        float sigma,
        float* outPtr) {
    const int plane = width * height;

    ncnn::Mat gray(width, height, 1);
    ncnn::Mat sigmaMat(width, height, 1);

    if (gray.empty() || sigmaMat.empty()) {
        LOGE("kernelnet full allocation failed");
        return JNI_FALSE;
    }

    memcpy(gray.data, grayPtr, static_cast<size_t>(plane) * sizeof(float));

    float* sigmaData = static_cast<float*>(sigmaMat.data);
    std::fill(sigmaData, sigmaData + plane, sigma);

    int64_t tStart = nowMs();

    ncnn::Extractor ex = ctx->net.create_extractor();

    if (ex.input("in0", gray) != 0 || ex.input("in1", sigmaMat) != 0) {
        LOGE("kernelnet full input failed");
        return JNI_FALSE;
    }

    ncnn::Mat out;

    if (ex.extract("out0", out) != 0) {
        LOGE("kernelnet full extract failed");
        return JNI_FALSE;
    }

    LOGI(
            "kernelnet forward %dx%d took %lld ms",
            width,
            height,
            (long long) (nowMs() - tStart));

    if (out.c != 3) {
        LOGE(
                "unexpected kernelnet output dims=%d w=%d h=%d c=%d",
                out.dims,
                out.w,
                out.h,
                out.c);
        return JNI_FALSE;
    }

    const int outPlane = out.w * out.h;

    for (int c = 0; c < 3; ++c) {
        const float* channel = static_cast<const float*>(out.channel(c));
        memcpy(
                outPtr + static_cast<size_t>(c) * outPlane,
                channel,
                static_cast<size_t>(outPlane) * sizeof(float));
    }

    return JNI_TRUE;
}

static void fillTileClamped(
        float* dst,
        const float* src,
        int width,
        int height,
        int x0,
        int y0,
        int tileSize) {
    for (int y = 0; y < tileSize; ++y) {
        int sourceY = y0 + y;
        if (sourceY > height - 1) {
            sourceY = height - 1;
        }

        const float* sourceRow =
                src + static_cast<size_t>(sourceY) * width;

        float* destinationRow =
                dst + static_cast<size_t>(y) * tileSize;

        const int copied =
                (x0 + tileSize <= width) ? tileSize : (width - x0);

        memcpy(
                destinationRow,
                sourceRow + x0,
                static_cast<size_t>(copied) * sizeof(float));

        if (copied < tileSize) {
            const float lastPixel = sourceRow[width - 1];

            for (int x = copied; x < tileSize; ++x) {
                destinationRow[x] = lastPixel;
            }
        }
    }
}

static jboolean kernelnetRunTiled(
        KernelNetCtx* ctx,
        const float* grayPtr,
        int width,
        int height,
        float sigma,
        float* outPtr) {
    const int border = ctx->tileBorder;
    const int core = ctx->tileCore;
    const int tileSize = core + border * 2;
    const int tileOut = tileSize / 2;

    const int outWidth = (width - 1) / 2 + 1;
    const int outHeight = (height - 1) / 2 + 1;
    const size_t outPlane =
            static_cast<size_t>(outWidth) * outHeight;

    if (ctx->grayTile.empty()) {
        ctx->grayTile.create(tileSize, tileSize, 1);
        ctx->sigmaTile.create(tileSize, tileSize, 1);
    }

    if (ctx->grayTile.empty() || ctx->sigmaTile.empty()) {
        LOGE("kernelnet tile allocation failed");
        return JNI_FALSE;
    }

    float* sigmaData = static_cast<float*>(ctx->sigmaTile.data);

    std::fill(
            sigmaData,
            sigmaData + static_cast<size_t>(tileSize) * tileSize,
            sigma);

    auto tileCount = [tileSize, core](int dimension) -> int {
        return dimension <= tileSize
                ? 1
                : (dimension - tileSize + core - 1) / core + 1;
    };

    const int nx = tileCount(width);
    const int ny = tileCount(height);

    int64_t started = nowMs();
    int64_t fillTime = 0;
    int64_t inputTime = 0;
    int64_t extractTime = 0;
    int64_t copyTime = 0;
    int64_t worstExtract = 0;

    for (int iy = 0; iy < ny; ++iy) {
        const int tileY = iy * core;
        const int validY0 = iy == 0 ? 0 : tileY + border;
        const int validY1 =
                iy == ny - 1 ? height : tileY + tileSize - border;

        for (int ix = 0; ix < nx; ++ix) {
            const int tileX = ix * core;
            const int validX0 = ix == 0 ? 0 : tileX + border;
            const int validX1 =
                    ix == nx - 1 ? width : tileX + tileSize - border;

            int64_t t0 = nowUs();

            fillTileClamped(
                    static_cast<float*>(ctx->grayTile.data),
                    grayPtr,
                    width,
                    height,
                    tileX,
                    tileY,
                    tileSize);

            int64_t t1 = nowUs();

            ncnn::Extractor ex = ctx->net.create_extractor();

            if (ex.input("in0", ctx->grayTile) != 0 ||
                ex.input("in1", ctx->sigmaTile) != 0) {
                LOGE("kernelnet tile input failed");
                return JNI_FALSE;
            }

            int64_t t2 = nowUs();

            ncnn::Mat out;

            if (ex.extract("out0", out) != 0) {
                LOGE("kernelnet tile extract failed");
                return JNI_FALSE;
            }

            int64_t t3 = nowUs();

            if (out.c != 3 || out.w != tileOut || out.h != tileOut) {
                LOGE(
                        "unexpected kernelnet tile output dims=%d w=%d h=%d c=%d",
                        out.dims,
                        out.w,
                        out.h,
                        out.c);
                return JNI_FALSE;
            }

            int64_t t4 = nowUs();

            const int globalX0 = validX0 / 2;
            const int globalX1 = (validX1 + 1) / 2;
            const int globalY0 = validY0 / 2;
            const int globalY1 = (validY1 + 1) / 2;

            const int localX0 = globalX0 - tileX / 2;
            const int localY0 = globalY0 - tileY / 2;
            const int rowLength = globalX1 - globalX0;

            for (int c = 0; c < 3; ++c) {
                const float* channel =
                        static_cast<const float*>(out.channel(c));

                float* destination =
                        outPtr + static_cast<size_t>(c) * outPlane;

                for (int y = 0; y < globalY1 - globalY0; ++y) {
                    memcpy(
                            destination +
                                    static_cast<size_t>(globalY0 + y) *
                                            outWidth +
                                    globalX0,
                            channel +
                                    static_cast<size_t>(localY0 + y) *
                                            tileOut +
                                    localX0,
                            static_cast<size_t>(rowLength) * sizeof(float));
                }
            }

            int64_t t5 = nowUs();

            fillTime += t1 - t0;
            inputTime += t2 - t1;
            extractTime += t3 - t2;
            copyTime += t5 - t4;

            if (t3 - t2 > worstExtract) {
                worstExtract = t3 - t2;
            }
        }
    }

    LOGI(
            "kernelnet tiled %dx%d -> %dx%d (%dx%d tiles of %dpx, core=%d border=%d) took %lld ms",
            width,
            height,
            outWidth,
            outHeight,
            nx,
            ny,
            tileSize,
            core,
            border,
            (long long) (nowMs() - started));

    if (ctx->stageTiming) {
        const int tileCountTotal = nx * ny;

        LOGI(
                "kernelnet stages (%d tiles): fill=%lldus input=%lldus extract=%lldus (worst %lldus) copy=%lldus",
                tileCountTotal,
                (long long) fillTime / tileCountTotal,
                (long long) inputTime / tileCountTotal,
                (long long) extractTime / tileCountTotal,
                (long long) worstExtract,
                (long long) copyTime / tileCountTotal);
    }

    return JNI_TRUE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_particlesdevs_photoncamera_processing_ml_KernelNetNcnnProcessor_nativeDestroy(
        JNIEnv*,
        jclass,
        jlong handle) {
    auto* ctx = reinterpret_cast<KernelNetCtx*>(handle);
    delete ctx;
}

// ===========================================================================
// FSRCNN-small x4, Y/luma-only
//
// Converted model contract:
//
//   Input blob:  in0
//   Output blob: out0
//
//   Input:  [1, H, W], float32 luma/Y in approximately [0, 1]
//   Output: [1, 4H, 4W], float32 upscaled luma/Y
//
// The model is FSRCNN(56, 12, 4), with a final stride-4 deconvolution.
// It is not an RGB model: do not allocate or copy three input/output planes.
// ===========================================================================

struct FsrcnnCtx {
    ncnn::Net net;

    // Source-pixel valid region per inference tile.
    int tileCore = 256;

    // The FSRCNN graph has an effective receptive-field radius smaller than
    // this. Eight source pixels provides a conservative seam-free margin.
    int tileBorder = 8;
};

static bool fsrcnnExtract(
        FsrcnnCtx* ctx,
        const ncnn::Mat& input,
        ncnn::Mat& output) {
    if (ctx == nullptr || input.empty()) {
        return false;
    }

    ncnn::Extractor ex = ctx->net.create_extractor();
    ex.set_light_mode(true);

    if (ex.input("in0", input) != 0) {
        LOGE("fsrcnn input(in0) failed");
        return false;
    }

    if (ex.extract("out0", output) != 0) {
        LOGE("fsrcnn extract(out0) failed");
        return false;
    }

    const int expectedWidth = input.w * 4;
    const int expectedHeight = input.h * 4;

    if (output.empty() ||
        output.c != 1 ||
        output.w != expectedWidth ||
        output.h != expectedHeight) {
        LOGE(
                "fsrcnn unexpected output dims=%d w=%d h=%d c=%d; expected 1x%dx%d",
                output.dims,
                output.w,
                output.h,
                output.c,
                expectedWidth,
                expectedHeight);
        return false;
    }

    return true;
}

static int fsrcnnReflect101(int value, int size) {
    if (size <= 1) {
        return 0;
    }

    while (value < 0 || value >= size) {
        if (value < 0) {
            value = -value;
        } else {
            value = size * 2 - value - 2;
        }
    }

    return value;
}

static jboolean fsrcnnRunFull(
        FsrcnnCtx* ctx,
        const float* input,
        int width,
        int height,
        float* output) {
    if (ctx == nullptr ||
        input == nullptr ||
        output == nullptr ||
        width <= 0 ||
        height <= 0) {
        return JNI_FALSE;
    }

    ncnn::Mat source(width, height, 1);

    if (source.empty()) {
        LOGE("fsrcnn full input allocation failed");
        return JNI_FALSE;
    }

    const size_t inputBytes =
            static_cast<size_t>(width) *
            static_cast<size_t>(height) *
            sizeof(float);

    memcpy(source.channel(0), input, inputBytes);

    ncnn::Mat result;

    int64_t started = nowMs();

    if (!fsrcnnExtract(ctx, source, result)) {
        return JNI_FALSE;
    }

    LOGI(
            "fsrcnn full forward %dx%d -> %dx%d took %lld ms",
            width,
            height,
            result.w,
            result.h,
            (long long) (nowMs() - started));

    const size_t outputBytes =
            static_cast<size_t>(result.w) *
            static_cast<size_t>(result.h) *
            sizeof(float);

    memcpy(output, result.channel(0), outputBytes);

    return JNI_TRUE;
}

static jboolean fsrcnnRunTiled(
        FsrcnnCtx* ctx,
        const float* input,
        int width,
        int height,
        float* output) {
    if (ctx == nullptr ||
        input == nullptr ||
        output == nullptr ||
        width <= 0 ||
        height <= 0) {
        return JNI_FALSE;
    }

    const int core = ctx->tileCore;
    const int border = ctx->tileBorder;

    if (core <= 0 || border < 0) {
        LOGE("fsrcnn invalid tile configuration");
        return JNI_FALSE;
    }

    const int outputWidth = width * 4;
    const int outputHeight = height * 4;

    int64_t started = nowMs();
    int tileCount = 0;

    for (int coreY = 0; coreY < height; coreY += core) {
        const int coreHeight = std::min(core, height - coreY);

        for (int coreX = 0; coreX < width; coreX += core) {
            const int coreWidth = std::min(core, width - coreX);

            const int tileWidth = coreWidth + border * 2;
            const int tileHeight = coreHeight + border * 2;

            ncnn::Mat tile(tileWidth, tileHeight, 1);

            if (tile.empty()) {
                LOGE("fsrcnn tile allocation failed");
                return JNI_FALSE;
            }

            float* tilePixels =
                    static_cast<float*>(tile.channel(0));

            for (int y = 0; y < tileHeight; ++y) {
                const int sourceY =
                        fsrcnnReflect101(coreY + y - border, height);

                const float* sourceRow =
                        input + static_cast<size_t>(sourceY) * width;

                float* destinationRow =
                        tilePixels + static_cast<size_t>(y) * tileWidth;

                for (int x = 0; x < tileWidth; ++x) {
                    const int sourceX =
                            fsrcnnReflect101(coreX + x - border, width);

                    destinationRow[x] = sourceRow[sourceX];
                }
            }

            ncnn::Mat tileOutput;

            if (!fsrcnnExtract(ctx, tile, tileOutput)) {
                return JNI_FALSE;
            }

            const int outputCropX = border * 4;
            const int outputCropY = border * 4;
            const int copiedWidth = coreWidth * 4;
            const int copiedHeight = coreHeight * 4;

            const float* tileOutputPixels =
                    static_cast<const float*>(tileOutput.channel(0));

            for (int y = 0; y < copiedHeight; ++y) {
                const float* sourceRow =
                        tileOutputPixels +
                        static_cast<size_t>(outputCropY + y) *
                                tileOutput.w +
                        outputCropX;

                float* destinationRow =
                        output +
                        static_cast<size_t>(coreY * 4 + y) *
                                outputWidth +
                        coreX * 4;

                memcpy(
                        destinationRow,
                        sourceRow,
                        static_cast<size_t>(copiedWidth) * sizeof(float));
            }

            ++tileCount;
        }
    }

    LOGI(
            "fsrcnn tiled %dx%d -> %dx%d; %d tiles; core=%d border=%d; took %lld ms",
            width,
            height,
            outputWidth,
            outputHeight,
            tileCount,
            core,
            border,
            (long long) (nowMs() - started));

    return JNI_TRUE;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_particlesdevs_photoncamera_processing_ml_FsrcnnNcnnProcessor_nativeCreate(
        JNIEnv* env,
        jclass,
        jobject assetManager,
        jstring paramPath) {
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    if (mgr == nullptr) {
        LOGE("fsrcnn AAssetManager_fromJava failed");
        return 0;
    }

    const char* path = env->GetStringUTFChars(paramPath, nullptr);

    if (path == nullptr) {
        LOGE("fsrcnn paramPath null");
        return 0;
    }

    std::string paramStr = path;
    env->ReleaseStringUTFChars(paramPath, path);

    auto* ctx = new (std::nothrow) FsrcnnCtx();

    if (ctx == nullptr) {
        LOGE("fsrcnn OOM allocating context");
        return 0;
    }

    /*
     * Correctness-first configuration:
     *
     * - CPU float32 avoids device-specific Vulkan/fp16 differences while
     *   validating the newly converted model.
     * - FSRCNN is small enough that this is an acceptable baseline.
     *
     * Optional debug environment settings:
     *
     *   FSRCNN_GPU=1   Enable Vulkan inference.
     *   FSRCNN_FP16=1  Enable fp16 storage/arithmetic when Vulkan is enabled.
     *   FSRCNN_TILE=N  Set core tile size, minimum 64 pixels.
     *   FSRCNN_BORDER=N Set source-pixel tile overlap, minimum 4 pixels.
     *   FSRCNN_NOTILE=1 Force one full-resolution inference.
     */
    ctx->net.opt.use_vulkan_compute =
            getenv("FSRCNN_GPU") && getenv("FSRCNN_GPU")[0] == '1';

    const bool enableFp16 =
            ctx->net.opt.use_vulkan_compute &&
            getenv("FSRCNN_FP16") &&
            getenv("FSRCNN_FP16")[0] == '1';

    ctx->net.opt.use_fp16_packed = enableFp16;
    ctx->net.opt.use_fp16_storage = enableFp16;
    ctx->net.opt.use_fp16_arithmetic = enableFp16;
    ctx->net.opt.use_bf16_storage = false;
    ctx->net.opt.use_subgroup_ops = false;
    ctx->net.opt.num_threads = 4;
    ctx->net.opt.lightmode = true;

    pinOpenMPThreads(ctx->net.opt.num_threads);

    if (const char* value = getenv("FSRCNN_TILE")) {
        const int requested = atoi(value);
        if (requested >= 64) {
            ctx->tileCore = requested;
        }
    }

    if (const char* value = getenv("FSRCNN_BORDER")) {
        const int requested = atoi(value);
        if (requested >= 4) {
            ctx->tileBorder = requested;
        }
    }

    const std::string binPath = paramToBinPath(paramStr);

    if (ctx->net.load_param(mgr, paramStr.c_str()) != 0) {
        LOGE("fsrcnn load_param(%s) failed", paramStr.c_str());
        delete ctx;
        return 0;
    }

    if (ctx->net.load_model(mgr, binPath.c_str()) != 0) {
        LOGE("fsrcnn load_model(%s) failed", binPath.c_str());
        delete ctx;
        return 0;
    }

#if NCNN_VULKAN
    if (ctx->net.opt.use_vulkan_compute) {
        const ncnn::VulkanDevice* vkdev = ctx->net.vulkan_device();

        if (vkdev != nullptr) {
            ncnn::VkAllocator* blobPool =
                    vkdev->acquire_blob_allocator();

            ctx->net.opt.blob_vkallocator = blobPool;
            ctx->net.opt.workspace_vkallocator = blobPool;
            ctx->net.opt.staging_vkallocator =
                    vkdev->acquire_staging_allocator();

            LOGI(
                    "fsrcnn Vulkan enabled; fp16=%d; pooled allocators attached",
                    enableFp16);
        } else {
            LOGE("fsrcnn requested Vulkan but no Vulkan device is available");
        }
    }
#endif

    LOGI(
            "fsrcnn model loaded: param=%s bin=%s; backend=%s; tile=%d border=%d",
            paramStr.c_str(),
            binPath.c_str(),
            ctx->net.opt.use_vulkan_compute ? "Vulkan" : "CPU",
            ctx->tileCore,
            ctx->tileBorder);

    return reinterpret_cast<jlong>(ctx);
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_particlesdevs_photoncamera_processing_ml_FsrcnnNcnnProcessor_nativeRun(
        JNIEnv* env,
        jclass,
        jlong handle,
        jobject yIn,
        jint width,
        jint height,
        jobject yOut) {
    auto* ctx = reinterpret_cast<FsrcnnCtx*>(handle);

    if (ctx == nullptr || width <= 0 || height <= 0) {
        return JNI_FALSE;
    }

    /*
     * Prevent both integer overflow in the 4x output shape and impossible
     * direct-buffer requirements.
     */
    if (width > INT_MAX / 4 || height > INT_MAX / 4) {
        LOGE("fsrcnn source dimensions overflow 4x output");
        return JNI_FALSE;
    }

    pinOpenMPThreads(ctx->net.opt.num_threads);

    const float* input =
            static_cast<const float*>(env->GetDirectBufferAddress(yIn));

    float* output =
            static_cast<float*>(env->GetDirectBufferAddress(yOut));

    if (input == nullptr || output == nullptr) {
        LOGE("fsrcnn requires direct input and output ByteBuffers");
        return JNI_FALSE;
    }

    const int64_t inputBytes =
            static_cast<int64_t>(width) *
            static_cast<int64_t>(height) *
            static_cast<int64_t>(sizeof(float));

    const int64_t outputBytes =
            static_cast<int64_t>(width) * 4LL *
            static_cast<int64_t>(height) * 4LL *
            static_cast<int64_t>(sizeof(float));

    if (env->GetDirectBufferCapacity(yIn) < inputBytes ||
        env->GetDirectBufferCapacity(yOut) < outputBytes) {
        LOGE(
                "fsrcnn direct buffer too small: input=%lld required=%lld; output=%lld required=%lld",
                (long long) env->GetDirectBufferCapacity(yIn),
                (long long) inputBytes,
                (long long) env->GetDirectBufferCapacity(yOut),
                (long long) outputBytes);
        return JNI_FALSE;
    }

    const bool forceFull =
            getenv("FSRCNN_NOTILE") &&
            getenv("FSRCNN_NOTILE")[0] == '1';

    if (forceFull ||
        (width <= ctx->tileCore && height <= ctx->tileCore)) {
        return fsrcnnRunFull(ctx, input, width, height, output);
    }

    return fsrcnnRunTiled(ctx, input, width, height, output);
}

extern "C" JNIEXPORT void JNICALL
Java_com_particlesdevs_photoncamera_processing_ml_FsrcnnNcnnProcessor_nativeDestroy(
        JNIEnv*,
        jclass,
        jlong handle) {
    auto* ctx = reinterpret_cast<FsrcnnCtx*>(handle);
    delete ctx;
}
