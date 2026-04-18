#include <math.h>
#include <stddef.h>

#if defined(_WIN32)
#define SDXL_ACCEL_EXPORT __declspec(dllexport)
#else
#define SDXL_ACCEL_EXPORT __attribute__((visibility("default")))
#endif

static int validate_rw(const float* input, float* output) {
    return (input != NULL && output != NULL) ? 0 : -1;
}

static int validate_rw2(const float* input_a, const float* input_b, float* output) {
    return (input_a != NULL && input_b != NULL && output != NULL) ? 0 : -1;
}

static int validate_rw3(const float* input_a, const float* input_b, const float* input_c, float* output) {
    return (input_a != NULL && input_b != NULL && input_c != NULL && output != NULL) ? 0 : -1;
}

SDXL_ACCEL_EXPORT int sdxl_scale_model_input_f32(const float* sample,
                                                 float* output,
                                                 size_t count,
                                                 float sigma) {
    if (validate_rw(sample, output) != 0) {
        return -1;
    }

    const float inv = 1.0f / sqrtf(sigma * sigma + 1.0f);
    for (size_t i = 0; i < count; ++i) {
        output[i] = sample[i] * inv;
    }
    return 0;
}

SDXL_ACCEL_EXPORT int sdxl_euler_step_f32(const float* sample,
                                          const float* model_output,
                                          float* output,
                                          size_t count,
                                          float sigma,
                                          float sigma_next) {
    if (validate_rw2(sample, model_output, output) != 0) {
        return -1;
    }

    (void)sigma;
    const float delta = sigma_next - sigma;
    for (size_t i = 0; i < count; ++i) {
        output[i] = sample[i] + delta * model_output[i];
    }
    return 0;
}

SDXL_ACCEL_EXPORT int sdxl_cfg_euler_step_f32(const float* cond,
                                              const float* uncond,
                                              const float* sample,
                                              float* output,
                                              size_t count,
                                              float cfg_scale,
                                              float sigma,
                                              float sigma_next) {
    if (validate_rw3(cond, uncond, sample, output) != 0) {
        return -1;
    }

    (void)sigma;
    const float delta = sigma_next - sigma;
    for (size_t i = 0; i < count; ++i) {
        const float guided = uncond[i] + cfg_scale * (cond[i] - uncond[i]);
        output[i] = sample[i] + delta * guided;
    }
    return 0;
}

SDXL_ACCEL_EXPORT int sdxl_nchw_to_nhwc_scaled_f32(const float* src,
                                                   float* output,
                                                   int n,
                                                   int c,
                                                   int h,
                                                   int w,
                                                   float scale) {
    if (validate_rw(src, output) != 0) {
        return -1;
    }
    if (n <= 0 || c <= 0 || h <= 0 || w <= 0) {
        return -2;
    }

    const size_t hw = (size_t)h * (size_t)w;
    const size_t chw = (size_t)c * hw;
    const size_t hwc = hw * (size_t)c;

    for (int batch = 0; batch < n; ++batch) {
        const size_t batch_src_offset = (size_t)batch * chw;
        const size_t batch_out_offset = (size_t)batch * hwc;
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const size_t spatial = (size_t)y * (size_t)w + (size_t)x;
                const size_t out_base = batch_out_offset + spatial * (size_t)c;
                for (int channel = 0; channel < c; ++channel) {
                    const size_t src_index = batch_src_offset + (size_t)channel * hw + spatial;
                    output[out_base + (size_t)channel] = src[src_index] * scale;
                }
            }
        }
    }
    return 0;
}
