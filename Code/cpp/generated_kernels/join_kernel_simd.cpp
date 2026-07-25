#pragma once
#include <arm_neon.h>

static inline void join_one_batch_simd(const float32x4_t* x, const float32x4_t* y, float32x4_t* o) {
    o[0] = vdupq_n_f32(0.0f);
    o[0] = vfmaq_f32(o[0], x[0], y[15]);
    o[0] = vfmaq_f32(o[0], x[1], y[14]);
    o[0] = vfmsq_f32(o[0], x[2], y[13]);
    o[0] = vfmaq_f32(o[0], x[3], y[12]);
    o[0] = vfmsq_f32(o[0], x[4], y[11]);
    o[0] = vfmaq_f32(o[0], x[5], y[10]);
    o[0] = vfmsq_f32(o[0], x[6], y[9]);
    o[0] = vfmaq_f32(o[0], x[7], y[8]);
    o[0] = vfmaq_f32(o[0], x[8], y[7]);
    o[0] = vfmsq_f32(o[0], x[9], y[6]);
    o[0] = vfmaq_f32(o[0], x[10], y[5]);
    o[0] = vfmaq_f32(o[0], x[11], y[4]);
    o[0] = vfmsq_f32(o[0], x[12], y[3]);
    o[0] = vfmaq_f32(o[0], x[13], y[2]);
    o[0] = vfmsq_f32(o[0], x[14], y[1]);
    o[0] = vfmaq_f32(o[0], x[15], y[0]);

    o[1] = vdupq_n_f32(0.0f);
    o[1] = vfmsq_f32(o[1], x[1], y[15]);
    o[1] = vfmsq_f32(o[1], x[5], y[13]);
    o[1] = vfmaq_f32(o[1], x[6], y[12]);
    o[1] = vfmsq_f32(o[1], x[7], y[11]);
    o[1] = vfmsq_f32(o[1], x[11], y[7]);
    o[1] = vfmaq_f32(o[1], x[12], y[6]);
    o[1] = vfmsq_f32(o[1], x[13], y[5]);
    o[1] = vfmsq_f32(o[1], x[15], y[1]);

    o[2] = vdupq_n_f32(0.0f);
    o[2] = vfmsq_f32(o[2], x[2], y[15]);
    o[2] = vfmsq_f32(o[2], x[5], y[14]);
    o[2] = vfmaq_f32(o[2], x[8], y[12]);
    o[2] = vfmsq_f32(o[2], x[9], y[11]);
    o[2] = vfmsq_f32(o[2], x[11], y[9]);
    o[2] = vfmaq_f32(o[2], x[12], y[8]);
    o[2] = vfmsq_f32(o[2], x[14], y[5]);
    o[2] = vfmsq_f32(o[2], x[15], y[2]);

    o[3] = vdupq_n_f32(0.0f);
    o[3] = vfmsq_f32(o[3], x[3], y[15]);
    o[3] = vfmsq_f32(o[3], x[6], y[14]);
    o[3] = vfmaq_f32(o[3], x[8], y[13]);
    o[3] = vfmsq_f32(o[3], x[10], y[11]);
    o[3] = vfmsq_f32(o[3], x[11], y[10]);
    o[3] = vfmaq_f32(o[3], x[13], y[8]);
    o[3] = vfmsq_f32(o[3], x[14], y[6]);
    o[3] = vfmsq_f32(o[3], x[15], y[3]);

    o[4] = vdupq_n_f32(0.0f);
    o[4] = vfmsq_f32(o[4], x[4], y[15]);
    o[4] = vfmsq_f32(o[4], x[7], y[14]);
    o[4] = vfmaq_f32(o[4], x[9], y[13]);
    o[4] = vfmsq_f32(o[4], x[10], y[12]);
    o[4] = vfmsq_f32(o[4], x[12], y[10]);
    o[4] = vfmaq_f32(o[4], x[13], y[9]);
    o[4] = vfmsq_f32(o[4], x[14], y[7]);
    o[4] = vfmsq_f32(o[4], x[15], y[4]);

    o[5] = vdupq_n_f32(0.0f);
    o[5] = vfmaq_f32(o[5], x[5], y[15]);
    o[5] = vfmaq_f32(o[5], x[11], y[12]);
    o[5] = vfmsq_f32(o[5], x[12], y[11]);
    o[5] = vfmaq_f32(o[5], x[15], y[5]);

    o[6] = vdupq_n_f32(0.0f);
    o[6] = vfmaq_f32(o[6], x[6], y[15]);
    o[6] = vfmaq_f32(o[6], x[11], y[13]);
    o[6] = vfmsq_f32(o[6], x[13], y[11]);
    o[6] = vfmaq_f32(o[6], x[15], y[6]);

    o[7] = vdupq_n_f32(0.0f);
    o[7] = vfmaq_f32(o[7], x[7], y[15]);
    o[7] = vfmaq_f32(o[7], x[12], y[13]);
    o[7] = vfmsq_f32(o[7], x[13], y[12]);
    o[7] = vfmaq_f32(o[7], x[15], y[7]);

    o[8] = vdupq_n_f32(0.0f);
    o[8] = vfmaq_f32(o[8], x[8], y[15]);
    o[8] = vfmaq_f32(o[8], x[11], y[14]);
    o[8] = vfmsq_f32(o[8], x[14], y[11]);
    o[8] = vfmaq_f32(o[8], x[15], y[8]);

    o[9] = vdupq_n_f32(0.0f);
    o[9] = vfmaq_f32(o[9], x[9], y[15]);
    o[9] = vfmaq_f32(o[9], x[12], y[14]);
    o[9] = vfmsq_f32(o[9], x[14], y[12]);
    o[9] = vfmaq_f32(o[9], x[15], y[9]);

    o[10] = vdupq_n_f32(0.0f);
    o[10] = vfmaq_f32(o[10], x[10], y[15]);
    o[10] = vfmaq_f32(o[10], x[13], y[14]);
    o[10] = vfmsq_f32(o[10], x[14], y[13]);
    o[10] = vfmaq_f32(o[10], x[15], y[10]);

    o[11] = vdupq_n_f32(0.0f);
    o[11] = vfmsq_f32(o[11], x[11], y[15]);
    o[11] = vfmsq_f32(o[11], x[15], y[11]);

    o[12] = vdupq_n_f32(0.0f);
    o[12] = vfmsq_f32(o[12], x[12], y[15]);
    o[12] = vfmsq_f32(o[12], x[15], y[12]);

    o[13] = vdupq_n_f32(0.0f);
    o[13] = vfmsq_f32(o[13], x[13], y[15]);
    o[13] = vfmsq_f32(o[13], x[15], y[13]);

    o[14] = vdupq_n_f32(0.0f);
    o[14] = vfmsq_f32(o[14], x[14], y[15]);
    o[14] = vfmsq_f32(o[14], x[15], y[14]);

    o[15] = vdupq_n_f32(0.0f);
    o[15] = vfmaq_f32(o[15], x[15], y[15]);

}
