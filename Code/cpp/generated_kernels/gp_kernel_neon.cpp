#include <arm_neon.h>

// The Main Kernel
static inline void gp_kernel_four_batch(
    const float* __restrict__ x_row,
    const float* __restrict__ y_row,
    float* __restrict__ o_row
) {
    // Prefetch next call's data
    __builtin_prefetch(x_row + 64, 0, 1);
    __builtin_prefetch(y_row + 64, 0, 1);
    float32x4_t x[16];
    float32x4_t y[16];
    float32x4_t o[16];

    // Interleaved transpose, x and y loaded together each iteration
    for (int i = 0; i < 4; ++i) {
        const int offset = i * 4;

        // --- x ---
        float32x4_t rx0 = vld1q_f32(x_row + offset);
        float32x4_t rx1 = vld1q_f32(x_row + 16 + offset);
        float32x4_t rx2 = vld1q_f32(x_row + 32 + offset);
        float32x4_t rx3 = vld1q_f32(x_row + 48 + offset);

        float32x4_t xq0 = vzip1q_f32(rx0, rx1);
        float32x4_t xq2 = vzip2q_f32(rx0, rx1);
        float32x4_t xq1 = vzip1q_f32(rx2, rx3);
        float32x4_t xq3 = vzip2q_f32(rx2, rx3);

        x[offset] = vreinterpretq_f32_f64(vzip1q_f64(vreinterpretq_f64_f32(xq0), vreinterpretq_f64_f32(xq1)));
        x[offset + 1] = vreinterpretq_f32_f64(vzip2q_f64(vreinterpretq_f64_f32(xq0), vreinterpretq_f64_f32(xq1)));
        x[offset + 2] = vreinterpretq_f32_f64(vzip1q_f64(vreinterpretq_f64_f32(xq2), vreinterpretq_f64_f32(xq3)));
        x[offset + 3] = vreinterpretq_f32_f64(vzip2q_f64(vreinterpretq_f64_f32(xq2), vreinterpretq_f64_f32(xq3)));

        // --- y part
        float32x4_t ry0 = vld1q_f32(y_row + offset);
        float32x4_t ry1 = vld1q_f32(y_row + 16 + offset);
        float32x4_t ry2 = vld1q_f32(y_row + 32 + offset);
        float32x4_t ry3 = vld1q_f32(y_row + 48 + offset);

        float32x4_t yq0 = vzip1q_f32(ry0, ry1);
        float32x4_t yq2 = vzip2q_f32(ry0, ry1);
        float32x4_t yq1 = vzip1q_f32(ry2, ry3);
        float32x4_t yq3 = vzip2q_f32(ry2, ry3);

        y[offset] = vreinterpretq_f32_f64(vzip1q_f64(vreinterpretq_f64_f32(yq0), vreinterpretq_f64_f32(yq1)));
        y[offset + 1] = vreinterpretq_f32_f64(vzip2q_f64(vreinterpretq_f64_f32(yq0), vreinterpretq_f64_f32(yq1)));
        y[offset + 2] = vreinterpretq_f32_f64(vzip1q_f64(vreinterpretq_f64_f32(yq2), vreinterpretq_f64_f32(yq3)));
        y[offset + 3] = vreinterpretq_f32_f64(vzip2q_f64(vreinterpretq_f64_f32(yq2), vreinterpretq_f64_f32(yq3)));
    }

    // GP computation: first term uses vmulq (no zero-init), rest are fma/fms
    o[0] = vmulq_f32(x[0], y[0]);
    o[0] = vfmaq_f32(o[0], x[2], y[2]);
    o[0] = vfmaq_f32(o[0], x[3], y[3]);
    o[0] = vfmaq_f32(o[0], x[4], y[4]);
    o[0] = vfmsq_f32(o[0], x[8], y[8]);
    o[0] = vfmsq_f32(o[0], x[9], y[9]);
    o[0] = vfmsq_f32(o[0], x[10], y[10]);
    o[0] = vfmsq_f32(o[0], x[14], y[14]);

    o[1] = vmulq_f32(x[0], y[1]);
    o[1] = vfmaq_f32(o[1], x[1], y[0]);
    o[1] = vfmsq_f32(o[1], x[2], y[5]);
    o[1] = vfmsq_f32(o[1], x[3], y[6]);
    o[1] = vfmsq_f32(o[1], x[4], y[7]);
    o[1] = vfmaq_f32(o[1], x[5], y[2]);
    o[1] = vfmaq_f32(o[1], x[6], y[3]);
    o[1] = vfmaq_f32(o[1], x[7], y[4]);
    o[1] = vfmsq_f32(o[1], x[8], y[11]);
    o[1] = vfmsq_f32(o[1], x[9], y[12]);
    o[1] = vfmsq_f32(o[1], x[10], y[13]);
    o[1] = vfmsq_f32(o[1], x[11], y[8]);
    o[1] = vfmsq_f32(o[1], x[12], y[9]);
    o[1] = vfmsq_f32(o[1], x[13], y[10]);
    o[1] = vfmaq_f32(o[1], x[14], y[15]);
    o[1] = vfmsq_f32(o[1], x[15], y[14]);

    o[2] = vmulq_f32(x[0], y[2]);
    o[2] = vfmaq_f32(o[2], x[2], y[0]);
    o[2] = vfmsq_f32(o[2], x[3], y[8]);
    o[2] = vfmsq_f32(o[2], x[4], y[9]);
    o[2] = vfmaq_f32(o[2], x[8], y[3]);
    o[2] = vfmaq_f32(o[2], x[9], y[4]);
    o[2] = vfmsq_f32(o[2], x[10], y[14]);
    o[2] = vfmsq_f32(o[2], x[14], y[10]);

    o[3] = vmulq_f32(x[0], y[3]);
    o[3] = vfmaq_f32(o[3], x[2], y[8]);
    o[3] = vfmaq_f32(o[3], x[3], y[0]);
    o[3] = vfmsq_f32(o[3], x[4], y[10]);
    o[3] = vfmsq_f32(o[3], x[8], y[2]);
    o[3] = vfmaq_f32(o[3], x[9], y[14]);
    o[3] = vfmaq_f32(o[3], x[10], y[4]);
    o[3] = vfmaq_f32(o[3], x[14], y[9]);

    o[4] = vmulq_f32(x[0], y[4]);
    o[4] = vfmaq_f32(o[4], x[2], y[9]);
    o[4] = vfmaq_f32(o[4], x[3], y[10]);
    o[4] = vfmaq_f32(o[4], x[4], y[0]);
    o[4] = vfmsq_f32(o[4], x[8], y[14]);
    o[4] = vfmsq_f32(o[4], x[9], y[2]);
    o[4] = vfmsq_f32(o[4], x[10], y[3]);
    o[4] = vfmsq_f32(o[4], x[14], y[8]);

    o[5] = vmulq_f32(x[0], y[5]);
    o[5] = vfmaq_f32(o[5], x[1], y[2]);
    o[5] = vfmsq_f32(o[5], x[2], y[1]);
    o[5] = vfmaq_f32(o[5], x[3], y[11]);
    o[5] = vfmaq_f32(o[5], x[4], y[12]);
    o[5] = vfmaq_f32(o[5], x[5], y[0]);
    o[5] = vfmsq_f32(o[5], x[6], y[8]);
    o[5] = vfmsq_f32(o[5], x[7], y[9]);
    o[5] = vfmaq_f32(o[5], x[8], y[6]);
    o[5] = vfmaq_f32(o[5], x[9], y[7]);
    o[5] = vfmsq_f32(o[5], x[10], y[15]);
    o[5] = vfmaq_f32(o[5], x[11], y[3]);
    o[5] = vfmaq_f32(o[5], x[12], y[4]);
    o[5] = vfmsq_f32(o[5], x[13], y[14]);
    o[5] = vfmaq_f32(o[5], x[14], y[13]);
    o[5] = vfmsq_f32(o[5], x[15], y[10]);

    o[6] = vmulq_f32(x[0], y[6]);
    o[6] = vfmaq_f32(o[6], x[1], y[3]);
    o[6] = vfmsq_f32(o[6], x[2], y[11]);
    o[6] = vfmsq_f32(o[6], x[3], y[1]);
    o[6] = vfmaq_f32(o[6], x[4], y[13]);
    o[6] = vfmaq_f32(o[6], x[5], y[8]);
    o[6] = vfmaq_f32(o[6], x[6], y[0]);
    o[6] = vfmsq_f32(o[6], x[7], y[10]);
    o[6] = vfmsq_f32(o[6], x[8], y[5]);
    o[6] = vfmaq_f32(o[6], x[9], y[15]);
    o[6] = vfmaq_f32(o[6], x[10], y[7]);
    o[6] = vfmsq_f32(o[6], x[11], y[2]);
    o[6] = vfmaq_f32(o[6], x[12], y[14]);
    o[6] = vfmaq_f32(o[6], x[13], y[4]);
    o[6] = vfmsq_f32(o[6], x[14], y[12]);
    o[6] = vfmaq_f32(o[6], x[15], y[9]);

    o[7] = vmulq_f32(x[0], y[7]);
    o[7] = vfmaq_f32(o[7], x[1], y[4]);
    o[7] = vfmsq_f32(o[7], x[2], y[12]);
    o[7] = vfmsq_f32(o[7], x[3], y[13]);
    o[7] = vfmsq_f32(o[7], x[4], y[1]);
    o[7] = vfmaq_f32(o[7], x[5], y[9]);
    o[7] = vfmaq_f32(o[7], x[6], y[10]);
    o[7] = vfmaq_f32(o[7], x[7], y[0]);
    o[7] = vfmsq_f32(o[7], x[8], y[15]);
    o[7] = vfmsq_f32(o[7], x[9], y[5]);
    o[7] = vfmsq_f32(o[7], x[10], y[6]);
    o[7] = vfmsq_f32(o[7], x[11], y[14]);
    o[7] = vfmsq_f32(o[7], x[12], y[2]);
    o[7] = vfmsq_f32(o[7], x[13], y[3]);
    o[7] = vfmaq_f32(o[7], x[14], y[11]);
    o[7] = vfmsq_f32(o[7], x[15], y[8]);

    o[8] = vmulq_f32(x[0], y[8]);
    o[8] = vfmaq_f32(o[8], x[2], y[3]);
    o[8] = vfmsq_f32(o[8], x[3], y[2]);
    o[8] = vfmaq_f32(o[8], x[4], y[14]);
    o[8] = vfmaq_f32(o[8], x[8], y[0]);
    o[8] = vfmsq_f32(o[8], x[9], y[10]);
    o[8] = vfmaq_f32(o[8], x[10], y[9]);
    o[8] = vfmaq_f32(o[8], x[14], y[4]);

    o[9] = vmulq_f32(x[0], y[9]);
    o[9] = vfmaq_f32(o[9], x[2], y[4]);
    o[9] = vfmsq_f32(o[9], x[3], y[14]);
    o[9] = vfmsq_f32(o[9], x[4], y[2]);
    o[9] = vfmaq_f32(o[9], x[8], y[10]);
    o[9] = vfmaq_f32(o[9], x[9], y[0]);
    o[9] = vfmsq_f32(o[9], x[10], y[8]);
    o[9] = vfmsq_f32(o[9], x[14], y[3]);

    o[10] = vmulq_f32(x[0], y[10]);
    o[10] = vfmaq_f32(o[10], x[2], y[14]);
    o[10] = vfmaq_f32(o[10], x[3], y[4]);
    o[10] = vfmsq_f32(o[10], x[4], y[3]);
    o[10] = vfmsq_f32(o[10], x[8], y[9]);
    o[10] = vfmaq_f32(o[10], x[9], y[8]);
    o[10] = vfmaq_f32(o[10], x[10], y[0]);
    o[10] = vfmaq_f32(o[10], x[14], y[2]);

    o[11] = vmulq_f32(x[0], y[11]);
    o[11] = vfmaq_f32(o[11], x[1], y[8]);
    o[11] = vfmsq_f32(o[11], x[2], y[6]);
    o[11] = vfmaq_f32(o[11], x[3], y[5]);
    o[11] = vfmsq_f32(o[11], x[4], y[15]);
    o[11] = vfmaq_f32(o[11], x[5], y[3]);
    o[11] = vfmsq_f32(o[11], x[6], y[2]);
    o[11] = vfmaq_f32(o[11], x[7], y[14]);
    o[11] = vfmaq_f32(o[11], x[8], y[1]);
    o[11] = vfmsq_f32(o[11], x[9], y[13]);
    o[11] = vfmaq_f32(o[11], x[10], y[12]);
    o[11] = vfmaq_f32(o[11], x[11], y[0]);
    o[11] = vfmsq_f32(o[11], x[12], y[10]);
    o[11] = vfmaq_f32(o[11], x[13], y[9]);
    o[11] = vfmsq_f32(o[11], x[14], y[7]);
    o[11] = vfmaq_f32(o[11], x[15], y[4]);

    o[12] = vmulq_f32(x[0], y[12]);
    o[12] = vfmaq_f32(o[12], x[1], y[9]);
    o[12] = vfmsq_f32(o[12], x[2], y[7]);
    o[12] = vfmaq_f32(o[12], x[3], y[15]);
    o[12] = vfmaq_f32(o[12], x[4], y[5]);
    o[12] = vfmaq_f32(o[12], x[5], y[4]);
    o[12] = vfmsq_f32(o[12], x[6], y[14]);
    o[12] = vfmsq_f32(o[12], x[7], y[2]);
    o[12] = vfmaq_f32(o[12], x[8], y[13]);
    o[12] = vfmaq_f32(o[12], x[9], y[1]);
    o[12] = vfmsq_f32(o[12], x[10], y[11]);
    o[12] = vfmaq_f32(o[12], x[11], y[10]);
    o[12] = vfmaq_f32(o[12], x[12], y[0]);
    o[12] = vfmsq_f32(o[12], x[13], y[8]);
    o[12] = vfmaq_f32(o[12], x[14], y[6]);
    o[12] = vfmsq_f32(o[12], x[15], y[3]);

    o[13] = vmulq_f32(x[0], y[13]);
    o[13] = vfmaq_f32(o[13], x[1], y[10]);
    o[13] = vfmsq_f32(o[13], x[2], y[15]);
    o[13] = vfmsq_f32(o[13], x[3], y[7]);
    o[13] = vfmaq_f32(o[13], x[4], y[6]);
    o[13] = vfmaq_f32(o[13], x[5], y[14]);
    o[13] = vfmaq_f32(o[13], x[6], y[4]);
    o[13] = vfmsq_f32(o[13], x[7], y[3]);
    o[13] = vfmsq_f32(o[13], x[8], y[12]);
    o[13] = vfmaq_f32(o[13], x[9], y[11]);
    o[13] = vfmaq_f32(o[13], x[10], y[1]);
    o[13] = vfmsq_f32(o[13], x[11], y[9]);
    o[13] = vfmaq_f32(o[13], x[12], y[8]);
    o[13] = vfmaq_f32(o[13], x[13], y[0]);
    o[13] = vfmsq_f32(o[13], x[14], y[5]);
    o[13] = vfmaq_f32(o[13], x[15], y[2]);

    o[14] = vmulq_f32(x[0], y[14]);
    o[14] = vfmaq_f32(o[14], x[2], y[10]);
    o[14] = vfmsq_f32(o[14], x[3], y[9]);
    o[14] = vfmaq_f32(o[14], x[4], y[8]);
    o[14] = vfmaq_f32(o[14], x[8], y[4]);
    o[14] = vfmsq_f32(o[14], x[9], y[3]);
    o[14] = vfmaq_f32(o[14], x[10], y[2]);
    o[14] = vfmaq_f32(o[14], x[14], y[0]);

    o[15] = vmulq_f32(x[0], y[15]);
    o[15] = vfmaq_f32(o[15], x[1], y[14]);
    o[15] = vfmsq_f32(o[15], x[2], y[13]);
    o[15] = vfmaq_f32(o[15], x[3], y[12]);
    o[15] = vfmsq_f32(o[15], x[4], y[11]);
    o[15] = vfmaq_f32(o[15], x[5], y[10]);
    o[15] = vfmsq_f32(o[15], x[6], y[9]);
    o[15] = vfmaq_f32(o[15], x[7], y[8]);
    o[15] = vfmaq_f32(o[15], x[8], y[7]);
    o[15] = vfmsq_f32(o[15], x[9], y[6]);
    o[15] = vfmaq_f32(o[15], x[10], y[5]);
    o[15] = vfmaq_f32(o[15], x[11], y[4]);
    o[15] = vfmsq_f32(o[15], x[12], y[3]);
    o[15] = vfmaq_f32(o[15], x[13], y[2]);
    o[15] = vfmsq_f32(o[15], x[14], y[1]);
    o[15] = vfmaq_f32(o[15], x[15], y[0]);

    // Transpose outputs back to row-major
    for (int i = 0; i < 4; ++i) {
        const int offset = i * 4;
        float32x4_t a0 = o[offset + 0];
        float32x4_t a1 = o[offset + 1];
        float32x4_t a2 = o[offset + 2];
        float32x4_t a3 = o[offset + 3];

        float32x4_t q0 = vzip1q_f32(a0, a1);
        float32x4_t q2 = vzip2q_f32(a0, a1);
        float32x4_t q1 = vzip1q_f32(a2, a3);
        float32x4_t q3 = vzip2q_f32(a2, a3);

        vst1q_f32(o_row + offset, vreinterpretq_f32_f64(vzip1q_f64(vreinterpretq_f64_f32(q0), vreinterpretq_f64_f32(q1))));
        vst1q_f32(o_row + 16 + offset, vreinterpretq_f32_f64(vzip2q_f64(vreinterpretq_f64_f32(q0), vreinterpretq_f64_f32(q1))));
        vst1q_f32(o_row + 32 + offset, vreinterpretq_f32_f64(vzip1q_f64(vreinterpretq_f64_f32(q2), vreinterpretq_f64_f32(q3))));
        vst1q_f32(o_row + 48 + offset, vreinterpretq_f32_f64(vzip2q_f64(vreinterpretq_f64_f32(q2), vreinterpretq_f64_f32(q3))));
    }
}