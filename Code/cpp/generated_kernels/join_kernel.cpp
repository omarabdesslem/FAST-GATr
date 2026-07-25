static inline void join_kernel_one_batch(
    const float* __restrict x_row,
    const float* __restrict y_row,
    float*       __restrict o_row
) {
    const float x0 = x_row[0];
    const float x1 = x_row[1];
    const float x2 = x_row[2];
    const float x3 = x_row[3];
    const float x4 = x_row[4];
    const float x5 = x_row[5];
    const float x6 = x_row[6];
    const float x7 = x_row[7];
    const float x8 = x_row[8];
    const float x9 = x_row[9];
    const float x10 = x_row[10];
    const float x11 = x_row[11];
    const float x12 = x_row[12];
    const float x13 = x_row[13];
    const float x14 = x_row[14];
    const float x15 = x_row[15];
    const float y0 = y_row[0];
    const float y1 = y_row[1];
    const float y2 = y_row[2];
    const float y3 = y_row[3];
    const float y4 = y_row[4];
    const float y5 = y_row[5];
    const float y6 = y_row[6];
    const float y7 = y_row[7];
    const float y8 = y_row[8];
    const float y9 = y_row[9];
    const float y10 = y_row[10];
    const float y11 = y_row[11];
    const float y12 = y_row[12];
    const float y13 = y_row[13];
    const float y14 = y_row[14];
    const float y15 = y_row[15];

    float a0 = 0.0f;
    float a1 = 0.0f;
    float a2 = 0.0f;
    float a3 = 0.0f;
    float a4 = 0.0f;
    float a5 = 0.0f;
    float a6 = 0.0f;
    float a7 = 0.0f;
    float a8 = 0.0f;
    float a9 = 0.0f;
    float a10 = 0.0f;
    float a11 = 0.0f;
    float a12 = 0.0f;
    float a13 = 0.0f;
    float a14 = 0.0f;
    float a15 = 0.0f;

    a0 += x0 * y15;
    a0 += x1 * y14;
    a0 -= x2 * y13;
    a0 += x3 * y12;
    a0 -= x4 * y11;
    a0 += x5 * y10;
    a0 -= x6 * y9;
    a0 += x7 * y8;
    a0 += x8 * y7;
    a0 -= x9 * y6;
    a0 += x10 * y5;
    a0 += x11 * y4;
    a0 -= x12 * y3;
    a0 += x13 * y2;
    a0 -= x14 * y1;
    a0 += x15 * y0;

    a1 -= x1 * y15;
    a1 -= x5 * y13;
    a1 += x6 * y12;
    a1 -= x7 * y11;
    a1 -= x11 * y7;
    a1 += x12 * y6;
    a1 -= x13 * y5;
    a1 -= x15 * y1;

    a2 -= x2 * y15;
    a2 -= x5 * y14;
    a2 += x8 * y12;
    a2 -= x9 * y11;
    a2 -= x11 * y9;
    a2 += x12 * y8;
    a2 -= x14 * y5;
    a2 -= x15 * y2;

    a3 -= x3 * y15;
    a3 -= x6 * y14;
    a3 += x8 * y13;
    a3 -= x10 * y11;
    a3 -= x11 * y10;
    a3 += x13 * y8;
    a3 -= x14 * y6;
    a3 -= x15 * y3;

    a4 -= x4 * y15;
    a4 -= x7 * y14;
    a4 += x9 * y13;
    a4 -= x10 * y12;
    a4 -= x12 * y10;
    a4 += x13 * y9;
    a4 -= x14 * y7;
    a4 -= x15 * y4;

    a5 += x5 * y15;
    a5 += x11 * y12;
    a5 -= x12 * y11;
    a5 += x15 * y5;

    a6 += x6 * y15;
    a6 += x11 * y13;
    a6 -= x13 * y11;
    a6 += x15 * y6;

    a7 += x7 * y15;
    a7 += x12 * y13;
    a7 -= x13 * y12;
    a7 += x15 * y7;

    a8 += x8 * y15;
    a8 += x11 * y14;
    a8 -= x14 * y11;
    a8 += x15 * y8;

    a9 += x9 * y15;
    a9 += x12 * y14;
    a9 -= x14 * y12;
    a9 += x15 * y9;

    a10 += x10 * y15;
    a10 += x13 * y14;
    a10 -= x14 * y13;
    a10 += x15 * y10;

    a11 -= x11 * y15;
    a11 -= x15 * y11;

    a12 -= x12 * y15;
    a12 -= x15 * y12;

    a13 -= x13 * y15;
    a13 -= x15 * y13;

    a14 -= x14 * y15;
    a14 -= x15 * y14;

    a15 += x15 * y15;

    o_row[0] = a0;
    o_row[1] = a1;
    o_row[2] = a2;
    o_row[3] = a3;
    o_row[4] = a4;
    o_row[5] = a5;
    o_row[6] = a6;
    o_row[7] = a7;
    o_row[8] = a8;
    o_row[9] = a9;
    o_row[10] = a10;
    o_row[11] = a11;
    o_row[12] = a12;
    o_row[13] = a13;
    o_row[14] = a14;
    o_row[15] = a15;
}
