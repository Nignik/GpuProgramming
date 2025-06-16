#include <gtest/gtest.h>
#include <iostream>

#include "../src/matmul.cu"

template<int N, int M>
void MAT_EQ(mat<M, N> A, mat<M, N> B) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            EXPECT_EQ(A[i][j], B[i][j]);
        }
    }
}

mat<4, 4> A{{{ 2,  1,  3,  0},
                 { 4, -1,  2,  2},
                 { 0,  5, -2,  1},
                 { 3,  0,  0,  1}}};

mat<4, 4> B{{{ 1,  2,  0, -1},
             { 0,  1,  4,  2},
             { 3,  0,  1,  5},
             {-2,  1,  0,  3}}};

mat<4, 4> C{{{11,  5,  7, 15},
             { 6,  9, -2, 10},
             {-8,  6, 18,  3},
             { 1,  7,  0,  0}}};

TEST(MathTests, CpuMatmul) {
    MAT_EQ(matmul(A, B), C);
}

TEST(MathTests, TiledCpuMatmul) {
    MAT_EQ(tiled_matmul(A, B), C);
}

TEST(MathTests, GpuMatmul) {
    MAT_EQ(gpu_matmul(A, B), C);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}