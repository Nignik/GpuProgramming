#include <iostream>
#include <chrono>
#include <thread>

#include "matmul.cu"
#include "utils.h"

int main() {
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

    print_matrix<>(A * B);
}