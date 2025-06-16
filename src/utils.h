#pragma once

#include <iostream>

#include "matmul.cu"

template<int N, int M>
void print_matrix(const mat<N, M> mat) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            std::cout << mat[i][j] << ' ';
        }
        std::cout << '\n';
    }
}
