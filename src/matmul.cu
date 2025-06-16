#pragma once

#include <iostream>
#include <vector>
#include <ranges>

#define TILE_DIM 16

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(-1); \
    }

template<int N, int M, int P>
__global__ void matmul_kernel(const float* A, const float* B, float* C) {
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < P) {
        float dot = 0.f;
        for (int k = 0; k < N; k++) {
            dot += A[row * N + k] * B[k * P + col];
        }
        C[row * P + col] = dot;
    }
}

template<int N, int M, int P>
__global__ void tiled_matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C)
{
    const unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    float acc = 0;
    for (int i = 0; i < N; i += TILE_DIM)
    {
        int aCol = i + threadIdx.x;
        int bRow = i + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && aCol < N) ? A[row * N + aCol] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < N && col < P) ? B[bRow * P + col] : 0.f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; k++)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < P)
        C[row * P + col] = acc;
}

template<int M, int N>
class mat {
public:
    explicit mat(std::vector<std::vector<float>>&& matrix)
    {
        _mat.reserve(M * N);
        for (const auto& row : matrix) {
            _mat.insert(_mat.end(), row.begin(), row.end());
        }
    }
    mat() 
        : _mat(std::vector<float>(M * N))
    {}

    [[nodiscard]] const float* data() const { return _mat.data(); }
    [[nodiscard]] float* data() { return _mat.data(); }

    std::span<const float> operator[](const uint32_t i) const { return {&_mat[i * N], N}; }
    std::span<float> operator[](const uint32_t i) { return {&_mat[i * N], N}; }

    template<int P>
    mat<M, P> operator*(const mat<N, P>& other) {
        return gpu_matmul(*this, other);
    }

    void print() {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                std::cout << this[i][j] << ' ';
            }
            std::cout << '\n';
        }
    }

private:
    std::vector<float> _mat{};
};

template<int N, int M, int P>
mat<M, P> tiled_matmul(const mat<M, N>& A, const mat<N, P>& B) {
    constexpr int T = 8;
    mat<M, P> result;
    for (int i = 0; i < M; i+=T) {
        for (int j = 0; j < P; j+=T) {
            for (int k = 0; k < N; k+=T) {
                for (int ii = i; ii < std::min(i + T, M); ii++) {
                    for (int jj = j; jj < std::min(j + T, P); jj++) {
                        float dot = result[ii][jj];
                        for (int kk = k; kk < std::min(k + T, N); kk++) {
                            dot += A[ii][kk] * B[kk][jj];
                        }
                        result[ii][jj] = dot;
                    }
                }
            }
        }
    }
    return result;
}

template <int N, int M, int P>
mat<M, P> matmul(const mat<M, N>& A, const mat<N, P>& B) {
    mat<M, P> result;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < P; j++) {
            float dot = 0.f;
            for (int k = 0; k < N; k++) {
                dot += A[i][k] * B[k][j];
            }
            result[i][j] = dot;
        }
    }
    return result;
}

template<int N, int M, int P>
mat<M, P> gpu_matmul(const mat<M, N>& h_A, const mat<N, P>& h_B) {
    mat<M, P> h_C;

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A), M*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_B), N*P*sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_C), M*P*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M*N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), N*P*sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockDim{16, 16};
    dim3 gridDim((P + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    matmul_kernel<N, M, P><<<gridDim, blockDim>>>(d_A, d_B, d_C);

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, N*N*sizeof(float), cudaMemcpyDeviceToHost));

    return h_C;
}

template<int N, int M, int P>
mat<M, P> gpu_tiled_matmul(const mat<M, N>& h_A, const mat<N, P>& h_B) {
    mat<M, P> h_C;

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A), M*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_B), N*P*sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_C), M*P*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M*N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), N*P*sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockDim{TILE_DIM, TILE_DIM};
    dim3 gridDim((P + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    tiled_matmul_kernel<N, M, P><<<gridDim, blockDim>>>(d_A, d_B, d_C);

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, N*N*sizeof(float), cudaMemcpyDeviceToHost));

    return h_C;
}
