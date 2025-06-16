#include <random>
#include <benchmark/benchmark.h>

#include "../src/matmul.cu"

constexpr int R = 100, C = 100;

mat<R, C> A;
mat<R, C> B;

template<int Rows, int Cols>
mat<Rows, Cols> random_matrix(float min, float max)
{
    std::mt19937                      rng{ std::random_device{}() };
    std::uniform_real_distribution<float>  dist(min, max);

    mat<Rows, Cols> m{};
    for (int i = 0; i < Rows; i++)
        for (int j = 0; j < Cols; j++)
            m[i][j] = dist(rng);

    return m;
}

static void BM_CpuMatmul(benchmark::State& state) {
    for (auto _ : state)
        benchmark::DoNotOptimize(matmul(A, B));
}

static void BM_CpuTiledMatmul(benchmark::State& state) {
    for (auto _ : state)
        benchmark::DoNotOptimize(tiled_matmul(A, B));
}
static void BM_GpuMatmul(benchmark::State& state) {
    for (auto _ : state)
        benchmark::DoNotOptimize(gpu_matmul(A, B));
}

BENCHMARK(BM_CpuMatmul);
BENCHMARK(BM_CpuTiledMatmul);
BENCHMARK(BM_GpuMatmul);

int main(int argc, char** argv)
{

    A = random_matrix<R, C>(-1000.f, 1000.f);
    B = random_matrix<R, C>(-1000.f, 1000.f);

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}