#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>

#include <cuda_runtime_api.h>

#include "sane_safetensors.hpp"

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

void check_CUDA_error(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(-1);
    }
}

extern __global__ void mat_vec_ans_warp_per_row(
    uint4 *__restrict__ out_vec,
    const uint4 *__restrict__ compressed,
    const uint32_t *__restrict__ warp_offsets,
    const uint32_t *__restrict__ ppf,
    const uint32_t *__restrict__ ppf_cum,
    const uint32_t *__restrict__ ppf_prob,
    const uint4 *__restrict__ in_vec,
    int in_dim,
    int out_dim,
    float scale);

void mat_vec_mul_coalesced(
    const uint4 *in_vector_gpu,
    uint4 *out_vector_gpu,
    size_t in_dim,
    size_t out_dim,
    float scale,
    const uint4 *__restrict__ compressed,
    const uint32_t *__restrict__ warp_offsets,
    const uint32_t *__restrict__ ppf,
    const uint32_t *__restrict__ ppf_cum,
    const uint32_t *__restrict__ ppf_prob)
{
    check_CUDA_error("Before mat_vec_mul");

    int warpsPerBlock = 1; // TODO: tune (use `nvcc --ptxas-options=-v` to see register usage)
    dim3 grid((out_dim + warpsPerBlock - 1) / warpsPerBlock);
    dim3 block(warpsPerBlock * WARP_SIZE);
    mat_vec_ans_warp_per_row<<<grid, block, in_dim * sizeof(int8_t)>>>(
        out_vector_gpu, compressed, warp_offsets,
        ppf, ppf_cum, ppf_prob, in_vector_gpu, in_dim, out_dim, scale);

    check_CUDA_error("After mat_vec_mul");
}

uint32_t hash_int8_array(std::span<const int8_t> arr)
{
    uint32_t hash = 0;

    for (size_t i = 0; i < arr.size(); i++)
    {
        hash = (hash >> 27) | (hash << 5); // Rotate left by 5 bits
        hash = (hash ^ *reinterpret_cast<const uint8_t *>(&arr[i])) * 0x27220A95;
    }

    return hash;
}

void print_data_summary(std::span<const uint32_t> compressed, std::span<const int8_t> initial_vector, std::span<const float> scales)
{
    auto megabytes = static_cast<float>(compressed.size() * sizeof(uint32_t)) / (1024.0 * 1024.0);
    std::cout << "Read " << scales.size() << " matrices of dimensions "
              << initial_vector.size() << "x" << initial_vector.size() << " each; "
              << "total compressed size: " << std::fixed << std::setprecision(2) << megabytes << " MiB."
              << std::endl;

    std::cout << "Initial vector: [";
    for (auto i = 0; i < std::min(initial_vector.size(), static_cast<size_t>(10)); i++)
    {
        std::cout << static_cast<int>(initial_vector[i]) << ", ";
    }
    if (initial_vector.size() > 10)
        std::cout << "...";
    std::cout << "]" << std::endl;

    std::cout << "Scales: [" << std::fixed << std::setprecision(4);
    for (auto i = 0; i < std::min(scales.size(), size_t{10}); i++)
    {
        std::cout << scales[i] << ", ";
    }
    if (scales.size() > 10)
        std::cout << "...";
    std::cout << "]" << std::endl;
}

void print_result_summary(std::span<const int8_t> result, uint32_t expected_hash)
{
    std::cout << "result = [";
    for (auto i = 0; i < std::min(result.size(), static_cast<size_t>(10)); i++)
    {
        std::cout << static_cast<int>(result[i]) << ", ";
    }
    if (result.size() > 10)
        std::cout << "...";
    std::cout << "]" << std::endl;

    const auto hash = hash_int8_array(result);
    std::cout << "Hash of result: " << std::hex << hash
              << " (expected: " << expected_hash << ")" << std::dec << std::endl;
    assert(hash == expected_hash);
    std::cout << "Hash matches expected value." << std::endl;
}

template <typename Dtype>
__align__(16) Dtype *upload_to_gpu(std::span<const Dtype> data)
{
    auto num_bytes = sizeof(Dtype) * data.size();
    assert(num_bytes % sizeof(uint4) == 0); // Ensure alignment for uint4 for all data for simplicity

    __align__(16) Dtype *data_gpu;
    cudaMalloc(&data_gpu, num_bytes);
    check_CUDA_error("after allocating memory on GPU");
    cudaMemcpy(data_gpu, data.data(), num_bytes, cudaMemcpyHostToDevice);
    check_CUDA_error("after copying data onto GPU");

    return data_gpu;
}

template <typename Dtype>
__align__(16) Dtype *gpu_malloc_like(std::span<const Dtype> data)
{
    auto num_bytes = sizeof(Dtype) * data.size();
    assert(num_bytes % sizeof(uint4) == 0); // Ensure alignment for uint4 for all data for simplicity.

    __align__(16) Dtype *data_gpu;
    cudaMalloc(&data_gpu, num_bytes);
    check_CUDA_error("after allocating empty memory on GPU");

    return data_gpu;
}

int main(int argc, char *argv[])
{
    const int NUM_ITERATIONS = 1000;

    // Load matrices from disk.
    assert(argc == 2);
    std::string filename = argv[1];

    // Load data from the safetensors file.
    // Start with the tensors that expose the matrix dimensions and number of matrices most directly.
    auto file = SaneSafetensorFile(filename);
    auto [scales, scales_shape] = file.get_tensor<float, 1>("scales");
    size_t num_matrices = scales_shape[0];

    auto [initial_vector, in_vec_shape] = file.get_tensor<int8_t, 1>("initial_vector");
    size_t dim = in_vec_shape[0];
    assert(dim % (4 * WARP_SIZE) == 0);

    auto [compressed, compressed_shape] = file.get_tensor<uint32_t, 1>("compressed");
    size_t total_compressed_size = compressed_shape[0];

    // Then load the tensors whose shapes we can now verify.
    auto hashes = file.get_data<uint32_t, 1>("hashes", std::move(std::array{num_matrices}));
    auto all_warp_offsets = file.get_data<uint32_t, 2>("all_warp_offsets", std::move(std::array{num_matrices, dim}));
    auto all_ppfs = file.get_data<uint32_t, 2>("all_ppfs", std::move(std::array<size_t, 2>{num_matrices, WARP_SIZE}));
    auto all_ppfs_cum = file.get_data<uint32_t, 2>("all_ppfs_cum", std::move(std::array<size_t, 2>{num_matrices, WARP_SIZE}));
    auto all_ppfs_prob = file.get_data<uint32_t, 2>("all_ppfs_prob", std::move(std::array<size_t, 2>{num_matrices, WARP_SIZE}));

    print_data_summary(compressed, initial_vector, scales);

    auto final_vector = std::vector<int8_t>(dim);

    // Copy data onto the GPU.
    auto all_warp_offsets_gpu = reinterpret_cast<uint32_t *>(upload_to_gpu(all_warp_offsets));
    auto all_ppfs_gpu = upload_to_gpu(all_ppfs);
    auto all_ppfs_cum_gpu = upload_to_gpu(all_ppfs_cum);
    auto all_ppfs_prob_gpu = upload_to_gpu(all_ppfs_prob);
    auto compressed_gpu = reinterpret_cast<uint4 *>(upload_to_gpu(compressed));
    auto initial_vector_gpu = reinterpret_cast<uint4 *>(upload_to_gpu(initial_vector));

    // Allocate memory for input and output of each matrix multiplication.
    auto in_vec_gpu = reinterpret_cast<uint4 *>(gpu_malloc_like(initial_vector));
    auto out_vec_gpu = reinterpret_cast<uint4 *>(gpu_malloc_like(initial_vector));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    auto num_warps_per_matrix = dim; // Assuming one warp per matrix row

    // outer loop for benchmarking
    int8_t dummy_checksum = 0;
    for (int iteration = 0; iteration != NUM_ITERATIONS + 1; iteration++)
    {
        // Use 0th iteration for warmup
        if (iteration == 1)
        {
            cudaEventRecord(start);
        }

        for (int k = 0; k < num_matrices; k++)
        {
            __align__(16) const uint32_t *warp_offsets_gpu = all_warp_offsets_gpu + k * num_warps_per_matrix;
            __align__(16) const uint32_t *ppf_gpu = all_ppfs_gpu + k * WARP_SIZE;
            __align__(16) const uint32_t *ppf_cum_gpu = all_ppfs_cum_gpu + k * WARP_SIZE;
            __align__(16) const uint32_t *ppf_prob_gpu = all_ppfs_prob_gpu + k * WARP_SIZE;

            mat_vec_mul_coalesced(
                k == 0 ? initial_vector_gpu : in_vec_gpu,
                out_vec_gpu, dim, dim, scales[k], compressed_gpu,
                warp_offsets_gpu, ppf_gpu, ppf_cum_gpu, ppf_prob_gpu);

            std::swap(in_vec_gpu, out_vec_gpu);
        }
        check_CUDA_error("after loop");

        cudaMemcpy(final_vector.data(), in_vec_gpu, sizeof(int8_t) * dim, cudaMemcpyDeviceToHost);
        check_CUDA_error("after downloading final vector");
        dummy_checksum ^= final_vector[iteration % final_vector.size()]; // Prevent compiler from optimizing away the computation
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0.0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float throughput = 1000.0 * num_matrices * dim * dim * NUM_ITERATIONS / milliseconds;
    std::cout << "Average time (without memcpy of matrices): " << (milliseconds / NUM_ITERATIONS) << " ms." << std::endl
              << "Throughput: " << std::scientific << throughput << " multiply-and-accumulates per second" << std::endl
              << "(dummy checksum: " << static_cast<int>(dummy_checksum) << ")" << std::endl;

    print_result_summary(final_vector, hashes[num_matrices - 1]);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(initial_vector_gpu);
    cudaFree(in_vec_gpu);
    cudaFree(out_vec_gpu);
    cudaFree(compressed_gpu);
    cudaFree(all_ppfs_gpu);
    cudaFree(all_ppfs_cum_gpu);
    cudaFree(all_ppfs_prob_gpu);
    check_CUDA_error("after freeing the data");

    cudaDeviceReset();
    check_CUDA_error("End of program.");

    return 0;
}
