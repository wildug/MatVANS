#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

#include <cuda_runtime_api.h>

#include "safetensor_loader.hpp"

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

extern __global__ void mat_vec_uncompressed_warp_per_row(
    int8_t *__restrict__ out_vec,
    const int4 *__restrict__ mat, // row-major [out_dim x in_dim]
    const int4 *__restrict__ in_vec,
    int in_dim, int out_dim, float scale);

extern __global__ void mat_vec_uncompressed_thread_per_row(
    int8_t *__restrict__ out_vec,
    const int4 *__restrict__ mat, // indexed by: [row_group, col_group, row, col]
    const int4 *__restrict__ in_vec,
    int in_dim, int out_dim, float scale);

void mat_vec_mul(
    const int4 *matrix_gpu,
    const int4 *in_vector_gpu,
    int4 *out_vector_gpu,
    size_t in_dim,
    size_t out_dim,
    float scale)
{
    check_CUDA_error("Before mat_vec_mul");

    int warpsPerBlock = 4; // TODO: tune
    dim3 grid((out_dim + warpsPerBlock - 1) / warpsPerBlock);
    dim3 block(warpsPerBlock * WARP_SIZE);
    mat_vec_uncompressed_warp_per_row<<<grid, block, in_dim * sizeof(int8_t)>>>(
        reinterpret_cast<int8_t *>(out_vector_gpu), matrix_gpu, in_vector_gpu, in_dim, out_dim, scale);

    check_CUDA_error("After mat_vec_mul");
}

void mat_vec_mul_coalesced(
    const int4 *matrix_gpu,
    const int4 *in_vector_gpu,
    int4 *out_vector_gpu,
    size_t in_dim,
    size_t out_dim,
    float scale)
{
    check_CUDA_error("Before mat_vec_mul_coalesced");

    int warpsPerBlock = 4; // TODO: why?
    dim3 grid((out_dim + (warpsPerBlock * WARP_SIZE) - 1) / (warpsPerBlock * WARP_SIZE));
    dim3 block(warpsPerBlock * WARP_SIZE);
    mat_vec_uncompressed_thread_per_row<<<grid, block, in_dim * sizeof(int8_t)>>>(
        reinterpret_cast<int8_t *>(out_vector_gpu), matrix_gpu, in_vector_gpu, in_dim, out_dim, scale);

    check_CUDA_error("After mat_vec_mul_coalesced");
}

uint32_t hash_int8_array(int8_t *arr, size_t length)
{
    uint32_t hash = 0;
    uint8_t *byte_arr = reinterpret_cast<uint8_t *>(arr);

    for (size_t i = 0; i < length; i++)
    {
        hash = (hash >> 27) | (hash << 5); // Rotate left by 5 bits
        hash = (hash ^ byte_arr[i]) * 0x27220A95;
    }

    return hash;
}

int main(int argc, char *argv[])
{
    const int NUM_ITERATIONS = 1000;

    // Load matrices from disk.
    assert(argc == 2);
    std::string filename = argv[1];
    auto file = SafetensorFile(filename);

    auto matrices = file.get_tensor("matrices");
    assert(matrices.tensor.dtype == safetensors::dtype::kINT8);

    auto shape = matrices.tensor.shape;
    assert(shape.size() == 3 || shape.size() == 5);
    bool coalesced = (shape.size() == 5);
    const size_t num_matrices = shape[0];
    size_t dim;

    if (coalesced)
    { // Coalesced (indices: [matrix, row_group, col_group, row, col])
        auto warp_size = shape[3];
        assert(warp_size == WARP_SIZE);
        dim = shape[1] * warp_size;
        assert(shape[4] == 16); // We read 16 bytes (128 bits) at a time.
        assert(shape[2] * shape[4] == dim);
    }
    else
    {
        // Uncoalesced (indices: [matrix, row, col])
        dim = shape[1];
        assert(shape[2] == dim);
    }
    assert(dim % sizeof(int4) == 0); // To ensure each row can be aligned to 128 bit

    auto initial_vector = file.get_tensor("initial_vector");
    assert(initial_vector.tensor.dtype == safetensors::dtype::kINT8);
    assert(initial_vector.tensor.shape.size() == 1);
    assert(initial_vector.tensor.shape[0] == dim);

    auto final_vector = new int4[dim / sizeof(int4)];

    auto scales = file.get_tensor("scales");
    assert(scales.tensor.dtype == safetensors::dtype::kFLOAT32);
    assert(scales.tensor.shape.size() == 1);
    assert(scales.tensor.shape[0] == num_matrices);
    const float *scales_data = reinterpret_cast<const float *>(scales.data);

    auto hashes = file.get_tensor("hashes");
    assert(hashes.tensor.dtype == safetensors::dtype::kUINT32);
    assert(hashes.tensor.shape.size() == 1);
    assert(hashes.tensor.shape[0] == num_matrices);
    const uint32_t *hashes_data = reinterpret_cast<const uint32_t *>(hashes.data);

    std::cout << "Number of matrices: " << num_matrices << std::endl;
    std::cout << "Matrix dimensions: " << dim << "x" << dim << std::endl;
    if (coalesced)
    {
        std::cout << "Coalesced with warp size: " << WARP_SIZE << std::endl;
    }
    else
    {
        std::cout << "Uncoalesced storage." << std::endl;
    }

    std::cout << "Initial vector: [";
    for (auto i = 0; i < std::min(dim, static_cast<size_t>(10)); i++)
    {
        std::cout << static_cast<int>(reinterpret_cast<const int8_t *>(initial_vector.data)[i]) << ", ";
    }
    if (dim > 10)
        std::cout << "...";
    std::cout << "]" << std::endl;

    std::cout << "Scales: [";
    for (auto i = 0; i < std::min(num_matrices, static_cast<size_t>(10)); i++)
    {
        std::cout << scales_data[i] << ", ";
    }
    if (num_matrices > 10)
        std::cout << "...";
    std::cout << "]" << std::endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Initialize cuBLAS handle
    // cuBLASLt handle

    __align__(16) int4 *in_vec_gpu;
    __align__(16) int4 *out_vec_gpu;
    __align__(16) int4 *matrices_gpu;
    __align__(16) int4 *initial_vector_gpu;
    cudaMalloc(&in_vec_gpu, dim);
    cudaMalloc(&out_vec_gpu, dim);
    cudaMalloc(&initial_vector_gpu, dim);
    cudaMemcpy(
        initial_vector_gpu, initial_vector.data,
        sizeof(int8_t) * dim, cudaMemcpyHostToDevice);
    cudaMalloc(&matrices_gpu, num_matrices * dim * dim);
    cudaMemcpy(
        matrices_gpu, matrices.data,
        sizeof(int8_t) * num_matrices * dim * dim, cudaMemcpyHostToDevice);
    check_CUDA_error("after copying matrices onto GPU");

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
            __align__(16) int4 *matrix_gpu = &matrices_gpu[k * dim * dim / sizeof(int4)];
            if (coalesced)
            {
                mat_vec_mul_coalesced(matrix_gpu, k == 0 ? initial_vector_gpu : in_vec_gpu, out_vec_gpu, dim, dim, scales_data[k]);
            }
            else
            {
                mat_vec_mul(matrix_gpu, k == 0 ? initial_vector_gpu : in_vec_gpu, out_vec_gpu, dim, dim, scales_data[k]);
            }

            std::swap(in_vec_gpu, out_vec_gpu);
        }
        check_CUDA_error("after loop");

        cudaMemcpy(final_vector, in_vec_gpu, sizeof(int8_t) * dim, cudaMemcpyDeviceToHost);
        check_CUDA_error("after downloading final vector");
        dummy_checksum ^= reinterpret_cast<int8_t *>(final_vector)[iteration % dim]; // Prevent compiler from optimizing away the computation
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0.0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float throughput = 1000.0 * num_matrices * dim * dim * NUM_ITERATIONS / milliseconds;
    std::cout << "Average time (without memcpy of matrices): " << (milliseconds / NUM_ITERATIONS) << " ms." << std::endl
              << "Throughput: " << std::scientific << throughput << " multiply-and-accumulates per second" << std::endl
              << "(dummy checksum: " << static_cast<int>(dummy_checksum) << ")" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(matrices_gpu);
    cudaFree(initial_vector_gpu);
    cudaFree(in_vec_gpu);
    cudaFree(out_vec_gpu);
    check_CUDA_error("after freeing the data");

    // Output result
    std::cout << "result = [";
    for (auto i = 0; i < std::min(dim, static_cast<size_t>(10)); i++)
    {
        // printf("Result at index %d: %d\n", i, h_result[i]);
        std::cout << static_cast<int>(reinterpret_cast<int8_t *>(final_vector)[i]) << ", ";
    }
    if (dim > 10)
        std::cout << "...";
    std::cout << "]" << std::endl;

    const auto hash = hash_int8_array(reinterpret_cast<int8_t *>(final_vector), dim);
    std::cout << "Hash of result: " << std::hex << hash << std::dec << std::endl;
    assert(hash == hashes_data[num_matrices - 1]);
    std::cout << "Hash matches expected value." << std::endl;

    delete[] final_vector;

    cudaDeviceReset();
    check_CUDA_error("End of program.");

    return 0;
}
