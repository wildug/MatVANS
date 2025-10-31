#include <iostream>
#include <cuda_runtime.h>

extern __global__ void matrixVectorMul(const float *matrix, const float *vector, float *result, int rows, int cols);

int main()
{
    const int rows = 4;
    const int cols = 4;
    float h_matrix[rows][cols] = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16}};
    float h_vector[cols] = {1, 1, 1, 1};
    float h_result[rows];

    float *d_matrix, *d_vector, *d_result;

    cudaMalloc((void **)&d_matrix, rows * cols * sizeof(float));
    cudaMalloc((void **)&d_vector, cols * sizeof(float));
    cudaMalloc((void **)&d_result, rows * sizeof(float));

    cudaMemcpy(d_matrix, h_matrix, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, h_vector, cols * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 2;
    int numBlocks = (rows + blockSize - 1) / blockSize;
    matrixVectorMul<<<numBlocks, blockSize>>>(d_matrix, d_vector, d_result, rows, cols);

    cudaMemcpy(h_result, d_result, rows * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < rows; i++)
    {
        std::cout << "Result[" << i << "] = " << h_result[i] << std::endl;
    }

    cudaFree(d_matrix);
    cudaFree(d_vector);
    cudaFree(d_result);

    return 0;
}
