#include <stdio.h>

__global__ void matrixVectorMul(const float *matrix, const float *vector, float *result, int rows, int cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows)
    {
        float sum = 0.0f;
        for (int col = 0; col < cols; col++)
        {
            sum += matrix[row * cols + col] * vector[col];
        }
        result[row] = sum;
    }
}

extern "C" void launchMatrixVectorMul(const float *matrix, const float *vector, float *result, int rows, int cols)
{
    float *d_matrix, *d_vector, *d_result;
    size_t matrixSize = rows * cols * sizeof(float);
    size_t vectorSize = cols * sizeof(float);
    size_t resultSize = rows * sizeof(float);

    cudaMalloc((void **)&d_matrix, matrixSize);
    cudaMalloc((void **)&d_vector, vectorSize);
    cudaMalloc((void **)&d_result, resultSize);

    cudaMemcpy(d_matrix, matrix, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector, vector, vectorSize, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (rows + blockSize - 1) / blockSize;
    matrixVectorMul<<<numBlocks, blockSize>>>(d_matrix, d_vector, d_result, rows, cols);

    cudaMemcpy(result, d_result, resultSize, cudaMemcpyDeviceToHost);

    cudaFree(d_matrix);
    cudaFree(d_vector);
    cudaFree(d_result);
}
