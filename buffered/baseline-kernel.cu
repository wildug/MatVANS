#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

__global__ void mat_vec_uncompressed_warp_per_row(
    int8_t *__restrict__ out_vec,
    const int4 *__restrict__ mat, // row-major [out_dim x in_dim]
    const int4 *__restrict__ in_vec,
    int in_dim, int out_dim, float scale)
{
    // Warp bookkeeping
    const int tid = threadIdx.x;
    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;

    // One warp == one row
    const int warpsPerBlock = blockDim.x / WARP_SIZE;
    const int row = blockIdx.x * warpsPerBlock + warp;
    if (row >= out_dim)
        return;

    // ---- Load vector into shared (int32-packed) once per block ----
    extern __shared__ __align__(16) int4 in_vec_shared[];

    const int nChunks128 = in_dim / sizeof(int4);
    for (int i = tid; i < nChunks128; i += blockDim.x)
    {
        in_vec_shared[i] = in_vec[i];
    }
    __syncthreads();

    const __align__(16) int4 *__restrict__ mat_row = reinterpret_cast<const int4 *>(mat + row * in_dim / sizeof(int4));

    // TODO: test if this is faster than the loop below
    // const __align__(16) int4 *__restrict__ vec_end = in_vec_shared + in_dim / sizeof(int4);
    // int acc = 0;
    // const __align__(16) int4 *__restrict__ mat_pointer = mat_row + lane;
    // const __align__(16) int4 *__restrict__ vec_pointer = in_vec_shared + lane;
    // while (vec_pointer < vec_end)

    // Each lane walks over int4 chunks with stride = warpSize
    int acc = 0;
    for (int j = lane; j < nChunks128; j += WARP_SIZE)
    {
        // load 16B (4x int32 packed int8) from row and vector
        const int4 a4 = *(reinterpret_cast<const int4 *>(mat_row) + j);
        const int4 b4 = *(reinterpret_cast<const int4 *>(in_vec_shared) + j);

        // 4 dp4a per 128-bit chunk
        acc = __dp4a(a4.x, b4.x, acc);
        acc = __dp4a(a4.y, b4.y, acc);
        acc = __dp4a(a4.z, b4.z, acc);
        acc = __dp4a(a4.w, b4.w, acc);
    }

// Warp reduction (sum partials across lanes)
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    // Lane 0 writes the row result
    if (lane == 0)
        out_vec[row] = static_cast<int8_t>(__float2int_rn(static_cast<float>(acc) * scale));
}

__global__ void mat_vec_uncompressed_thread_per_row(
    int8_t *__restrict__ out_vec,
    const int4 *__restrict__ mat, // indexed by: [row_group, col_group, row, col]
    const int4 *__restrict__ in_vec,
    int in_dim, int out_dim, float scale)
{
    // Warp bookkeeping
    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int row = blockIdx.x * blockDim.x + tid;

    // One thread == one row
    if (row >= out_dim)
        return;

    // ---- Load vector into shared (int32-packed) once per block ----
    extern __shared__ __align__(16) int4 in_vec_shared[];

    const int nChunks128 = in_dim / sizeof(int4);
    for (int i = tid; i < nChunks128; i += blockDim.x)
    {
        in_vec_shared[i] = in_vec[i];
    }
    __syncthreads();

    // const __align__(16) int4 *__restrict__ row_group_start = mat + (row - lane) * in_dim / sizeof(int4);
    const __align__(16) int4 *__restrict__ row_group_start = mat + (row / WARP_SIZE) * WARP_SIZE * in_dim / sizeof(int4);
    const __align__(16) int4 *__restrict__ row_ptr = row_group_start + lane;

    // Each thread walks over int4 chunks with stride = warpSize
    int acc = 0;
    for (int j = 0; j < nChunks128; j += 1)
    {
        // load 16B (4x int32 packed int8) from row and vector
        const int4 a4 = *row_ptr;
        const int4 b4 = in_vec_shared[j];

        // 4 dp4a per 128-bit chunk
        acc = __dp4a(a4.x, b4.x, acc);
        acc = __dp4a(a4.y, b4.y, acc);
        acc = __dp4a(a4.z, b4.z, acc);
        acc = __dp4a(a4.w, b4.w, acc);

        row_ptr += WARP_SIZE;
    }

    out_vec[row] = static_cast<int8_t>(__float2int_rn(static_cast<float>(acc) * scale));
}
