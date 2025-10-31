#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#include <cstdint>

/**
 * @brief Hints to the GPU to prefetch a cache line from global memory to L2 cache.
 *
 * This is a poor-man's version of asynchronous memory copy, which is not available on
 * all NVIDIA GPU architectures (e.g., not on Ampere).
 *
 * Apparently, it's not an issue if `ptr` points to out-of-bounds data as long as
 * prefetched data that later turns out to be out of bounds is never actually accessed.
 */
__device__ __forceinline__ void prefetch_global(const void *ptr)
{
    asm volatile("prefetch.global.L2 [%0];" ::"l"(ptr));
}

__global__ void mat_vec_ans_thread_per_row(
    uint4 *__restrict__ out_vec,
    const uint4 *__restrict__ compressed,
    const uint32_t *__restrict__ warp_offsets,
    const uint32_t *__restrict__ ppf,
    const uint32_t *__restrict__ ppf_cum,
    const uint32_t *__restrict__ ppf_prob,
    const uint4 *__restrict__ in_vec,
    int in_dim,
    int out_dim,
    float scale)
{
    // Warp bookkeeping
    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int row = blockIdx.x * blockDim.x + tid;
    const int global_warp_id = row / WARP_SIZE;
    const __align__(16) uint4 *__restrict__ cursor = compressed + warp_offsets[global_warp_id];

    const uint32_t lane_ppf = ppf[lane];
    const uint32_t lane_ppf_cum = ppf_cum[lane];
    const uint32_t lane_ppf_prob = ppf_prob[lane];

    // ---- Load vector into shared (int32-packed) once per block ----
    extern __shared__ __align__(16) uint4 in_vec_shared[];

    const int nChunks128 = in_dim / sizeof(int4);
    for (int i = tid; i < nChunks128; i += blockDim.x)
    {
        in_vec_shared[i] = in_vec[i];
    }
    __syncthreads();
    const char4 *in_vec_shared_char4 = reinterpret_cast<const char4 *>(in_vec_shared);

    // One thread == one row
    if (row >= out_dim)
        return;

    // Use the following registers to cache compressed data for this warp:
    // - `state`: this thread's 2-word (64-bit) ANS coder state
    // - `warp_cache`: one 4-word (128 bit) register per thread in the warp, which together make up
    //   a warp-wide variably sized array that holds between 0 and 4 * WARP_SIZE words. Initialized
    //   full.
    //
    // Decoding then reads data from the following cache hierarchy:
    // 1. For all threads in the warp whose 2-word sized `state` is less than half full, try to
    //    refill their `state`s from the `warp_cache` using `__ballot_sync` and `__shfl_sync`. If
    //    the `warp_cache` doesn't have enough data to refill all necessary `state`s in the warp,
    //    then:
    //    - refill as many of them as possible (thus emptying the `warp_cache`),
    //    - refill the entire `warp_cache` from global memory, and
    //    - refill any remaining `state`s from the `warp_cache`. This can never underflow.
    // 2. Decode 4 symbols at a time on each warp from `state`.
    // 3. Repeat from step 1 until the warp stripe is fully decoded.

    // Read the initial coder state and the first two words of the `warp_cache` in a single memory access:
    uint4 warp_cache = cursor[lane];
    uint64_t state = *reinterpret_cast<uint64_t *>(&warp_cache.z); // state = (warp_cache.w << 32) | warp_cache.z
    cursor += WARP_SIZE;
    // We've effectively already read `2 * WARP_SIZE` words from the warp cache (and interpreted
    // them as the initial coder states), so initialize `warp_cache_cursor` accordingly:
    unsigned int warp_cache_cursor = 2 * WARP_SIZE; // always in {0, 1, ..., 4 * WARP_SIZE}

    int acc = 0;
    const int in_dim_div4 = in_dim / 4;
    const unsigned int lower_lanes_mask = (1U << lane) - 1;

    for (int j = 0; j < in_dim_div4; j += 1)
    {
        auto this_lane_needs_refill = (state >> 32) == 0;
        unsigned int refilling_lanes = __ballot_sync(0xFFFFFFFF, this_lane_needs_refill);

        if (refilling_lanes != 0) // This branching never diverges across the warp.
        {
            // Get this lane's prefix sum, i.e., the number of lanes with a lower ID that need a refill.
            const int offset = __popc(refilling_lanes & lower_lanes_mask);

            // If we interpret `warp_cache` as a linear array of words, then we now have to read
            // `warp_cache[warp_cache_cursor .. warp_cache_cursor + num_refilling_lanes - 1]`.
            // We map this to the 4-word `warp_cache` per thread as follows: we first read from all
            // `warp_cache.x` fields of all threads in the warp, then from all `warp_cache.y` fields,
            // then `warp_cache.z`, and finally `warp_cache.w`. Thus, the read can cross at most one
            // `WARP_SIZE` boundary, and the part before that boundary is always available.

            // Read the values from the first part (returns some garbage data on lanes that don't
            // need a refill or where `warp_cache_cursor + offset >= WARP_SIZE`, but this data will
            // never be used):
            auto warp_cache_lane_cursor = (warp_cache_cursor + offset) % WARP_SIZE;
            uint32_t refill_value = __shfl_sync(
                0xFFFFFFFF,
                // reinterpret_cast<uint32_t *>(&warp_cache)[warp_cache_generation],
                warp_cache.x,
                warp_cache_lane_cursor);

            const int num_refilling_lanes = __popc(refilling_lanes);

            if ((warp_cache_cursor + num_refilling_lanes) / WARP_SIZE != warp_cache_cursor / WARP_SIZE) // This branching never diverges across the warp.
            {
                // There is a second part to read from.
                if (warp_cache_cursor / WARP_SIZE == 3)
                {
                    // We need to refill `warp_cache` from global memory
                    warp_cache = cursor[lane];
                    cursor += WARP_SIZE;
                    prefetch_global(&cursor[lane]);
                }
                else
                {
                    warp_cache.x = warp_cache.y;
                    warp_cache.y = warp_cache.z;
                    warp_cache.z = warp_cache.w;
                    // `warp_cache.w` won't be used before it's overwritten for the next time.
                }

                uint32_t refill_value_part2 = __shfl_sync(
                    0xFFFFFFFF,
                    // reinterpret_cast<uint32_t *>(&warp_cache)[warp_cache_generation],
                    warp_cache.x,
                    warp_cache_lane_cursor);

                if ((warp_cache_cursor + offset) / WARP_SIZE != warp_cache_cursor / WARP_SIZE) // This branch diverges.
                {
                    refill_value = refill_value_part2;
                }
            }

            // Merge the two parts
            if (this_lane_needs_refill) // This branch diverges.
            {
                state = (state << 32) | refill_value;
            }

            warp_cache_cursor = (warp_cache_cursor + num_refilling_lanes) % (4 * WARP_SIZE);
        }

        // Decode 4 symbols from `state` and pack them into `a`. Keep all lookup tables in registers to avoid cross-warp communication.
        char4 a;
        // TODO: maybe read `in_vec_shared` in chunks of `int4` at a time to reduce memory lookups.
        // (alternatively: use one warp per matrix row)
        const char4 b = in_vec_shared_char4[j];

        const auto quantile0 = state & 0x7F;
        const auto quantile0_lo = state & 0x03;
        const auto quantile0_hi = (state >> 2) & 0x1F;
        const auto symbols0 = __shfl_sync(0xFFFFFFFF, lane_ppf, quantile0_hi);
        a.x = static_cast<int8_t>(symbols0 >> (8 * quantile0_lo)); // Note: NVIDIA GPUs are little-endian.
        const auto left_cums0 = __shfl_sync(0xFFFFFFFF, lane_ppf_cum, quantile0_hi);
        const auto left_cum0 = static_cast<uint8_t>(left_cums0 >> (8 * quantile0_lo));
        const uint32_t remainder0 = quantile0 - left_cum0;
        const auto probs0 = __shfl_sync(0xFFFFFFFF, lane_ppf_prob, quantile0_hi);
        const auto prob0 = static_cast<uint8_t>(probs0 >> (8 * quantile0_lo));

        const auto quantile1 = (state >> 7) & 0x7F;
        const auto quantile1_lo = (state >> 7) & 0x03;
        const auto quantile1_hi = (state >> 9) & 0x1F;
        const auto symbols1 = __shfl_sync(0xFFFFFFFF, lane_ppf, quantile1_hi);
        a.y = static_cast<int8_t>(symbols1 >> (8 * quantile1_lo)); // Note: NVIDIA GPUs are little-endian.
        const auto left_cums1 = __shfl_sync(0xFFFFFFFF, lane_ppf_cum, quantile1_hi);
        const auto left_cum1 = static_cast<uint8_t>(left_cums1 >> (8 * quantile1_lo));
        const uint32_t remainder1 = quantile1 - left_cum1;
        const auto probs1 = __shfl_sync(0xFFFFFFFF, lane_ppf_prob, quantile1_hi);
        const auto prob1 = static_cast<uint8_t>(probs1 >> (8 * quantile1_lo));

        const auto quantile2 = (state >> 14) & 0x7F;
        const auto quantile2_lo = (state >> 14) & 0x03;
        const auto quantile2_hi = (state >> 16) & 0x1F;
        const auto symbols2 = __shfl_sync(0xFFFFFFFF, lane_ppf, quantile2_hi);
        a.z = static_cast<int8_t>(symbols2 >> (8 * quantile2_lo)); // Note: NVIDIA GPUs are little-endian.
        const auto left_cums2 = __shfl_sync(0xFFFFFFFF, lane_ppf_cum, quantile2_hi);
        const auto left_cum2 = static_cast<uint8_t>(left_cums2 >> (8 * quantile2_lo));
        const uint32_t remainder2 = quantile2 - left_cum2;
        const auto probs2 = __shfl_sync(0xFFFFFFFF, lane_ppf_prob, quantile2_hi);
        const auto prob2 = static_cast<uint8_t>(probs2 >> (8 * quantile2_lo));

        const auto quantile3 = (state >> 21) & 0x7F;
        const auto quantile3_lo = (state >> 21) & 0x03;
        const auto quantile3_hi = (state >> 23) & 0x1F;
        const auto symbols3 = __shfl_sync(0xFFFFFFFF, lane_ppf, quantile3_hi);
        a.w = static_cast<int8_t>(symbols3 >> (8 * quantile3_lo)); // Note: NVIDIA GPUs are little-endian.
        const auto left_cums3 = __shfl_sync(0xFFFFFFFF, lane_ppf_cum, quantile3_hi);
        const auto left_cum3 = static_cast<uint8_t>(left_cums3 >> (8 * quantile3_lo));
        const uint32_t remainder3 = quantile3 - left_cum3;
        const auto probs3 = __shfl_sync(0xFFFFFFFF, lane_ppf_prob, quantile3_hi);
        const auto prob3 = static_cast<uint8_t>(probs3 >> (8 * quantile3_lo));

        acc = __dp4a(a, b, acc);

        // Update `state`: we logically do the following operation:
        //   state = ((((state >> 28) * prob3 + remainder3) * prob2 + remainder2) * prob1 + remainder1) * prob0 + remainder0;
        // But naive evaluation as expressed above would require a lot of 64-bit integer operations, which
        // are slow on NVIDIA GPUs (precise analysis: 4x 64-bit multiplications + 4x 64-bit additions).
        // Therefore, we rearrange the expression to do only 2x 64-bit operations and 9x 32-bit operations:
        uint32_t prob01 = prob0 * prob1;                                                                    // 1x 32 bit multiplication
        uint32_t prob012 = prob01 * prob2;                                                                  // 1x 32 bit multiplication
        uint32_t prob0123 = prob012 * prob3;                                                                // 1x 32 bit multiplication
        uint32_t new_state1 = remainder0 + prob0 * remainder1 + prob01 * remainder2 + prob012 * remainder3; // 3x 32 bit multiplications + 3x 32 bit additions
        state = (state >> 28) * static_cast<uint64_t>(prob0123) + static_cast<uint64_t>(new_state1);        // 1x 64 bit multiplication + 1x 64 bit addition
        // Total cost: 1x 64 bit multiplication + 1x 64 bit addition + 6x 32 bit multiplications + 3x 32 bit additions
    }

    // Scale result and round to integer (resolving ties by rounding to nearest even):
    auto scaled_result = __float2int_rn(static_cast<float>(acc) * scale);
    // Clamp to `int8_t` range (should rarely have an effect for well-chosen `scale`):
    int8_t clipped_scaled_result = max(-128, min(127, scaled_result));

    reinterpret_cast<int8_t *>(out_vec)[row] = clipped_scaled_result;
}
