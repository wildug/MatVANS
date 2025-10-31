#include <vector>
#include <array>
#include <span>
#include <tuple>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <iostream>

#include "sane_safetensors.hpp"

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif
#include <iomanip>

template <typename Word, typename State>
class DetachedAnsQuadCoder
{
    static_assert(std::is_unsigned_v<Word>);
    static_assert(std::is_unsigned_v<State>);
    static_assert(sizeof(State) > sizeof(Word));
    static_assert(sizeof(State) % sizeof(Word) == 0);

public:
    DetachedAnsQuadCoder(unsigned int precision, std::vector<Word> *buf = nullptr)
        : precision(precision), state(0)
    {
        assert(8 * sizeof(Word) >= precision * 4);
        reset();
    }

    void reset(std::vector<Word> *buf = nullptr)
    {
        const size_t STATE_SIZE = 8 * sizeof(State);
        const size_t WORD_SIZE = 8 * sizeof(Word);

        if (buf)
        {
            while ((state >> (STATE_SIZE - WORD_SIZE)) == 0)
            {
                state = (state << WORD_SIZE) | buf->back();
                buf->pop_back();
            }
        }
        else
        {
            state = 0; // Means uninitialized, will be set to a valid value on first encode.
        }
    }

    template <typename Symbol, typename Probability>
    void encode_quad(std::span<const Symbol> symbols, Symbol min_symbol, std::span<Probability> cdf, std::vector<Word> &buf)
    {
        static_assert(sizeof(Probability) <= sizeof(Word));
        assert(symbols.size() == 4);

        std::array<Probability, 4> left_cumulatives, right_cumulatives, probabilities;
        for (size_t i = 0; i < 4; ++i)
        {
            left_cumulatives[i] = cdf[symbols[i] - min_symbol];
            right_cumulatives[i] = cdf[symbols[i] - min_symbol + 1];
            probabilities[i] = right_cumulatives[i] - left_cumulatives[i];
        }

        State total_probability = std::accumulate(
            probabilities.begin(), probabilities.end(), static_cast<State>(1), [](Probability a, Probability b)
            { return static_cast<State>(a) * static_cast<State>(b); });

        if ((state >> (8 * sizeof(State) - 4 * precision)) >= total_probability)
        {
            // Encoding directly onto `state` would overflow.
            buf.push_back(static_cast<Word>(state));
            state >>= 8 * sizeof(Word);
        }
        else if (state == 0)
        {
            // `state` is not yet initialized. Initialize it to the smallest value that still
            // ensures that the condition for refilling is false once this method call is done.
            state = total_probability << (8 * sizeof(State) - 8 * sizeof(Word) - 4 * precision);
        }

        State concat = 0;
        for (size_t i = 0; i < 4; ++i)
        {
            State remainder = state % static_cast<State>(probabilities[i]);
            state /= static_cast<State>(probabilities[i]);
            State quantile = static_cast<State>(left_cumulatives[i] + remainder);
            concat |= quantile << (i * precision);
        }

        state = (state << (4 * precision)) | concat;
    }

    template <typename Symbol, typename Probability>
    std::array<Symbol, 4> decode_quad(std::span<const Symbol> ppf, std::span<const Probability> ppf_cum, std::span<const Probability> ppf_prob, std::vector<Word> &buf)
    {
        static_assert(sizeof(Probability) <= sizeof(Word));

        if ((state >> (8 * sizeof(State) - 8 * sizeof(Word))) == 0)
        {
            state = (state << 8 * sizeof(Word)) | buf.back();
            buf.pop_back();
        }

        std::array<State, 4> quantiles;
        for (size_t i = 0; i < 4; ++i)
        {
            quantiles[i] = (state >> (i * precision)) & ((static_cast<State>(1) << precision) - 1);
        }
        state >>= 4 * precision;

        std::array<Symbol, 4> symbols;
        for (size_t i = 0; i < 4; ++i)
        {
            symbols[i] = ppf[quantiles[i]];
        }

        for (size_t i = 0; i < 4; ++i)
        {
            Probability prob = ppf_prob[quantiles[i]];
            Probability left_cum = ppf_cum[quantiles[i]];
            Probability remainder = quantiles[i] - left_cum;
            state = state * static_cast<State>(prob) + static_cast<State>(remainder);
        }

        return symbols;
    }

    void finalize_to(std::vector<Word> &buf)
    {
        while (state != 0)
        {
            buf.push_back(static_cast<Word>(state & ((static_cast<State>(1) << precision) - 1)));
            state >>= 8 * sizeof(Word);
        }
    }

    State get_state() const
    {
        return state;
    }

private:
    const int precision;
    State state;
};

struct EncoderModelResult
{
    double entropy;
    double cross_entropy;
    int8_t min_value;
    std::vector<uint8_t> cdf;
};

EncoderModelResult fit_128bit_encoder_model(std::span<const int8_t> data)
{
    assert(!data.empty());

    // Count occurrences of each value
    std::vector<int> counts(256, 0);
    for (int8_t value : data)
    {
        counts[static_cast<uint8_t>(value)]++;
    }

    // Get unique values and their counts
    std::vector<int8_t> values;
    std::vector<int> value_counts;
    for (int value = -128; value < 128; ++value)
    {
        auto count = counts[static_cast<uint8_t>(value)];
        if (count > 0)
        {
            values.push_back(value);
            value_counts.push_back(count);
        }
    }

    int numel = data.size();
    std::vector<uint8_t> compacted_pmf(value_counts.size());
    for (size_t i = 0; i < value_counts.size(); ++i)
    {
        compacted_pmf[i] = std::max(uint8_t(1), static_cast<uint8_t>(std::round(value_counts[i] * 128.0f / numel)));
    }

    int freq_sum = std::accumulate(compacted_pmf.begin(), compacted_pmf.end(), 0);
    if (freq_sum > 128)
    {
        for (int i = 0; i < freq_sum - 128; ++i)
        {
            auto max_it = std::max_element(compacted_pmf.begin(), compacted_pmf.end());
            (*max_it)--;
        }
    }
    else if (freq_sum < 128)
    {
        for (int i = 0; i < 128 - freq_sum; ++i)
        {
            auto max_it = std::max_element(compacted_pmf.begin(), compacted_pmf.end());
            (*max_it)++;
        }
    }

    // Calculate entropy and cross-entropy
    double entropy = std::log2(numel) - std::accumulate(
                                            value_counts.begin(), value_counts.end(), 0.0,
                                            [](double sum, int count)
                                            { return sum + count * std::log2(count); }) /
                                            numel;

    double cross_entropy = 7.0 - std::inner_product(
                                     value_counts.begin(), value_counts.end(), compacted_pmf.begin(), 0.0,
                                     std::plus<>(), [](int count, float pmf)
                                     { return count * std::log2(pmf); }) /
                                     numel;

    // Build the CDF
    int8_t min_value = values.front();
    int8_t max_value = values.back();
    std::vector<uint8_t> cdf(static_cast<size_t>(max_value) - static_cast<size_t>(min_value) + 2, 0);
    uint8_t left_cumulative = 0;
    size_t last_index = 0;

    for (size_t i = 0; i < values.size(); ++i)
    {
        size_t index = values[i] - min_value;
        std::fill(cdf.begin() + last_index + 1, cdf.begin() + index + 1, left_cumulative);
        last_index = index;
        left_cumulative += static_cast<uint8_t>(compacted_pmf[i]);
    }
    cdf.back() = left_cumulative;

    return {entropy, cross_entropy, min_value, cdf};
}

struct DecoderModelResult
{
    std::vector<int8_t> ppf;
    std::vector<uint8_t> ppf_cum;
    std::vector<uint8_t> ppf_prob;
};

DecoderModelResult decoder_model_from_128bit_encoder_model(int8_t min_value, std::span<const uint8_t> cdf)
{
    std::vector<int8_t> ppf(128);
    std::vector<uint8_t> ppf_cum(128);
    std::vector<uint8_t> ppf_prob(128);

    for (size_t i = 0; i < cdf.size() - 1; ++i)
    {
        uint8_t left_cumulative = cdf[i];
        uint8_t right_cumulative = cdf[i + 1];
        for (uint8_t q = left_cumulative; q < right_cumulative; ++q)
        {
            ppf[q] = static_cast<int8_t>(i + min_value);
            ppf_cum[q] = left_cumulative;
            ppf_prob[q] = right_cumulative - left_cumulative;
        }
    }

    return {ppf, ppf_cum, ppf_prob};
}

struct CompressAnsWarpStripesResult
{
    std::vector<uint32_t> compressed;
    std::vector<uint32_t> warp_offsets;
    int8_t min_value;
    std::vector<uint8_t> cdf;
};

CompressAnsWarpStripesResult
compress_ans_warp_stripes(std::span<const int8_t> matrix, size_t out_dim, size_t in_dim)
{
    assert(out_dim % WARP_SIZE == 0);
    assert(in_dim % 4 == 0);
    assert(matrix.size() == out_dim * in_dim);

    // Fit the entropy model
    auto [entropy, cross_entropy, min_value, cdf] = fit_128bit_encoder_model(matrix);

    // Initialize coders and buffers
    std::vector<DetachedAnsQuadCoder<uint32_t, uint64_t>> coders(WARP_SIZE, DetachedAnsQuadCoder<uint32_t, uint64_t>(7));
    std::vector<uint32_t> warp_offsets(out_dim / WARP_SIZE);
    auto compressed = std::vector<uint32_t>();
    auto compressed_stripe = std::vector<uint32_t>();

    const size_t stripe_stride = in_dim * WARP_SIZE;
    for (auto stripe = matrix.begin(); stripe < matrix.end(); stripe += stripe_stride)
    {
        warp_offsets[(stripe - matrix.begin()) / stripe_stride] = compressed.size() / 4; // Store offset in 128-bit words

        compressed_stripe.clear();
        for (auto &coder : coders)
        {
            coder.reset();
        }

        // Iterate over quad columns in reverse order.
        auto quad_col = stripe + in_dim;
        while (quad_col > stripe)
        {
            quad_col -= 4;

            auto coder = coders.begin();
            for (auto quad = quad_col; quad < quad_col + stripe_stride; quad += in_dim)
            {
                coder->encode_quad(std::span(quad, 4), min_value, std::span(cdf), compressed);
                ++coder;
            }
        }

        // Rearrange the stripe's compressed data into the order expected by the decoder
        // by (conceptually) doing the following steps:
        // 1. Reverse the order of the joint bulk compressed data of all coders in the warp.
        // 2. Pad the joint bulk data so that its length in words modulo `4 * warp_size`
        //    is `2 * warp_size`.
        // 3. Insert the coder states between index `2 * warp_size-1` and `2 * warp_size`
        //    of the joint bulk data by first inserting the `warp_size` low significant
        //    words of the coder states, followed by the `warp_size` high significant
        //    words of the coder states.
        // 4. Transpose the resulting data (whose length in words is now a multiple of
        //    `4 * warp_size`) in such a way that the decoder can read it in chunks, where
        //    each chunk is a matrix of dimensions `warp_size x 4`, which is read in from
        //    the in_file in row-major order but then traversed in column-major order.
        std::reverse(compressed_stripe.begin(), compressed_stripe.end());
        auto padding_size = (6 * WARP_SIZE - compressed_stripe.size() % (4 * WARP_SIZE)) % (4 * WARP_SIZE);
        compressed_stripe.insert(compressed_stripe.end(), padding_size, 0);

        // Make space for final coder states; `compressed_stripe.begin() + 2 * warp_size` exists because of the above padding.
        compressed_stripe.insert(compressed_stripe.begin() + 2 * WARP_SIZE, 2 * WARP_SIZE, 0);

        for (auto i = 0; i < WARP_SIZE; ++i)
        {
            compressed_stripe[2 * WARP_SIZE + i] = coders[i].get_state() & 0xffffffff;
            compressed_stripe[3 * WARP_SIZE + i] = coders[i].get_state() >> 32;
        }

        compressed.reserve(compressed.size() + compressed_stripe.size());
        for (auto chunk = compressed_stripe.begin(); chunk < compressed_stripe.end(); chunk += 4 * WARP_SIZE)
        {
            for (size_t i = 0; i < WARP_SIZE; ++i)
            {
                for (size_t j = 0; j < 4; ++j)
                {
                    compressed.push_back(chunk[i + j * WARP_SIZE]);
                }
            }
        }
    }

    return CompressAnsWarpStripesResult{compressed, warp_offsets, min_value, cdf};
}

void print_data_summary(std::span<const int8_t> initial_vector, std::span<const float> scales)
{
    const auto num_matrices = scales.size();
    const auto dim = initial_vector.size();
    auto megabytes = static_cast<float>(num_matrices * dim * dim * sizeof(int8_t)) / (1024.0 * 1024.0);
    std::cout << "Read " << num_matrices << " matrices of dimensions "
              << dim << "x" << dim << " each; "
              << "total uncompressed size: " << std::fixed << std::setprecision(2) << megabytes << " MiB."
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
    for (auto i = 0; i < std::min(num_matrices, size_t{10}); i++)
    {
        std::cout << scales[i] << ", ";
    }
    if (scales.size() > 10)
        std::cout << "...";
    std::cout << "]" << std::endl;
}

int main(int argc, char **argv)
{
    // Load uncompressed matrices from disk.
    assert(argc == 3);
    std::string in_filename = argv[1];
    std::string out_filename = argv[2];
    auto in_file = SaneSafetensorFileReader(in_filename);

    auto [matrices, matrices_shape] = in_file.get_tensor<int8_t, 3>("matrices");
    size_t num_matrices = matrices_shape[0];
    size_t dim = matrices_shape[1];
    assert(dim % WARP_SIZE == 0);
    assert(matrices_shape[2] == dim);

    auto initial_vector = in_file.get_data<int8_t, 1>("initial_vector", std::move(std::array{dim}));
    auto scales = in_file.get_data<float, 1>("scales", std::move(std::array{num_matrices}));
    auto hashes = in_file.get_data<uint32_t, 1>("hashes", std::move(std::array{num_matrices}));

    print_data_summary(initial_vector, scales);

    auto all_compressed = std::vector<uint32_t>();
    auto all_warp_offsets = std::vector<uint32_t>();
    auto all_ppfs = std::vector<int8_t>();
    auto all_ppfs_cum = std::vector<uint8_t>();
    auto all_ppfs_prob = std::vector<uint8_t>();

    auto compressed_size = 0;
    for (size_t i = 0; i < num_matrices; ++i)
    {
        std::cout << "Compressing matrix " << i + 1 << " of " << num_matrices << " ... " << std::endl;
        auto matrix = matrices.subspan(i * dim * dim, dim * dim);
        auto [compressed, warp_offsets, min_value, cdf] = compress_ans_warp_stripes(matrix, dim, dim);
        auto [ppf, ppf_cum, ppf_prob] = decoder_model_from_128bit_encoder_model(min_value, std::span(cdf));

        all_compressed.insert(all_compressed.end(), compressed.begin(), compressed.end());
        for (auto warp_offset : warp_offsets)
        {
            all_warp_offsets.push_back(warp_offset + compressed_size);
        }
        all_ppfs.insert(all_ppfs.end(), ppf.begin(), ppf.end());
        all_ppfs_cum.insert(all_ppfs_cum.end(), ppf_cum.begin(), ppf_cum.end());
        all_ppfs_prob.insert(all_ppfs_prob.end(), ppf_prob.begin(), ppf_prob.end());

        // The CUDA kernel accesses memory in 128-bit units.
        compressed_size += compressed.size() / 4;
    }

    // Assume the machine has little-endian architecture.
    std::span<uint32_t> all_ppfs_u32(reinterpret_cast<uint32_t *>(all_ppfs.data()), all_ppfs.size() / 4);
    std::span<uint32_t> all_ppfs_cum_u32(reinterpret_cast<uint32_t *>(all_ppfs_cum.data()), all_ppfs_cum.size() / 4);
    std::span<uint32_t> all_ppfs_prob_u32(reinterpret_cast<uint32_t *>(all_ppfs_prob.data()), all_ppfs.size() / 4);

    SaneSafetensorFileWriter writer;
    writer.add_tensor("initial_vector", initial_vector, {dim});
    writer.add_tensor("scales", scales, {num_matrices});
    writer.add_tensor("hashes", hashes, {num_matrices});
    writer.add_tensor("compressed", std::span(all_compressed), {all_compressed.size()});
    writer.add_tensor("all_warp_offsets", std::span(all_warp_offsets), {num_matrices, dim / WARP_SIZE});
    writer.add_tensor("all_ppfs", all_ppfs_u32, {num_matrices, 32});
    writer.add_tensor("all_ppfs_cum", all_ppfs_cum_u32, {num_matrices, 32});
    writer.add_tensor("all_ppfs_prob", all_ppfs_prob_u32, {num_matrices, 32});
    std::cout << "Saving to " << out_filename << " ... " << std::endl;
    writer.save_to_file(out_filename);
    std::cout << "Done." << std::endl;
}
