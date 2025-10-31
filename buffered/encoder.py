import numpy as np
import argparse
from safetensors.numpy import save_file
from tqdm import tqdm
import sys


def hash_int8_array(arr):
    """Simple hasher adopted from the "FxHash" rust crate.
    Not cryptographically secure, but _accidental_ collisions are very unlikely."""

    assert arr.dtype == np.int8

    hash = 0
    for byte in arr.astype(np.uint8).ravel():
        # Rotate left by 5 bits, then mix in the new byte.
        hash = ((hash >> 27) | (hash << 5)) & 0xffff_ffff
        hash = ((hash ^ byte.item()) * 0x2722_0a95) & 0xffff_ffff

    return hash


def create_rnd_matrices_and_vec(dim, count, target_entropy):
    std = 2**target_entropy / np.sqrt(2 * np.pi * np.e)
    initial_vector = np.round(
        np.random.normal(0, 4.0, size=(dim,))
    ).clip(-127.0, 127.0).astype(np.int8)

    print(
        f"Initial vector (dimension {dim}): "
        f"mean={initial_vector.mean(): 2.2f}, std={initial_vector.std():.2f}, "
        f"min={initial_vector.min()}, max={initial_vector.max()}, "
        f"hash={hash_int8_array(initial_vector):08x}"
    )

    # Create matrices one by one so that we never have to hold all
    # matrices in float32-format in memory at the same time.
    matrices = np.empty((count, dim, dim), dtype=np.int8)
    vector = initial_vector
    scales = np.empty((count,), dtype=np.float32)
    hashes = np.empty((count,), dtype=np.uint32)
    for i in range(count):
        matrices[i] = np.round(
            np.random.normal(0, std, size=(dim, dim))
        ).clip(-127.0, 127.0).astype(np.int8)
        vector = matrices[i] @ vector.astype(np.int32)

        # ensure that `vector * scales[i]` fits into int8
        scales[i] = 127.0 / np.abs(vector).max()
        vector = np.round(vector * scales[i]).astype(np.int8)
        hashes[i] = hash_int8_array(vector)

        print(
            f"Matrix {i} ({dim}x{dim}): "
            f"mean={matrices[i].mean(): 2.2f}, std={matrices[i].std():.2f}, "
            f"min={matrices[i].min()}, max={matrices[i].max()}, "
            f"scale={scales[i]:.2g}; "
            f"hash(vector)={hashes[i]:08x}"
        )

    return matrices, initial_vector, scales, hashes


class ArrayVector:
    def __init__(self, data=None, dtype=None, initial_capacity=32):
        if data is not None:
            assert isinstance(data, np.ndarray)
            assert dtype is None or data.dtype == dtype
            self.data = data
            self.capacity = len(data)
            self.size = len(data)
        else:
            assert dtype is not None
            self.data = np.empty((initial_capacity,), dtype=dtype)
            self.capacity = initial_capacity
            self.size = 0

    def push_back(self, x):
        if self.size == self.capacity:
            self.capacity = max(self.capacity * 2, 32)
            new_data = np.empty((self.capacity,), dtype=self.data.dtype)
            new_data[:self.size] = self.data
            self.data = new_data

        self.data[self.size] = x
        self.size += 1

    def pop_back(self):
        assert self.size > 0
        self.size -= 1
        return self.data[self.size].item()

    def append(self, x):
        new_size = self.size + len(x)
        if new_size >= self.capacity:
            self.capacity = max(self.capacity * 2, new_size)
            new_data = np.empty((self.capacity,), dtype=self.data.dtype)
            new_data[:self.size] = self.data[:self.size]
            self.data = new_data

        self.data[self.size:new_size] = x
        self.size = new_size

    def chop_off(self, n):
        assert n <= self.size
        self.size -= n
        return ArrayVector(data=self.data[self.size:self.size + n].copy())

    def finalize(self):
        return self.data[:self.size]


class DetachedAnsQuadCoder:
    """The "head" of an ANS-like entropy coder that encodes/decodes in chunks of 4 symbols.

    This class only holds a fixed size (up to `state_size` bits) of compressed
    data. Therefore, the `encode` and `decode` function both expect a `buf`
    argument of type `ArrayVector`, which they may use to transfer up to 1 word
    of compressed data from or to the initial state when it overflows or
    underflows, respectively.

    Assumes that the `word_size` is at least `4 * precision`, i.e., that 
    encoding 4 symbols never creates more than one word of compressed data.

    The encoding and decoding operations are as follows:

    - Decoding is designed to be efficient on `word_size`-bit hardware:
      - Interpret `state` as the following concatenation of bits, from MSB to LSB:
        `x | quantile[3] | quantile[2] | quantile[1] | quantile[0]`,
        where `quantile[i]` is `precision` bits wide, and `x` is the rest.
      - Use `quantile[i]` to look up the 4 decoded symbols and their probabilities
        `prob[i]` and exclusive cumulative probabilities `cdf[i]`.
      - Compute the remainders as `rem[i] = quantile[i] - cdf[i]`
      - Update `state` as follows:
        `state = (((x * prob[3] + rem[3]) * prob[2] + rem[2]) * prob[1] + rem[1]) * prob[0] + rem[0]`
        (this can be done in 2x 46-bit operations and 9x 32-bit operations)
      - Do a simple check if refilling is possible: transfer one word from the
        provided buffer to `state` if there is space, i.e., if
        `state >> (state_size - word_size) == 0` (this is actually done at the
        beginning rather than the end of the `decode` method, see discussion below).

    - Encoding is inference over the decoder, i.e., it inverts the above steps
      in reverse order:
      - Check if encoding directly onto `state` would overflow: if
        `(state >> (state_size - 4 * precision)) >= total_prob`, then flush the
        lowest significant word from `state` to `bulk`. Here, `total_prob` is the
        product of the probabilities of the 4 symbols to be encoded.
      - For each `i` in {0, 1, 2, 3}: `rem[i] = state % prob[i]; state /= prob[i]`;
        interpret the value of `state` after these operations as `x`.
      - Calculate the corresponding quantiles: `quantile[i] = cdf[i] + rem[i]`.
      - Update `state` by setting it to the concatenation stated above:
        `state = x | quantile[3] | quantile[2] | quantile[1] | quantile[0]`

    To ensure that the decoder does not read past the end of the compressed data,
    one could initialize `state` in the encoder to `1 << (state_size - word_size)`
    (so that the condition for refilling is false at the end of decoding). We 
    instead do a minor variation here: we actually move the check for refilling
    in the decoder to the beginning of each 4-symbol chunk. This allows us to
    initialize `state` in the encoder to a slightly smaller value as the encoder
    only has to ensure that the condition for refilling is false _after encoding
    the first 4-symbol chunk_.
    """

    def __init__(self, precision, word_size, state_size, buf=None):
        """Initializes a `DetachedAnsQuadCoder`.

        Arguments:
            precision: The number of bits used to represent probabilities.
            word_size: The number of bits in a word of compressed data;
                       must be at least `4 * precision` so that encoding 4
                       symbols emits at most one word of compressed data.
            state_size: The number of bits used to represent the internal state;
                        must be a multiple of `word_size`.
            buf: An optional `ArrayVector` to read the initial state from
        """
        assert word_size >= precision * 4
        assert state_size % word_size == 0

        self.precision = precision
        self.word_size = word_size
        self.state_size = state_size
        self.state = 0
        self.word_mask = (1 << word_size) - 1
        self.prob_mask = (1 << precision) - 1

        if buf is not None:
            while (self.state >> (state_size - word_size)) == 0:
                self.state = (self.state << word_size) | buf.pop_back()

    def encode(self, symbols, min_symbol, cdf, buf):
        """Encodes 4 symbols.

        Arguments:
            symbols: an array, list, or tuple of the 4 symbols to encode.
            min_symbol: the minimum symbol value (used to convert symbols to
                        indices into `cdf`).
            cdf: The (exclusive) cumulative distribution function as a 1D numpy
                 array of integers. Must be one item longer than the range
                 of symbols, and the last entry must be `1 << precision`.
            buf: An `ArrayVector` to which 1 word may be pushed if necessary.
        """
        left_cumulatives = cdf[symbols - min_symbol]
        right_cumulatives = cdf[symbols - min_symbol + 1]
        probabilities = right_cumulatives - left_cumulatives
        # `np.prod` automatically upcasts to uint32 or uint64
        total_probability = np.prod(probabilities).item()

        if (self.state >> (self.state_size - 4 * self.precision)) >= total_probability:
            # Encoding directly onto `state` would overflow.
            buf.push_back(self.state & self.word_mask)
            self.state >>= self.word_size
        elif self.state == 0:
            # `state` is not yet initialized. Initialize it to the smallest
            # value that still ensures that the condition for refilling is
            # false once this method call is done.
            self.state = total_probability << (
                self.state_size - self.word_size - 4 * self.precision)

        concat = 0
        for i, (prob, left_cum) in enumerate(zip(probabilities, left_cumulatives)):
            remainder = self.state % prob.item()
            self.state //= prob.item()
            quantile = left_cum.item() + remainder
            concat |= quantile << (i * self.precision)

        self.state = (self.state << (4 * self.precision)) | concat

    def decode(self, ppf, ppf_cum, ppf_prob, buf):
        """Decodes 4 symbols.

        Arguments:
            ppf: The percent point function (inverse of a cdf): `ppf[quantile]`
                 is the largest `symbol` such that `cdf[symbol] <= quantile`, where
                 `cdf` is the (exclusive) cumulative distribution function.
            ppf_cum: The concatenation of ppf and cdf, i.e.,
                     `ppf_cum[quantile] = cdf[ppf[quantile]]`.
            ppf_prob: The concatenation of ppf and probability mass function, i.e.,
                      `ppf_prob[quantile] = cdf[ppf[quantile] + 1] - cdf[ppf[quantile]]`.
            buf: An `ArrayVector` from which 1 word may be refilled if necessary.

        Returns the decoded symbol.
        """
        # Refill `state` if possible:
        if self.state >> (self.state_size - self.word_size) == 0:
            self.state = (self.state << self.word_size) | buf.pop_back()

        # Decode symbols:
        quantiles = np.array([(self.state >> (i * self.precision)) & self.prob_mask
                              for i in range(4)])
        self.state >>= 4 * self.precision
        symbols = ppf[quantiles]

        # Re-encode unused information onto `state` ("bits-back trick"):
        probs = ppf_prob[quantiles]
        left_cums = ppf_cum[quantiles]
        remainders = quantiles - left_cums
        for prob, rem in zip(reversed(probs), reversed(remainders)):
            self.state = self.state * prob.item() + rem.item()

        return symbols

    def finalize_to(self, buf):
        """Writes the remaining encapsulated data to the end of the `ArrayVector`
        `buf` as `self.state_size // self.word_size` words and resets this
        `DetachedAnsQuadCoder` for writing as if it was newly initialized."""
        while self.state != 0:
            buf.push_back(self.state & self.word_mask)
            self.state >>= self.word_size


def fit_128bit_encoder_model(data):
    assert data.dtype == np.int8

    numel = data.size  # `.size` in numpy is like `.numel()` in PyTorch
    values, counts = np.unique(data, return_counts=True)

    # Build CDF with 7 bit precision (approximate for now):
    counts = counts.astype(np.float32)
    compacted_pmf = np.maximum(1, np.round(
        counts * 128 / numel).astype(np.uint32))
    freq_sum = compacted_pmf.sum()
    if freq_sum > 128:
        for _ in range(freq_sum - 128):
            compacted_pmf[np.argmax(compacted_pmf)] -= 1
    else:
        for _ in range(128 - freq_sum):
            compacted_pmf[np.argmax(compacted_pmf)] += 1

    # Calculate entropy with and without overhead due to low-precision probabilities:
    entropy = np.log2(numel) - np.sum(counts * np.log2(counts)) / numel
    cross_entropy = 7 - np.sum(counts * np.log2(compacted_pmf)) / numel

    # Invert CDF to obtain PPF; also decompactify cdf for encoder:
    min_value, max_value = values[0].item(), values[-1].item()
    cdf = np.zeros((max_value - min_value + 2,), dtype=np.uint8)
    left_cumulative = 0
    last_index = 0
    for v, f in zip(values, compacted_pmf):
        index = v.item() - min_value
        cdf[last_index+1:index+1] = left_cumulative
        last_index = index
        left_cumulative += f.item()
    assert left_cumulative == 128
    cdf[-1] = left_cumulative

    return entropy, cross_entropy, min_value, cdf


def decoder_model_from_128bit_encoder_model(min_value, cdf):
    ppf = np.empty((128,), dtype=np.int8)
    ppf_cum = np.empty((128,), dtype=np.uint8)
    ppf_prob = np.empty((128,), dtype=np.uint8)

    for index, (left_cumulative, right_cumulative) in enumerate(zip(cdf[:-1], cdf[1:])):
        ppf[left_cumulative:right_cumulative] = index + min_value
        ppf_cum[left_cumulative:right_cumulative] = left_cumulative
        ppf_prob[left_cumulative:right_cumulative] = right_cumulative - \
            left_cumulative

    return ppf, ppf_cum, ppf_prob


def compress_ans_warp_stripes(matrix, warp_size):
    assert matrix.dtype == np.int8
    out_dim, in_dim = matrix.shape  # asserts that `matrix` has rank 2
    assert out_dim % warp_size == 0
    assert in_dim % 4 == 0

    entropy, cross_entropy, min_value, cdf = fit_128bit_encoder_model(matrix)

    compressed = ArrayVector(dtype=np.uint32)
    coders = [DetachedAnsQuadCoder(7, 32, 64) for _ in range(warp_size)]
    warp_offsets = np.empty((out_dim // warp_size,), dtype=np.uint32)

    # Reorder the matrix elements in the order we process them:
    # - in stripes of `warp_size` rows,
    # - within each stripe: in chunks of 4 columns (in reverse),
    # - within each chunk: in `out_dim // warp_size` rows of length 4 each.
    reordered_matrix = matrix.reshape(
        (out_dim // warp_size, warp_size, in_dim // 4, 4)).transpose((0, 2, 1, 3))
    for i, stripe in enumerate(tqdm(reordered_matrix)):
        warp_offsets[i] = compressed.size // 4
        compressed_stripe = ArrayVector(dtype=np.uint32)
        for quad_col in reversed(stripe):
            for coder, quad in zip(reversed(coders), reversed(quad_col)):
                coder.encode(quad, min_value, cdf, compressed_stripe)

        # Rearrange the stripe's compressed data into the order expected by the decoder
        # by (conceptually) doing the following steps:
        # 1. Reverse the order of the joint bulk compressed data of all coders in the warp.
        # 2. Pad the joint bulk data so that its length in words modulo `4 * warp_size`
        #    is `2 * warp_size`.
        # 3. Insert the coder states between index `2 * warp_size-1` and `2 * warp_size`
        #    of the joint bulk data by first inserting the `warp_size` low significant
        #    words of the coder states, followed by the `warp_size` high significant
        #    words of the coder states.
        # 4. Transpose the resulting data (whose length in words is now a multiple of
        #    `4 * warp_size`) in such a way that the decoder can read it in chunks, where
        #    each chunk is a matrix of dimensions `warp_size x 4`, which is read in from
        #    the file in row-major order but then traversed in column-major order.
        bulk = compressed_stripe.finalize()[::-1]
        padding = np.zeros(
            ((2 * warp_size - bulk.size) % (4 * warp_size),), dtype=np.uint32)
        bulk = np.concatenate((bulk, padding))

        # print(
        #     f"stripe {i}, final coder states: {[coder.state for coder in coders]}")
        untransposed = ArrayVector(data=bulk[:2 * warp_size])
        for coder in coders:
            untransposed.push_back(coder.state & 0xffff_ffff)
        for coder in coders:
            untransposed.push_back(coder.state >> 32)
        untransposed.append(bulk[2 * warp_size:])

        untransposed = untransposed.finalize().reshape((-1, 4, warp_size))
        compressed_stripe = untransposed.transpose((0, 2, 1)).ravel()
        compressed.append(compressed_stripe)

    compressed = compressed.finalize()
    if sys.byteorder != 'little':
        # NVIDIA GPUs are little-endian.
        compressed.byteswap(inplace=True)

    print(
        f"Compressed {out_dim}x{in_dim} matrix.\n"
        f"- entropy       = {entropy:.2f} bit/element\n"
        f"- cross entropy = {cross_entropy:.2f} bit/element (includes overhead due to 7-bit probabilities)\n"
        f"- bit rate      = {compressed.nbytes * 8 / (out_dim * in_dim):.2f} bit/element "
        f"({compressed.nbytes / 1024:.1f} KiB total)")

    return compressed, warp_offsets, min_value, cdf


def compress_ans_warp_rows(matrix, warp_size):
    assert matrix.dtype == np.int8
    out_dim, in_dim = matrix.shape  # asserts that `matrix` has rank 2
    assert in_dim % (4 * warp_size) == 0

    entropy, cross_entropy, min_value, cdf = fit_128bit_encoder_model(matrix)

    compressed = ArrayVector(dtype=np.uint32)
    coders = [DetachedAnsQuadCoder(7, 32, 64) for _ in range(warp_size)]
    warp_offsets = np.empty((out_dim,), dtype=np.uint32)

    # Reshape the matrix elements in the order we process them:
    # - in rows
    # - within each row: in chunks of size `4 * warp_size`
    # - within each chunk: in "quads", i.e., groups of 4
    reshaped_matrix = matrix.reshape(
        (out_dim, in_dim // (warp_size * 4), warp_size, 4))
    for i, row in enumerate(tqdm(reshaped_matrix)):
        warp_offsets[i] = compressed.size // 4
        compressed_row = ArrayVector(dtype=np.uint32)
        for warp_chunk in reversed(row):
            for coder, quad in zip(reversed(coders), reversed(warp_chunk)):
                coder.encode(quad, min_value, cdf, compressed_row)

        # Rearrange the row's compressed data into the order expected by the decoder
        # by (conceptually) doing the following steps:
        # 1. Reverse the order of the joint bulk compressed data of all coders in the warp.
        # 2. Pad the joint bulk data so that its length in words modulo `4 * warp_size`
        #    is `2 * warp_size`.
        # 3. Insert the coder states between index `2 * warp_size-1` and `2 * warp_size`
        #    of the joint bulk data by first inserting the `warp_size` low significant
        #    words of the coder states, followed by the `warp_size` high significant
        #    words of the coder states.
        # 4. Transpose the resulting data (whose length in words is now a multiple of
        #    `4 * warp_size`) in such a way that the decoder can read it in chunks, where
        #    each chunk is a matrix of dimensions `warp_size x 4`, which is read in from
        #    the file in row-major order but then traversed in column-major order.
        bulk = compressed_row.finalize()[::-1]
        padding = np.zeros(
            ((2 * warp_size - bulk.size) % (4 * warp_size),), dtype=np.uint32)
        bulk = np.concatenate((bulk, padding))

        # print(
        #     f"stripe {i}, final coder states: {[coder.state for coder in coders]}")
        untransposed = ArrayVector(data=bulk[:2 * warp_size])
        for coder in coders:
            untransposed.push_back(coder.state & 0xffff_ffff)
        for coder in coders:
            untransposed.push_back(coder.state >> 32)
        untransposed.append(bulk[2 * warp_size:])

        untransposed = untransposed.finalize().reshape((-1, 4, warp_size))
        compressed_row = untransposed.transpose((0, 2, 1)).ravel()
        compressed.append(compressed_row)

    compressed = compressed.finalize()
    if sys.byteorder != 'little':
        # NVIDIA GPUs are little-endian.
        compressed.byteswap(inplace=True)

    print(
        f"Compressed {out_dim}x{in_dim} matrix.\n"
        f"- entropy       = {entropy:.2f} bit/element\n"
        f"- cross entropy = {cross_entropy:.2f} bit/element (includes overhead due to 7-bit probabilities)\n"
        f"- bit rate      = {compressed.nbytes * 8 / (out_dim * in_dim):.2f} bit/element "
        f"({compressed.nbytes / 1024:.1f} KiB total)")

    return compressed, warp_offsets, min_value, cdf


def pack_4bytes(arr):
    """Converts (4*n)-entry [u]int8 arrays to n-entry uint32 arrays using little-endian packing."""
    assert arr.dtype == np.uint8 or arr.dtype == np.int8
    assert arr.ndim == 1

    return arr.astype(np.uint8).reshape((-1, 4)).dot(
        np.array([1, 256, 256**2, 256**3], dtype=np.uint32))


def compress_all_matrices(matrices, initial_vector, scales, hashes, compressor, filename):
    compressed, all_warp_offsets, all_ppfs, all_ppfs_cum, all_ppfs_prob = [], [], [], [], []

    compressed_size = 0
    for i, matrix in enumerate(matrices):
        print(f"Compressing matrix {i+1} of {len(matrices)}...")
        compressed_i, warp_offsets, min_value, cdf = compressor(matrix, 32)
        ppf, ppf_cum, ppf_prob = decoder_model_from_128bit_encoder_model(
            min_value, cdf)

        compressed.append(compressed_i)
        all_warp_offsets.append(warp_offsets + compressed_size)
        all_ppfs.append(pack_4bytes(ppf))
        all_ppfs_cum.append(pack_4bytes(ppf_cum))
        all_ppfs_prob.append(pack_4bytes(ppf_prob))

        # The CUDA kernel accesses memory in 128-bit units.
        compressed_size += len(compressed_i) // 4

    all_warp_offsets = np.stack(all_warp_offsets)
    all_ppfs = np.stack(all_ppfs)
    all_ppfs_cum = np.stack(all_ppfs_cum)
    all_ppfs_prob = np.stack(all_ppfs_prob)
    compressed = np.concatenate(compressed)

    save_file(
        {
            "initial_vector": initial_vector,
            "scales": scales,
            "hashes": hashes,
            "compressed": compressed,
            "all_warp_offsets": all_warp_offsets,
            "all_ppfs": all_ppfs,
            "all_ppfs_cum": all_ppfs_cum,
            "all_ppfs_prob": all_ppfs_prob,
        },
        filename,
    )
    print(f"Saved compressed matrices and initial vector to `{filename}`.")


def save_uncompressed_matrices_and_vec(matrices, initial_vector, scales, hashes, filename):
    save_file(
        {
            "matrices": matrices,
            "initial_vector": initial_vector,
            "scales": scales,
            "hashes": hashes,
        },
        filename,
    )
    print(f"Saved uncompressed matrices and initial vector to `{filename}`.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate random matrices and vectors.")
    parser.add_argument("--dim", type=int, default=4096,
                        help="Dimension of the matrices and vector.")
    parser.add_argument("--count", type=int, default=10,
                        help="Number of matrices to generate.")
    parser.add_argument("--target_entropy", type=float, default=4.0,
                        help="Target entropy (bits per matrix element).")
    parser.add_argument("--seed", type=int, default=20250829,
                        help="Seed for random number generator.")
    parser.add_argument(
        "--warp_size", type=int, default=32,
        help="Warp size (defaults to 32).")
    parser.add_argument(
        "--mode", type=str, choices=["uncompressed", "ans-warp-rows", "ans-warp-stripes"], required=True,
        help="Mode of operation: 'uncompressed' to save uncompressed matrices, "
             "'ans-warp-{rows, stripes}' to save compressed matrices.")
    parser.add_argument(
        "-o", "--output", type=str, required=True,
        help="Output file name.")

    args = parser.parse_args()

    assert args.dim % 16 == 0, \
        "ERROR: Dimension must be a multiple of 16 because we read data in chunks of 16 bytes."

    np.random.seed(args.seed)
    matrices, initial_vector, scales, hashes = create_rnd_matrices_and_vec(
        args.dim, args.count, args.target_entropy)

    if args.mode == "uncompressed":
        save_uncompressed_matrices_and_vec(
            matrices, initial_vector, scales, hashes, args.output)
    else:
        if args.mode == "ans-warp-rows":
            compressor = compress_ans_warp_rows
        elif args.mode == "ans-warp-stripes":
            compressor = compress_ans_warp_stripes
        else:
            raise ValueError(f"Unknown mode: {args.mode}")

        compress_all_matrices(
            matrices, initial_vector, scales, hashes, compressor, args.output)
