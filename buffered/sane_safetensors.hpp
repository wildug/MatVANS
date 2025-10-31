#include <span>

#if __cplusplus >= 202300L
#include <cstdfloat>
#endif

#include "safetensor_loader.hpp"
#include <numeric>
#include <cstring> // memcpy

template <typename Dtype>
safetensors::dtype get_dtype();
// {
//     return safetensors::dtype::kUINT8; // Default to bytes for unknown types.
// }

class SaneSafetensorFile
{
public:
    SaneSafetensorFile(const std::string &path) : file(path) {}

    /**
     * @brief Retrieves a tensor by its name.
     *
     * Returns a tuple `{data, shape}`, where `data` is a pointer to the tensor's data.
     * Verifies that the tensor has the expected dtype and rank.
     *
     * Use most easily in structured bindings:
     *
     * ```cpp
     * auto [data, shape] = file.get_tensor<float, 2>("tensor_name");
     * ```
     */
    template <typename Dtype, size_t rank>
    std::tuple<std::span<const Dtype>, std::array<size_t, rank>> get_tensor(const std::string &tensor_name) const
    {
        const auto tensor = this->file.get_tensor(tensor_name);
        assert(tensor.tensor.dtype == get_dtype<Dtype>());
        assert(tensor.tensor.shape.size() == rank);

        auto data = reinterpret_cast<const Dtype *>(tensor.data);
        auto size = (tensor.tensor.data_offsets[1] - tensor.tensor.data_offsets[0]) / sizeof(Dtype);
        std::array<size_t, rank> shape;
        std::copy(tensor.tensor.shape.begin(), tensor.tensor.shape.end(), shape.begin());

        return {std::span(data, size), shape};
    }

    /**
     * @brief Retrieves a tensor's data by its name, verifying its shape and dtype.
     *
     * Pass the `expected_shape` as an rvalue (e.g. `std::move(std::array{3, 5})`).
     */
    template <typename Dtype, size_t rank>
    const std::span<const Dtype> get_data(const std::string &tensor_name, std::array<size_t, rank> &&expected_shape) const
    {
        const auto tensor = this->file.get_tensor(tensor_name);
        assert(tensor.tensor.dtype == get_dtype<Dtype>());
        assert(std::equal(tensor.tensor.shape.begin(), tensor.tensor.shape.end(), expected_shape.begin(), expected_shape.end()));

        auto data = reinterpret_cast<const Dtype *>(tensor.data);
        auto size = (tensor.tensor.data_offsets[1] - tensor.tensor.data_offsets[0]) / sizeof(Dtype);
        return std::span(data, size);
    }

private:
    SafetensorFile file;
};

template <>
safetensors::dtype get_dtype<bool>()
{
    return safetensors::dtype::kBOOL;
}

template <>
safetensors::dtype get_dtype<uint8_t>()
{
    return safetensors::dtype::kUINT8;
}

template <>
safetensors::dtype get_dtype<int8_t>()
{
    return safetensors::dtype::kINT8;
}

template <>
safetensors::dtype get_dtype<int16_t>()
{
    return safetensors::dtype::kINT16;
}

template <>
safetensors::dtype get_dtype<uint16_t>()
{
    return safetensors::dtype::kUINT16;
}

template <>
safetensors::dtype get_dtype<int32_t>()
{
    return safetensors::dtype::kINT32;
}

template <>
safetensors::dtype get_dtype<uint32_t>()
{
    return safetensors::dtype::kUINT32;
}

template <>
safetensors::dtype get_dtype<float>()
{
    return safetensors::dtype::kFLOAT32;
}

template <>
safetensors::dtype get_dtype<double>()
{
    return safetensors::dtype::kFLOAT64;
}

template <>
safetensors::dtype get_dtype<int64_t>()
{
    return safetensors::dtype::kINT64;
}

template <>
safetensors::dtype get_dtype<uint64_t>()
{
    return safetensors::dtype::kUINT64;
}

#if __cplusplus >= 202300L

template <>
safetensors::dtype get_dtype<std::float16_t>()
{
    return safetensors::dtype::kFLOAT16;
}

template <>
safetensors::dtype get_dtype<std::bfloat16_t>()
{
    return safetensors::dtype::kBFLOAT16;
}

#endif
