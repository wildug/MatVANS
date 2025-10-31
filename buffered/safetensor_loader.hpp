#ifndef SAFETENSOR_LOADER_HPP
#define SAFETENSOR_LOADER_HPP

#include <string>
#include <unordered_map>
#include <vector>

#include "safetensors.hh"

struct Safetensor
{
    safetensors::tensor_t tensor;
    const uint8_t *data;
};

/// A wrapper around the safetensors-cpp library with a slightly saner API.
class SafetensorFile
{
public:
    // Load a safetensor file from `path`
    explicit SafetensorFile(const std::string &path);

    // Get a tensor by name
    Safetensor get_tensor(const std::string &name) const;

private:
    safetensors::safetensors_t st;

    const uint8_t *databuffer;
};

#endif // SAFETENSOR_LOADER_HPP
