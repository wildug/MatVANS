#include "safetensor_loader.hpp"
#include <fstream>
#include <stdexcept>
#include <iostream>

// #define SAFETENSORS_CPP_IMPLEMENTATION
#include "safetensors.hh"

SafetensorFile::SafetensorFile(const std::string &path)
    : st(), databuffer(nullptr)
{
    std::string warn, err;
#if defined(USE_MMAP)
    printf("USE mmap\n");
    bool ret = safetensors::mmap_from_file(path, &this->st, &warn, &err);
#else
    bool ret = safetensors::load_from_file(path, &this->st, &warn, &err);
#endif

    if (warn.size())
    {
        std::cout << "WARN: " << warn << "\n";
    }

    if (!ret)
    {
        std::cerr << "Failed to load: " << path << "\n";
        std::cerr << "  ERR: " << err << "\n";

        throw EXIT_FAILURE;
    }

    // Check if data_offsets are valid.
    if (!safetensors::validate_data_offsets(st, err))
    {
        std::cerr << "Invalid data_offsets\n";
        std::cerr << err << "\n";

        throw EXIT_FAILURE;
    }

    if (st.mmaped)
    {
        this->databuffer = st.databuffer_addr;
    }
    else
    {
        this->databuffer = st.storage.data();
    }
}

// Method to get a tensor by name
Safetensor SafetensorFile::get_tensor(const std::string &name) const
{
    safetensors::tensor_t tensor;
    st.tensors.at(name, &tensor);

    return Safetensor{
        .tensor = tensor,
        .data = this->databuffer + tensor.data_offsets[0],
    };
}
