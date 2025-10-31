# Compressed Matrix-Vector Multiplication in CUDA

This directory implements the methods `ans-warp-rows` and `ans-warp-stripes` explained in the PDF-documentation in the parent directory.

## Encoding

- Create a python virtual environment, activate it, and install the required packages:

  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

- Then generate some random matrices and compress them:

  ```bash
  python encode.py --dim ⟨MATRIX_DIMENSION⟩ --mode ⟨COMPRESSION_MODE⟩ -o ⟨OUTPUT_FILENAME⟩.safetensors
  ```

  (run `python encode.py -h` for details.)


## Profiling the decoder on the GPU

- Make sure that `nvcc` is in your `PATH`.
- Run `make all`.
- Run either one of
  - `./baseline ⟨FILENAME⟩.safetensors`,
  - `./ans-warp-rows ⟨FILENAME⟩.safetensors`, or
  - `./ans-warp-stripes ⟨FILENAME⟩.safetensors`

  where `⟨FILENAME⟩.safetensors` is the file you created with the encoder as described above.
