# CUDA Matrix Multiplication with cuTENSOR

This project implements matrix multiplication on NVIDIA GPUs using the cuTENSOR library. The program generates two random matrices, performs `C = alpha * A * B + beta * C` on the GPU, and saves the result for review.

## Video Demonstration

Watch the full workflow demonstration here:  
[CUDA cuTENSOR Matrix Multiplication Project Demo](https://www.tumblr.com/khronotop/797213927110049792/this-video-demonstrates-the-full-workflow-of?source=share)

## Codebook

See [Codebook.md](https://github.com/MarcoGigante/cutensor-capstone-project/blob/main/CodeBook.md) for a detailed description of all project files and structure.

## Install

Please refer to the INSTALL file in the project root for complete setup instructions, including driver, CUDA, cuTENSOR, and environment configuration.

## Usage and Output

Follow these steps to build and run:

```
make clean
make build
make run
```

Or run the convenience script:

```
./run.sh
```

The output matrix is written to `results/matrix_output.txt` in row-major order.

## Author

This project was developed for the GPU Specialization Capstone Project in the course [CUDA Advanced Libraries](https://www.coursera.org/learn/cuda-advanced-libraries/home/welcome), part of the [GPU Programming Specialization](https://www.coursera.org/specializations/gpu-programming) specialization from Johns Hopkins University on Coursera.