# CUDA GeMM

In this project we run different cuda GeMM kernels for different matrix sizes for Benchmarks. The cuda kernels includes GeMM from cublas, cutlass and developed cuda GeMM kernels. 

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Code Structure](#code-structure)

## Prerequisites
Before you begin, ensure you have met the following requirements:

- [CUDA](https://developer.nvidia.com/cuda-downloads) is installed on your system.
- [Cutlass](https://github.com/NVIDIA/cutlass) is cloned and placed in the main directory of the project.

Please follow the instructions provided in the respective links to install CUDA and clone the Cutlass repository.

## Installation

To get the project up and running, follow these steps:

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/your-username/your-project.git
   cd CudaGeMM
   mkdir build && cd build
   cmake ..
   make
      ```

## Code Structure

The project is organized as follows : 
   ```bash

CudaGeMM
├── batched_gemm_benchmark.cu
│   - Executes batched gemm kernels on various matrix sizes for benchmarking.
├── CMakeLists.txt
│   - Configuration file for CMake, used for building the project.
├── gemm_benchmark.cu
│   - Executes gemm kernels on different matrix sizes for benchmarking.
├── include
│   ├── kernels_calls.h
│   │   - Header file containing template functions for calling each kernel.
│   ├── matrix.h
│   │   - Header file for matrix operations like generating random matrices and comparing matrices.
│   ├── kernels.h
│   │   - Header file containing gemm implementations.
│   └── wrapper.h
│       - Header file for wrapping the cublasSgemm and cublasDgemm functions as a template.
└── src
    ├── kernels_calls.cu
    │   - Source file implementing the template functions for calling each kernel.
    ├── matrix.cpp
    │   - Source file containing the implementation for matrix operations.
    ├── kernels.cu
    │   - Source file containing the implementation of gemm kernels.
    └── wrapper.cu
        - Source file implementing the wrapper for cublasSgemm and cublasDgemm as a template.

