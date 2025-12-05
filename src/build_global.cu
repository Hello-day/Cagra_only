#include "defs.h"
#include "utils.h"
#include "pq_utils.h"
#include "kmeans_gpu.cuh"
#include "cagra_adapter.cuh" // 使用 cuVS CAGRA 构建索引
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <cuda_runtime.h>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: ./build_global <base.fvecs>" << std::endl;
        std::cerr << "  This version builds a pure CAGRA index over ALL base vectors." << std::endl;
        return 1;
    }

    // 1. 加载全部 base 向量（CPU 上）
    float* data_ptr = nullptr;
    int n = read_vecs<float>(argv[1], data_ptr, DIM);
    if (n <= 0 || data_ptr == nullptr) {
        std::cerr << "Failed to load base vectors from " << argv[1] << std::endl;
        return 1;
    }
    std::cout << "Loaded " << n << " base vectors (dim=" << DIM << ")." << std::endl;

    // 2. 把 base 向量拷贝到 GPU
    float* d_base = nullptr;
    size_t bytes = static_cast<size_t>(n) * DIM * sizeof(float);
    cudaError_t st = cudaMalloc(&d_base, bytes);
    if (st != cudaSuccess) {
        std::cerr << "cudaMalloc failed for base vectors: " << cudaGetErrorString(st) << std::endl;
        delete[] data_ptr;
        return 1;
    }
    st = cudaMemcpy(d_base, data_ptr, bytes, cudaMemcpyHostToDevice);
    if (st != cudaSuccess) {
        std::cerr << "cudaMemcpy failed for base vectors: " << cudaGetErrorString(st) << std::endl;
        cudaFree(d_base);
        delete[] data_ptr;
        return 1;
    }

    // 3. 用「全部 base 向量」构建 CAGRA 索引
    std::cout << "Building CAGRA index over ALL base vectors..." << std::endl;
    const char* index_path = "../res/cagra_base.index";
    build_cagra_index(d_base, n, DIM, index_path);
    std::cout << "CAGRA index saved to: " << index_path << std::endl;

    // 4. 清理
    cudaFree(d_base);
    delete[] data_ptr;

    std::cout << "Build complete (pure CAGRA index)." << std::endl;
    return 0;
}