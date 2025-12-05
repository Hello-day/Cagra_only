#include "defs.h"
#include "utils.h"
#include "cagra_adapter.cuh"

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <unordered_set>

// 计算召回率（从 search_pipeline.cu 简化拷贝）
float compute_recall(const std::vector<std::vector<int>>& results,
                     const int* groundtruth,
                     int nq,
                     int gt_k,
                     int recall_k) {
    int total_hits = 0;
    for (int i = 0; i < nq; ++i) {
        std::unordered_set<int> gt_set;
        for (int j = 0; j < gt_k; ++j) {
            gt_set.insert(groundtruth[i * gt_k + j]);
        }

        int hits = 0;
        int check_k = std::min(recall_k, (int)results[i].size());
        for (int j = 0; j < check_k; ++j) {
            if (gt_set.count(results[i][j])) {
                hits++;
            }
        }
        total_hits += hits;
    }
    return (float)total_hits / (nq * recall_k);
}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: ./search_cagra_only <base.fvecs> <query.fvecs> [groundtruth.ivecs]" << std::endl;
        return 1;
    }

    std::string base = argv[1];
    std::string query = argv[2];
    std::string gt_file = (argc == 4) ? argv[3] : "";
    std::string res_dir = "../res";

    const int TOP_K = 10;  // 最终返回的 Top-K

    std::cout << "============ CAGRA-Only Search Benchmark ============" << std::endl;
    std::cout << "[Config]" << std::endl;
    std::cout << "  Vector dimension: " << DIM << std::endl;
    std::cout << "  Top-K:            " << TOP_K << std::endl;
    std::cout << "  Batch size:       " << DEFAULT_BATCH_SIZE << std::endl;
    std::cout << std::endl;

    // 1. 加载 base / query / groundtruth
    auto t_load_start = std::chrono::high_resolution_clock::now();

    float* base_ptr = nullptr;
    int n = read_vecs<float>(base, base_ptr, DIM);
    if (n <= 0 || base_ptr == nullptr) {
        std::cerr << "Failed to load base vectors from " << base << std::endl;
        return 1;
    }
    std::cout << "  Base vectors:  " << n << std::endl;

    float* q_ptr = nullptr;
    int nq = read_vecs<float>(query, q_ptr, DIM);
    if (nq <= 0 || q_ptr == nullptr) {
        std::cerr << "Failed to load query vectors from " << query << std::endl;
        delete[] base_ptr;
        return 1;
    }
    std::cout << "  Query vectors: " << nq << std::endl;

    int* gt_data = nullptr;
    int gt_k = 0;
    if (!gt_file.empty()) {
        int gt_nq = read_ivecs(gt_file, gt_data, gt_k);
        if (gt_nq != nq) {
            std::cerr << "Warning: groundtruth size mismatch (" << gt_nq << " vs " << nq << ")" << std::endl;
        }
        std::cout << "  Ground truth loaded (k=" << gt_k << ")" << std::endl;
    }

    // 加载 CAGRA 索引（对全量 base 向量）
    raft::device_resources raft_handle;
    std::string cagra_path = res_dir + "/cagra_base.index";
    auto cagra_index = load_cagra_index(raft_handle, cagra_path.c_str());
    std::cout << "  CAGRA index loaded from " << cagra_path << std::endl;

    auto t_load_end = std::chrono::high_resolution_clock::now();
    double load_time_ms =
        std::chrono::duration<double, std::milli>(t_load_end - t_load_start).count();
    std::cout << "  Load time: " << load_time_ms << " ms" << std::endl;
    std::cout << std::endl;

    // 2. 准备 GPU 内存（queries / indices / dists）
    float* d_queries = nullptr;
    uint32_t* d_indices = nullptr;
    float* d_dists = nullptr;

    int batch_size = DEFAULT_BATCH_SIZE;
    size_t max_batch_bytes = static_cast<size_t>(batch_size) * DIM * sizeof(float);
    cudaMalloc(&d_queries, max_batch_bytes);
    cudaMalloc(&d_indices, static_cast<size_t>(batch_size) * TOP_K * sizeof(uint32_t));
    cudaMalloc(&d_dists, static_cast<size_t>(batch_size) * TOP_K * sizeof(float));

    std::vector<std::vector<int>> results(nq);

    // 3. Benchmark：分批调用 CAGRA 搜索
    std::cout << "[Benchmark]" << std::endl;
    auto t_start = std::chrono::high_resolution_clock::now();

    int batches = (nq + batch_size - 1) / batch_size;
    for (int b = 0; b < batches; ++b) {
        int start = b * batch_size;
        int cur_n = std::min(batch_size, nq - start);
        size_t cur_bytes = static_cast<size_t>(cur_n) * DIM * sizeof(float);

        // H2D
        cudaMemcpy(d_queries,
                   q_ptr + static_cast<size_t>(start) * DIM,
                   cur_bytes,
                   cudaMemcpyHostToDevice);

        // CAGRA 搜索
        search_cagra_index(raft_handle,
                           cagra_index,
                           d_queries,
                           cur_n,
                           DIM,
                           TOP_K,
                           d_indices,
                           d_dists);

        // D2H
        std::vector<uint32_t> h_indices(static_cast<size_t>(cur_n) * TOP_K);
        cudaMemcpy(h_indices.data(),
                   d_indices,
                   static_cast<size_t>(cur_n) * TOP_K * sizeof(uint32_t),
                   cudaMemcpyDeviceToHost);

        // 保存结果
        for (int i = 0; i < cur_n; ++i) {
            int qid = start + i;
            results[qid].clear();
            for (int k = 0; k < TOP_K; ++k) {
                results[qid].push_back(static_cast<int>(h_indices[i * TOP_K + k]));
            }
        }
    }

    cudaDeviceSynchronize();
    auto t_end = std::chrono::high_resolution_clock::now();

    double elapsed_ms =
        std::chrono::duration<double, std::milli>(t_end - t_start).count();
    double qps = nq / (elapsed_ms / 1000.0);
    double latency_per_query_us = (elapsed_ms * 1000.0) / nq;

    std::cout << std::endl;
    std::cout << "=============== Results ===============" << std::endl;
    std::cout << "  Total queries:      " << nq << std::endl;
    std::cout << "  Total batches:      " << batches << std::endl;
    std::cout << "  Total time:         " << std::fixed << std::setprecision(3)
              << elapsed_ms << " ms" << std::endl;
    std::cout << "  Throughput (QPS):   " << std::fixed << std::setprecision(2)
              << qps << std::endl;
    std::cout << "  Avg latency/query:  " << std::fixed << std::setprecision(2)
              << latency_per_query_us << " us" << std::endl;

    // 4. 计算召回率（如果有 groundtruth）
    if (gt_data != nullptr) {
        float recall_1 = compute_recall(results, gt_data, nq, gt_k, 1);
        float recall_10 = compute_recall(results, gt_data, nq, gt_k, 10);
        std::cout << std::endl;
        std::cout << "  Recall@1:           " << std::fixed << std::setprecision(4)
                  << (recall_1 * 100) << "%" << std::endl;
        std::cout << "  Recall@10:          " << std::fixed << std::setprecision(4)
                  << (recall_10 * 100) << "%" << std::endl;
    }

    std::cout << "=======================================" << std::endl;

    // 5. 清理资源
    cudaFree(d_queries);
    cudaFree(d_indices);
    cudaFree(d_dists);
    delete[] base_ptr;
    delete[] q_ptr;
    if (gt_data) {
        delete[] gt_data;
    }

    // 使用 quick_exit 避免 cuVS/rmm 析构时的潜在问题（与 search_pipeline 保持一致）
    std::quick_exit(0);
}



