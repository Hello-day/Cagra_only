#ifndef DEFS_H_
#define DEFS_H_

#include <cstdint>
#include <vector>

// --- 基础配置 ---
#define DIM 128
#define PQ_M 32
#define PQ_K 256
#define PQ_SUB_DIM (DIM / PQ_M)

// --- 默认参数 ---
#define K_CLUSTERS 2048      // IVF 聚类数
#define DEFAULT_BATCH_SIZE 128  // 从64增加到128，分摊CAGRA调用开销
#define DEFAULT_TOP_M 200    // nprobe=200, 搜索 78% 的簇

// --- GPU 排序与回传配置 ---
#define RERANK_M 2000        // 回传 Top-3000 给 CPU 精排

using pq_code_t = uint8_t;

template<typename T>
struct point_t {
    T coordinates[DIM];
};

// --- GPU 索引结构 (只读) ---
struct IVFIndexGPU {
    float* d_pq_codebook;       // [PQ_M, PQ_K, PQ_SUB_DIM]
    
    // IVF 倒排表 (CSR)
    int* d_cluster_offsets;     // [K_CLUSTERS + 1]
    uint8_t* d_all_pq_codes;    // [Total_Vectors, PQ_M]
    int* d_all_vector_ids;      // [Total_Vectors]
};

#endif // DEFS_H_