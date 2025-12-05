#include "defs.h"
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <cub/cub.cuh>
#include <cstdio>

// 阶段计时结构
struct GPUStageTimings {
    float precompute_ms = 0;
    float count_ms = 0;
    float scan_offset_ms = 0;
    float resize_sync_ms = 0;  // 同步等待时间 (瓶颈!)
    float scan_candidates_ms = 0;
    float sort_ms = 0;
    float gather_ms = 0;
    float total_ms = 0;
    size_t total_candidates = 0;
};

// 全局计时统计 (累计)
static GPUStageTimings g_timing_accum;
static int g_timing_count = 0;
static bool g_enable_profiling = false;

void enable_gpu_profiling(bool enable) { g_enable_profiling = enable; }

void reset_gpu_timings() {
    g_timing_accum = GPUStageTimings();
    g_timing_count = 0;
}

GPUStageTimings get_avg_gpu_timings() {
    GPUStageTimings avg = g_timing_accum;
    if (g_timing_count > 0) {
        avg.precompute_ms /= g_timing_count;
        avg.count_ms /= g_timing_count;
        avg.scan_offset_ms /= g_timing_count;
        avg.resize_sync_ms /= g_timing_count;
        avg.scan_candidates_ms /= g_timing_count;
        avg.sort_ms /= g_timing_count;
        avg.gather_ms /= g_timing_count;
        avg.total_ms /= g_timing_count;
        avg.total_candidates /= g_timing_count;
    }
    return avg;
}

// 1. 预计算 (不变)
__global__ void batch_global_precompute_kernel(
    const float* __restrict__ queries, const float* __restrict__ codebook, float* __restrict__ out_tables) {
    int q_idx = blockIdx.x; int m_idx = blockIdx.y; int k_idx = threadIdx.x;
    if (k_idx >= PQ_K) return;
    const float* q_sub = queries + q_idx * DIM + m_idx * PQ_SUB_DIM;
    const float* c_sub = codebook + (m_idx * PQ_K + k_idx) * PQ_SUB_DIM;
    float dist = 0.0f;
    for (int d = 0; d < PQ_SUB_DIM; ++d) {
        float diff = q_sub[d] - c_sub[d]; dist += diff * diff;
    }
    out_tables[q_idx * (PQ_M * PQ_K) + m_idx * PQ_K + k_idx] = dist;
}

// 2. Count (int* -> uint32_t*)
__global__ void ivf_count_kernel(
    const uint32_t* top_m_ids, // <--- Fix type
    const int* cluster_offsets, 
    int* out_counts, 
    int top_m
) {
    int q_idx = blockIdx.x;
    // 每个线程处理多个簇，支持 top_m > blockDim.x
    for (int m_idx = threadIdx.x; m_idx < top_m; m_idx += blockDim.x) {
        int c_id = (int)top_m_ids[q_idx * top_m + m_idx];
        atomicAdd(&out_counts[q_idx], cluster_offsets[c_id + 1] - cluster_offsets[c_id]);
    }
}

// 3. Scan (int* -> uint32_t*)
__global__ void ivf_scan_kernel(
    const int* cluster_offsets, 
    const uint32_t* top_m_ids, // <--- Fix type
    const uint8_t* all_pq_codes, const int* all_vec_ids,
    const float* global_tables, const int* query_base_offsets, int* query_atomic_counters,
    int* out_ids, float* out_dists, int top_m) 
{
    int q_idx = blockIdx.x; int m_idx = blockIdx.y;
    int c_id = (int)top_m_ids[q_idx * gridDim.y + m_idx];
    int start = cluster_offsets[c_id]; int len = cluster_offsets[c_id + 1] - start;
    if (len == 0) return;

    __shared__ int write_start;
    if (threadIdx.x == 0) write_start = atomicAdd(&query_atomic_counters[q_idx], len);
    __syncthreads();

    int global_base = query_base_offsets[q_idx] + write_start;
    const float* my_table = global_tables + q_idx * (PQ_M * PQ_K);

    for (int i = threadIdx.x; i < len; i += blockDim.x) {
        int ivf_idx = start + i;
        float dist = 0.0f;
        #pragma unroll
        for (int m = 0; m < PQ_M; ++m) {
            dist += my_table[m * PQ_K + all_pq_codes[(size_t)ivf_idx * PQ_M + m]];
        }
        out_ids[global_base + i] = all_vec_ids[ivf_idx];
        out_dists[global_base + i] = dist;
    }
}

// 4. Gather (不变)
__global__ void gather_top_m_kernel(
    const int* __restrict__ sorted_ids, const float* __restrict__ sorted_dists,
    const int* __restrict__ offsets, const int* __restrict__ counts,
    int* __restrict__ out_top_ids, float* __restrict__ out_top_dists, int* __restrict__ out_real_counts, int M
) {
    int q_idx = blockIdx.x; int lane = threadIdx.x;
    if (lane >= M) return;
    int count = counts[q_idx];
    int actual_k = (count < M) ? count : M;
    if (lane == 0) out_real_counts[q_idx] = actual_k;
    if (lane < actual_k) {
        int src_idx = offsets[q_idx] + lane;
        int dst_idx = q_idx * M + lane;
        out_top_ids[dst_idx]   = sorted_ids[src_idx];
        out_top_dists[dst_idx] = sorted_dists[src_idx];
    }
}

// Host Wrapper (int* -> uint32_t*)
void run_gpu_batch_logic(
    const IVFIndexGPU& idx,
    float* d_queries, 
    uint32_t* d_cagra_top_m, // <--- Fix type
    float* d_global_tables,
    int*& d_flat_ids, float*& d_flat_dists, 
    int*& d_flat_ids_alt, float*& d_flat_dists_alt, 
    size_t& current_pool_cap,
    int* d_counts, int* d_offsets, int* d_atomic,
    int* d_top_ids, float* d_top_dists, int* d_top_counts,
    int batch_size, int top_m,
    cudaStream_t stream_main
) {
    static cudaStream_t stream_table = nullptr;
    static cudaEvent_t evt_table_done = nullptr;
    // Profiling events (静态分配避免重复创建)
    static cudaEvent_t evt_start, evt_precompute, evt_count, evt_scan_offset;
    static cudaEvent_t evt_resize_sync, evt_scan_cand, evt_sort, evt_gather;
    static bool events_created = false;
    
    if (!stream_table) {
        cudaStreamCreateWithFlags(&stream_table, cudaStreamNonBlocking);
        cudaEventCreate(&evt_table_done);
    }
    
    if (!events_created && g_enable_profiling) {
        cudaEventCreate(&evt_start);
        cudaEventCreate(&evt_precompute);
        cudaEventCreate(&evt_count);
        cudaEventCreate(&evt_scan_offset);
        cudaEventCreate(&evt_resize_sync);
        cudaEventCreate(&evt_scan_cand);
        cudaEventCreate(&evt_sort);
        cudaEventCreate(&evt_gather);
        events_created = true;
    }
    
    if (g_enable_profiling) {
        cudaEventRecord(evt_start, stream_main);
    }

    // 1. Precompute (在 stream_table 上并行执行)
    batch_global_precompute_kernel<<<dim3(batch_size, PQ_M), 256, 0, stream_table>>>(
        d_queries, idx.d_pq_codebook, d_global_tables
    );
    cudaEventRecord(evt_table_done, stream_table);
    
    if (g_enable_profiling) {
        cudaStreamSynchronize(stream_table);
        cudaEventRecord(evt_precompute, stream_main);
    }

    // 2. Count
    cudaMemsetAsync(d_counts, 0, batch_size * sizeof(int), stream_main);
    ivf_count_kernel<<<batch_size, 128, 0, stream_main>>>(
        d_cagra_top_m, idx.d_cluster_offsets, d_counts, top_m
    );
    
    if (g_enable_profiling) {
        cudaEventRecord(evt_count, stream_main);
    }

    // 3. Scan Offsets
    auto policy = thrust::cuda::par.on(stream_main);
    thrust::exclusive_scan(policy, 
        thrust::device_ptr<int>(d_counts), 
        thrust::device_ptr<int>(d_counts + batch_size), 
        thrust::device_ptr<int>(d_offsets));
    
    if (g_enable_profiling) {
        cudaEventRecord(evt_scan_offset, stream_main);
    }

    // 4. Resize Check (⚠️ 这里有同步，可能是瓶颈!)
    int last_count, last_offset;
    cudaMemcpyAsync(&last_count, d_counts + batch_size - 1, sizeof(int), cudaMemcpyDeviceToHost, stream_main);
    cudaMemcpyAsync(&last_offset, d_offsets + batch_size - 1, sizeof(int), cudaMemcpyDeviceToHost, stream_main);
    cudaStreamSynchronize(stream_main);  // ⚠️ 阻塞点!

    size_t needed = last_offset + last_count;
    if (needed > current_pool_cap) {
        size_t new_cap = needed * 1.5;
        cudaFree(d_flat_ids); cudaFree(d_flat_dists);
        cudaFree(d_flat_ids_alt); cudaFree(d_flat_dists_alt);
        cudaMalloc(&d_flat_ids, new_cap * sizeof(int));
        cudaMalloc(&d_flat_dists, new_cap * sizeof(float));
        cudaMalloc(&d_flat_ids_alt, new_cap * sizeof(int));
        cudaMalloc(&d_flat_dists_alt, new_cap * sizeof(float));
        current_pool_cap = new_cap;
        printf("[WARN] GPU pool resized to %zu elements\n", new_cap);
    }
    
    if (g_enable_profiling) {
        cudaEventRecord(evt_resize_sync, stream_main);
    }

    // 5. Scan Candidates
    cudaStreamWaitEvent(stream_main, evt_table_done, 0);
    cudaMemsetAsync(d_atomic, 0, batch_size * sizeof(int), stream_main);
    ivf_scan_kernel<<<dim3(batch_size, top_m), 128, 0, stream_main>>>(
        idx.d_cluster_offsets, d_cagra_top_m, idx.d_all_pq_codes, idx.d_all_vector_ids,
        d_global_tables, d_offsets, d_atomic, d_flat_ids, d_flat_dists, top_m
    );
    
    if (g_enable_profiling) {
        cudaEventRecord(evt_scan_cand, stream_main);
    }

    // 6. Sort (⚠️ 每次都 malloc/free 临时空间，可能是瓶颈!)
    cub::DoubleBuffer<float> d_keys(d_flat_dists, d_flat_dists_alt);
    cub::DoubleBuffer<int>   d_values(d_flat_ids, d_flat_ids_alt);
    
    void *d_temp = NULL; size_t temp_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairs(d_temp, temp_bytes, d_keys, d_values,
        needed, batch_size, d_offsets, d_offsets + 1, 0, sizeof(float)*8, stream_main);
    
    cudaMalloc(&d_temp, temp_bytes);
    cub::DeviceSegmentedRadixSort::SortPairs(d_temp, temp_bytes, d_keys, d_values,
        needed, batch_size, d_offsets, d_offsets + 1, 0, sizeof(float)*8, stream_main);
    cudaFree(d_temp);
    
    if (g_enable_profiling) {
        cudaEventRecord(evt_sort, stream_main);
    }

    // 7. Gather Top M
    gather_top_m_kernel<<<batch_size, 256, 0, stream_main>>>(
        d_values.Current(), d_keys.Current(), d_offsets, d_counts,
        d_top_ids, d_top_dists, d_top_counts, RERANK_M
    );
    
    if (g_enable_profiling) {
        cudaEventRecord(evt_gather, stream_main);
        cudaStreamSynchronize(stream_main);
        
        // 收集计时数据
        GPUStageTimings t;
        cudaEventElapsedTime(&t.precompute_ms, evt_start, evt_precompute);
        cudaEventElapsedTime(&t.count_ms, evt_precompute, evt_count);
        cudaEventElapsedTime(&t.scan_offset_ms, evt_count, evt_scan_offset);
        cudaEventElapsedTime(&t.resize_sync_ms, evt_scan_offset, evt_resize_sync);
        cudaEventElapsedTime(&t.scan_candidates_ms, evt_resize_sync, evt_scan_cand);
        cudaEventElapsedTime(&t.sort_ms, evt_scan_cand, evt_sort);
        cudaEventElapsedTime(&t.gather_ms, evt_sort, evt_gather);
        cudaEventElapsedTime(&t.total_ms, evt_start, evt_gather);
        t.total_candidates = needed;
        
        // 累计统计
        g_timing_accum.precompute_ms += t.precompute_ms;
        g_timing_accum.count_ms += t.count_ms;
        g_timing_accum.scan_offset_ms += t.scan_offset_ms;
        g_timing_accum.resize_sync_ms += t.resize_sync_ms;
        g_timing_accum.scan_candidates_ms += t.scan_candidates_ms;
        g_timing_accum.sort_ms += t.sort_ms;
        g_timing_accum.gather_ms += t.gather_ms;
        g_timing_accum.total_ms += t.total_ms;
        g_timing_accum.total_candidates += t.total_candidates;
        g_timing_count++;
    }
}