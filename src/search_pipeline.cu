#include "defs.h"
#include "utils.h"
#include "cagra_adapter.cuh"
#include <cuda_runtime.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <atomic>
#include <memory>
#include <optional>
#include <unordered_set>

// [修复] 外部函数声明: dc (cagra result) 改为 uint32_t*
extern void run_gpu_batch_logic(
    const IVFIndexGPU& idx, float* dq, uint32_t* dc, float* dt, 
    int*& did, float*& ddist, 
    int*& did_alt, float*& ddist_alt, 
    size_t& cap,
    int* cnt, int* off, int* atm,
    int* top_id, float* top_dist, int* top_cnt,
    int bs, int tm, cudaStream_t s
);

// GPU 阶段计时相关函数
struct GPUStageTimings {
    float precompute_ms = 0;
    float count_ms = 0;
    float scan_offset_ms = 0;
    float resize_sync_ms = 0;
    float scan_candidates_ms = 0;
    float sort_ms = 0;
    float gather_ms = 0;
    float total_ms = 0;
    size_t total_candidates = 0;
};
extern void enable_gpu_profiling(bool enable);
extern void reset_gpu_timings();
extern GPUStageTimings get_avg_gpu_timings();

// 流水线阶段计时结构
struct PipelineTimings {
    std::atomic<double> h2d_ms{0};
    std::atomic<double> cagra_ms{0};
    std::atomic<double> gpu_batch_ms{0};
    std::atomic<double> d2h_ms{0};
    std::atomic<double> cpu_rerank_ms{0};
    std::atomic<int> batch_count{0};
};

class SearchPipeline {
    struct BatchCtx {
        int id;
        cudaStream_t stream;
        cudaEvent_t cpu_wait_evt;
        
        // 计时事件（每个buffer独立）
        cudaEvent_t evt_h2d_start, evt_h2d_end;
        cudaEvent_t evt_cagra_start, evt_cagra_end;
        cudaEvent_t evt_gpu_batch_start, evt_gpu_batch_end;
        cudaEvent_t evt_d2h_start, evt_d2h_end;

        // Fixed GPU
        float* d_queries = nullptr;       
        uint32_t* d_cagra_res = nullptr;
        float* d_cagra_dists = nullptr;   
        float* d_global_tables = nullptr; 
        int* d_counts = nullptr;          
        int* d_offsets = nullptr;         
        int* d_atomic = nullptr;          

        // Dynamic GPU Pool
        int* d_flat_ids = nullptr;        
        float* d_flat_dists = nullptr;
        int* d_flat_ids_alt = nullptr;    
        float* d_flat_dists_alt = nullptr;
        size_t gpu_pool_cap = 0;    

        // Output (Top M)
        int *d_top_ids = nullptr, *d_top_counts = nullptr;
        float *d_top_dists = nullptr;

        // Host
        int *h_top_ids = nullptr, *h_top_counts = nullptr;
        float *h_top_dists = nullptr;
        
        std::vector<float> raw_queries; 
        int current_batch_size = 0;
    };

    int batch_size;
    BatchCtx* buffers[2];
    IVFIndexGPU ivf_index;
    const point_t<float>* raw_vecs_cpu;
    int total_base_vecs;  // 用于边界检查
    
    raft::device_resources raft_handle;
    std::optional<cuvs::neighbors::cagra::index<float, uint32_t>> cagra_idx_opt;

    std::thread worker_thread;
    std::queue<BatchCtx*> work_queue;
    std::mutex mtx;
    std::condition_variable cv;
    bool running = true;
    
    // 性能统计
    std::atomic<int> total_reranked{0};
    std::atomic<double> total_gpu_time_ms{0};
    std::atomic<double> total_cpu_time_ms{0};
    
    // 流水线阶段计时
    PipelineTimings pipeline_timings;
    
    // 存储最终结果
    std::vector<std::vector<int>> final_results;

public:
    SearchPipeline(int b_size, IVFIndexGPU index, const point_t<float>* data, const char* cagra_path, int total_queries, int total_base) 
        : batch_size(b_size), ivf_index(index), raw_vecs_cpu(data), total_base_vecs(total_base)
    {
        cagra_idx_opt.emplace(load_cagra_index(raft_handle, cagra_path));
        final_results.resize(total_queries);
        size_t init_cap = (size_t)batch_size * 2000;
        if (init_cap < 1024) init_cap = 1024;

        for(int i=0; i<2; ++i) {
            buffers[i] = new BatchCtx();
            buffers[i]->id = i;
            cudaStreamCreate(&buffers[i]->stream);
            cudaEventCreate(&buffers[i]->cpu_wait_evt);
            
            // 创建计时事件
            cudaEventCreate(&buffers[i]->evt_h2d_start);
            cudaEventCreate(&buffers[i]->evt_h2d_end);
            cudaEventCreate(&buffers[i]->evt_cagra_start);
            cudaEventCreate(&buffers[i]->evt_cagra_end);
            cudaEventCreate(&buffers[i]->evt_gpu_batch_start);
            cudaEventCreate(&buffers[i]->evt_gpu_batch_end);
            cudaEventCreate(&buffers[i]->evt_d2h_start);
            cudaEventCreate(&buffers[i]->evt_d2h_end);
            
            cudaMalloc(&buffers[i]->d_queries, batch_size * DIM * sizeof(float));
            // [修复] sizeof(uint32_t)
            cudaMalloc(&buffers[i]->d_cagra_res, batch_size * DEFAULT_TOP_M * sizeof(uint32_t));
            cudaMalloc(&buffers[i]->d_cagra_dists, batch_size * DEFAULT_TOP_M * sizeof(float));
            cudaMalloc(&buffers[i]->d_global_tables, batch_size * PQ_M * PQ_K * sizeof(float));
            cudaMalloc(&buffers[i]->d_counts, batch_size * sizeof(int));
            cudaMalloc(&buffers[i]->d_offsets, batch_size * sizeof(int));
            cudaMalloc(&buffers[i]->d_atomic, batch_size * sizeof(int));
            
            buffers[i]->gpu_pool_cap = init_cap;
            cudaMalloc(&buffers[i]->d_flat_ids, init_cap * sizeof(int));
            cudaMalloc(&buffers[i]->d_flat_dists, init_cap * sizeof(float));
            cudaMalloc(&buffers[i]->d_flat_ids_alt, init_cap * sizeof(int));
            cudaMalloc(&buffers[i]->d_flat_dists_alt, init_cap * sizeof(float));

            cudaMalloc(&buffers[i]->d_top_ids, batch_size * RERANK_M * sizeof(int));
            cudaMalloc(&buffers[i]->d_top_dists, batch_size * RERANK_M * sizeof(float));
            cudaMalloc(&buffers[i]->d_top_counts, batch_size * sizeof(int));

            cudaMallocHost(&buffers[i]->h_top_ids, batch_size * RERANK_M * sizeof(int));
            cudaMallocHost(&buffers[i]->h_top_dists, batch_size * RERANK_M * sizeof(float));
            cudaMallocHost(&buffers[i]->h_top_counts, batch_size * sizeof(int));
        }
        worker_thread = std::thread(&SearchPipeline::cpu_loop, this);
    }

    ~SearchPipeline() {
        // 先停止工作线程
        { std::lock_guard<std::mutex> lk(mtx); running = false; }
        cv.notify_all();
        if(worker_thread.joinable()) worker_thread.join();
        
        // 同步 raft handle 中的流
        raft::resource::sync_stream(raft_handle);
        
        // 释放 buffer 资源
        for(int i=0; i<2; ++i) {
            if (!buffers[i]) continue;
            
            // 先同步并销毁 stream
            if (buffers[i]->stream) {
                cudaStreamSynchronize(buffers[i]->stream);
                cudaStreamDestroy(buffers[i]->stream);
            }
            if (buffers[i]->cpu_wait_evt) cudaEventDestroy(buffers[i]->cpu_wait_evt);
            
            // 销毁计时事件
            cudaEventDestroy(buffers[i]->evt_h2d_start);
            cudaEventDestroy(buffers[i]->evt_h2d_end);
            cudaEventDestroy(buffers[i]->evt_cagra_start);
            cudaEventDestroy(buffers[i]->evt_cagra_end);
            cudaEventDestroy(buffers[i]->evt_gpu_batch_start);
            cudaEventDestroy(buffers[i]->evt_gpu_batch_end);
            cudaEventDestroy(buffers[i]->evt_d2h_start);
            cudaEventDestroy(buffers[i]->evt_d2h_end);
            
            // 再释放 GPU 内存
            if (buffers[i]->d_queries) cudaFree(buffers[i]->d_queries);
            if (buffers[i]->d_cagra_res) cudaFree(buffers[i]->d_cagra_res);
            if (buffers[i]->d_cagra_dists) cudaFree(buffers[i]->d_cagra_dists);
            if (buffers[i]->d_global_tables) cudaFree(buffers[i]->d_global_tables);
            if (buffers[i]->d_counts) cudaFree(buffers[i]->d_counts);
            if (buffers[i]->d_offsets) cudaFree(buffers[i]->d_offsets);
            if (buffers[i]->d_atomic) cudaFree(buffers[i]->d_atomic);
            
            if (buffers[i]->d_flat_ids) cudaFree(buffers[i]->d_flat_ids);
            if (buffers[i]->d_flat_dists) cudaFree(buffers[i]->d_flat_dists);
            if (buffers[i]->d_flat_ids_alt) cudaFree(buffers[i]->d_flat_ids_alt);
            if (buffers[i]->d_flat_dists_alt) cudaFree(buffers[i]->d_flat_dists_alt);

            if (buffers[i]->d_top_ids) cudaFree(buffers[i]->d_top_ids);
            if (buffers[i]->d_top_dists) cudaFree(buffers[i]->d_top_dists);
            if (buffers[i]->d_top_counts) cudaFree(buffers[i]->d_top_counts);

            if (buffers[i]->h_top_ids) cudaFreeHost(buffers[i]->h_top_ids);
            if (buffers[i]->h_top_dists) cudaFreeHost(buffers[i]->h_top_dists);
            if (buffers[i]->h_top_counts) cudaFreeHost(buffers[i]->h_top_counts);
            
            delete buffers[i];
        }
        
        // 手动释放 cagra 索引（在 raft_handle 析构前）
        cagra_idx_opt.reset();
    }
    
    void print_stats(int nq) {
        int avg_candidates = total_reranked.load() / nq;
        std::cout << "  Avg candidates per query for rerank: " << avg_candidates << std::endl;
    }
    
    void print_timing_stats() {
        int batch_cnt = pipeline_timings.batch_count.load();
        if (batch_cnt == 0) return;
        
        double avg_h2d = pipeline_timings.h2d_ms.load() / batch_cnt;
        double avg_cagra = pipeline_timings.cagra_ms.load() / batch_cnt;
        double avg_gpu_batch = pipeline_timings.gpu_batch_ms.load() / batch_cnt;
        double avg_d2h = pipeline_timings.d2h_ms.load() / batch_cnt;
        double avg_cpu_rerank = pipeline_timings.cpu_rerank_ms.load() / batch_cnt;
        double total_per_batch = avg_h2d + avg_cagra + avg_gpu_batch + avg_d2h + avg_cpu_rerank;
        
        std::cout << std::endl;
        std::cout << "============= Pipeline Timing Breakdown =============" << std::endl;
        std::cout << "  (Average per batch, " << batch_cnt << " batches)" << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        
        if (total_per_batch > 0.001) {  // 避免除零
            std::cout << "  H2D (Host->Device):        " << std::setw(8) << avg_h2d << " ms (" 
                      << std::setprecision(1) << (avg_h2d / total_per_batch * 100) << "%)" << std::endl;
            std::cout << std::setprecision(3);
            std::cout << "  CAGRA Search:              " << std::setw(8) << avg_cagra << " ms (" 
                      << std::setprecision(1) << (avg_cagra / total_per_batch * 100) << "%)" << std::endl;
            std::cout << std::setprecision(3);
            std::cout << "  GPU Batch Logic:           " << std::setw(8) << avg_gpu_batch << " ms (" 
                      << std::setprecision(1) << (avg_gpu_batch / total_per_batch * 100) << "%)" << std::endl;
            std::cout << std::setprecision(3);
            std::cout << "  D2H (Device->Host):        " << std::setw(8) << avg_d2h << " ms (" 
                      << std::setprecision(1) << (avg_d2h / total_per_batch * 100) << "%)" << std::endl;
            std::cout << std::setprecision(3);
            std::cout << "  CPU Rerank:                " << std::setw(8) << avg_cpu_rerank << " ms (" 
                      << std::setprecision(1) << (avg_cpu_rerank / total_per_batch * 100) << "%)" << std::endl;
        } else {
            std::cout << "  H2D (Host->Device):        " << std::setw(8) << avg_h2d << " ms" << std::endl;
            std::cout << "  CAGRA Search:              " << std::setw(8) << avg_cagra << " ms" << std::endl;
            std::cout << "  GPU Batch Logic:           " << std::setw(8) << avg_gpu_batch << " ms" << std::endl;
            std::cout << "  D2H (Device->Host):        " << std::setw(8) << avg_d2h << " ms" << std::endl;
            std::cout << "  CPU Rerank:                " << std::setw(8) << avg_cpu_rerank << " ms" << std::endl;
        }
        std::cout << std::setprecision(3);
        std::cout << "  Total per batch:           " << std::setw(8) << total_per_batch << " ms" << std::endl;
        std::cout << "=====================================================" << std::endl;
    }
    
    const std::vector<std::vector<int>>& get_results() const {
        return final_results;
    }

    void submit_batch(const std::vector<float>& queries, int query_offset) {
        static int buf_idx = 0;
        BatchCtx* c = buffers[buf_idx];
        cudaStreamSynchronize(c->stream); 
        
        int current_n = queries.size() / DIM;
        c->current_batch_size = current_n;
        c->raw_queries = queries;
        c->id = query_offset; // 记录 batch 的起始查询 ID

        // 1. H2D
        cudaEventRecord(c->evt_h2d_start, c->stream);
        cudaMemcpyAsync(c->d_queries, queries.data(), queries.size() * sizeof(float), 
                        cudaMemcpyHostToDevice, c->stream);
        cudaEventRecord(c->evt_h2d_end, c->stream);

        // 2. CAGRA Search (Real call)
        cudaEventRecord(c->evt_cagra_start, c->stream);
        raft::resource::set_cuda_stream(raft_handle, c->stream);
        search_cagra_index(raft_handle, *cagra_idx_opt, c->d_queries, current_n, DIM, DEFAULT_TOP_M, c->d_cagra_res, c->d_cagra_dists);
        cudaEventRecord(c->evt_cagra_end, c->stream);

        // 3. Filter & Sort & Gather
        cudaEventRecord(c->evt_gpu_batch_start, c->stream);
        run_gpu_batch_logic(
            ivf_index, c->d_queries, c->d_cagra_res, c->d_global_tables,
            c->d_flat_ids, c->d_flat_dists, 
            c->d_flat_ids_alt, c->d_flat_dists_alt,
            c->gpu_pool_cap,
            c->d_counts, c->d_offsets, c->d_atomic,
            c->d_top_ids, c->d_top_dists, c->d_top_counts,
            current_n, DEFAULT_TOP_M, c->stream
        );
        cudaEventRecord(c->evt_gpu_batch_end, c->stream);

        // 4. D2H
        cudaEventRecord(c->evt_d2h_start, c->stream);
        cudaMemcpyAsync(c->h_top_counts, c->d_top_counts, current_n * sizeof(int), cudaMemcpyDeviceToHost, c->stream);
        cudaMemcpyAsync(c->h_top_ids, c->d_top_ids, current_n * RERANK_M * sizeof(int), cudaMemcpyDeviceToHost, c->stream);
        cudaMemcpyAsync(c->h_top_dists, c->d_top_dists, current_n * RERANK_M * sizeof(float), cudaMemcpyDeviceToHost, c->stream);
        cudaEventRecord(c->evt_d2h_end, c->stream);

        // 计时将在cpu_loop中异步获取，避免阻塞流水线

        cudaEventRecord(c->cpu_wait_evt, c->stream);
        {
            std::lock_guard<std::mutex> lk(mtx);
            work_queue.push(c);
        }
        cv.notify_one();
        buf_idx = 1 - buf_idx;
    }
    
    void wait_all() {
        // 等待所有任务处理完成
        while(true) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            std::lock_guard<std::mutex> lk(mtx);
            if (work_queue.empty()) break;
        }
        cudaDeviceSynchronize();
    }

    void cpu_loop() {
        while (true) {
            BatchCtx* c = nullptr;
            {
                std::unique_lock<std::mutex> lk(mtx);
                cv.wait(lk, [&]{ return !work_queue.empty() || !running; });
                if (!running && work_queue.empty()) return;
                c = work_queue.front();
                work_queue.pop();
            }
            
            cudaEventSynchronize(c->cpu_wait_evt);
            
            // 获取GPU阶段的计时（此时事件已完成）
            float h2d_ms, cagra_ms, gpu_batch_ms, d2h_ms;
            cudaEventElapsedTime(&h2d_ms, c->evt_h2d_start, c->evt_h2d_end);
            cudaEventElapsedTime(&cagra_ms, c->evt_cagra_start, c->evt_cagra_end);
            cudaEventElapsedTime(&gpu_batch_ms, c->evt_gpu_batch_start, c->evt_gpu_batch_end);
            cudaEventElapsedTime(&d2h_ms, c->evt_d2h_start, c->evt_d2h_end);
            
            // 使用原子操作更新GPU阶段计时统计
            double old_val;
            old_val = pipeline_timings.h2d_ms.load();
            pipeline_timings.h2d_ms.store(old_val + h2d_ms);
            old_val = pipeline_timings.cagra_ms.load();
            pipeline_timings.cagra_ms.store(old_val + cagra_ms);
            old_val = pipeline_timings.gpu_batch_ms.load();
            pipeline_timings.gpu_batch_ms.store(old_val + gpu_batch_ms);
            old_val = pipeline_timings.d2h_ms.load();
            pipeline_timings.d2h_ms.store(old_val + d2h_ms);
            pipeline_timings.batch_count++;
            
            // 计时CPU rerank阶段
            auto cpu_start = std::chrono::high_resolution_clock::now();
            
            int batch_total_candidates = 0;
            int query_offset = c->id;
            
            #pragma omp parallel for reduction(+:batch_total_candidates)
            for(int i=0; i < c->current_batch_size; ++i) {
                int count = c->h_top_counts[i];
                batch_total_candidates += count;
                int* ids_ptr = c->h_top_ids + i * RERANK_M;
                
                const float* query_vec = c->raw_queries.data() + i * DIM;
                const point_t<float>& q_point = *(const point_t<float>*)query_vec;

                std::vector<std::pair<float, int>> exact_candidates;
                exact_candidates.reserve(count);

                for(int j=0; j<count; ++j) {
                    int vec_id = ids_ptr[j];
                    // 边界检查，避免段错误
                    if (vec_id < 0 || vec_id >= total_base_vecs) {
                        continue;  // 跳过无效的ID
                    }
                    float d = dis(q_point, raw_vecs_cpu[vec_id]);
                    exact_candidates.push_back({d, vec_id});
                }
                
                std::partial_sort(exact_candidates.begin(), 
                                  exact_candidates.begin() + std::min((int)exact_candidates.size(), 10), 
                                  exact_candidates.end());
                
                // 保存 Top-10 结果
                final_results[query_offset + i].clear();
                for (int j = 0; j < std::min(10, (int)exact_candidates.size()); ++j) {
                    final_results[query_offset + i].push_back(exact_candidates[j].second);
                }
            }
            
            auto cpu_end = std::chrono::high_resolution_clock::now();
            double cpu_rerank_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
            old_val = pipeline_timings.cpu_rerank_ms.load();
            pipeline_timings.cpu_rerank_ms.store(old_val + cpu_rerank_ms);
            
            total_reranked += batch_total_candidates;
        }
    }
};

void load_ivf_index_to_gpu(const std::string& res_dir, IVFIndexGPU& idx) {
    std::vector<float> codebook(PQ_M * PQ_K * PQ_SUB_DIM);
    read_binary_vector(res_dir + "/global_pq_codebook.bin", codebook);
    cudaMalloc(&idx.d_pq_codebook, codebook.size() * sizeof(float));
    cudaMemcpy(idx.d_pq_codebook, codebook.data(), codebook.size() * sizeof(float), cudaMemcpyHostToDevice);

    std::ifstream ifs(res_dir + "/ivf_data.bin", std::ios::binary);
    if(!ifs) { std::cerr << "Cannot open ivf_data.bin" << std::endl; exit(1); }
    int total_vecs, n_clusters;
    ifs.read((char*)&total_vecs, sizeof(int));
    ifs.read((char*)&n_clusters, sizeof(int));
    std::vector<int> offsets(n_clusters + 1);
    std::vector<int> ids(total_vecs);
    std::vector<uint8_t> codes((size_t)total_vecs * PQ_M);
    ifs.read((char*)offsets.data(), offsets.size() * sizeof(int));
    ifs.read((char*)ids.data(), ids.size() * sizeof(int));
    ifs.read((char*)codes.data(), codes.size() * sizeof(uint8_t));
    
    cudaMalloc(&idx.d_cluster_offsets, offsets.size() * sizeof(int));
    cudaMemcpy(idx.d_cluster_offsets, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&idx.d_all_vector_ids, ids.size() * sizeof(int));
    cudaMemcpy(idx.d_all_vector_ids, ids.data(), ids.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&idx.d_all_pq_codes, codes.size() * sizeof(uint8_t));
    cudaMemcpy(idx.d_all_pq_codes, codes.data(), codes.size() * sizeof(uint8_t), cudaMemcpyHostToDevice);
}

// 计算召回率
float compute_recall(const std::vector<std::vector<int>>& results, const int* groundtruth, int nq, int gt_k, int recall_k) {
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
        std::cerr << "Usage: ./search_pipeline <base.fvecs> <query.fvecs> [groundtruth.ivecs]" << std::endl;
        return 1;
    }
    std::string base = argv[1], query = argv[2], res = "../res";
    std::string gt_file = (argc == 4) ? argv[3] : "";
    
    std::cout << "============ FusionGPU Search Benchmark ============" << std::endl;
    std::cout << "[Config]" << std::endl;
    std::cout << "  Vector dimension: " << DIM << std::endl;
    std::cout << "  PQ segments (M): " << PQ_M << std::endl;
    std::cout << "  PQ centroids (K): " << PQ_K << std::endl;
    std::cout << "  IVF clusters: " << K_CLUSTERS << std::endl;
    std::cout << "  Batch size: " << DEFAULT_BATCH_SIZE << std::endl;
    std::cout << "  CAGRA top-M (nprobe): " << DEFAULT_TOP_M << std::endl;
    std::cout << "  Rerank candidates: " << RERANK_M << std::endl;
    std::cout << std::endl;
    
    std::cout << "[Loading Data]" << std::endl;
    auto t_load_start = std::chrono::high_resolution_clock::now();
    
    float* raw;
    int n = read_vecs<float>(base, raw, DIM);
    std::cout << "  Base vectors: " << n << std::endl;
    
    IVFIndexGPU idx_gpu;
    load_ivf_index_to_gpu(res, idx_gpu);
    std::cout << "  IVF index loaded to GPU" << std::endl;
    
    float* q_ptr;
    int nq = read_vecs<float>(query, q_ptr, DIM);
    std::cout << "  Query vectors: " << nq << std::endl;
    std::vector<float> all_q(q_ptr, q_ptr + (size_t)nq * DIM);
    
    SearchPipeline pipeline(DEFAULT_BATCH_SIZE, idx_gpu, (point_t<float>*)raw, "../res/cagra_centroids.index", nq, n);
    std::cout << "  CAGRA index loaded" << std::endl;
    
    // 加载 ground truth (可选)
    int* gt_data = nullptr;
    int gt_k = 0;
    if (!gt_file.empty()) {
        int gt_nq = read_ivecs(gt_file, gt_data, gt_k);
        if (gt_nq != nq) {
            std::cerr << "Warning: groundtruth size mismatch (" << gt_nq << " vs " << nq << ")" << std::endl;
        }
        std::cout << "  Ground truth loaded (k=" << gt_k << ")" << std::endl;
    }
    
    auto t_load_end = std::chrono::high_resolution_clock::now();
    double load_time_ms = std::chrono::duration<double, std::milli>(t_load_end - t_load_start).count();
    std::cout << "  Load time: " << load_time_ms << " ms" << std::endl;
    std::cout << std::endl;
    
    // Warmup
    std::cout << "[Warmup]" << std::endl;
    {
        std::vector<float> warmup_q(all_q.begin(), all_q.begin() + std::min(nq, DEFAULT_BATCH_SIZE) * DIM);
        pipeline.submit_batch(warmup_q, 0);
        cudaDeviceSynchronize();
    }
    std::cout << "  Warmup complete" << std::endl;
    std::cout << std::endl;
    
    // 启用GPU profiling并重置计时器
    enable_gpu_profiling(true);
    reset_gpu_timings();
    
    // Benchmark
    std::cout << "[Benchmark]" << std::endl;
    auto t_start = std::chrono::high_resolution_clock::now();
    
    int batches = (nq + DEFAULT_BATCH_SIZE - 1) / DEFAULT_BATCH_SIZE;
    for(int i=0; i<batches; ++i) {
        int start = i * DEFAULT_BATCH_SIZE;
        int sz = std::min(DEFAULT_BATCH_SIZE, nq - start);
        std::vector<float> bq(sz * DIM);
        std::memcpy(bq.data(), all_q.data() + start * DIM, sz * DIM * 4);
        pipeline.submit_batch(bq, start);
    }
    
    // 等待所有工作完成（包括 CPU rerank）
    pipeline.wait_all();
    
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    double qps = nq / (elapsed_ms / 1000.0);
    double latency_per_query_us = (elapsed_ms * 1000.0) / nq;
    
    std::cout << std::endl;
    std::cout << "=============== Results ===============" << std::endl;
    std::cout << "  Total queries:      " << nq << std::endl;
    std::cout << "  Total batches:      " << batches << std::endl;
    std::cout << "  Total time:         " << std::fixed << std::setprecision(3) << elapsed_ms << " ms" << std::endl;
    std::cout << "  Throughput (QPS):   " << std::fixed << std::setprecision(2) << qps << std::endl;
    std::cout << "  Avg latency/query:  " << std::fixed << std::setprecision(2) << latency_per_query_us << " us" << std::endl;
    pipeline.print_stats(nq);
    
    // 打印流水线各阶段耗时
    pipeline.print_timing_stats();
    
    // 打印GPU内部各子阶段耗时
    GPUStageTimings gpu_timings = get_avg_gpu_timings();
    if (gpu_timings.total_ms > 0) {
        std::cout << std::endl;
        std::cout << "============= GPU Batch Logic Breakdown =============" << std::endl;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  Precompute:                " << std::setw(8) << gpu_timings.precompute_ms << " ms" << std::endl;
        std::cout << "  Count:                     " << std::setw(8) << gpu_timings.count_ms << " ms" << std::endl;
        std::cout << "  Scan Offset:               " << std::setw(8) << gpu_timings.scan_offset_ms << " ms" << std::endl;
        std::cout << "  Resize Sync (⚠️瓶颈):      " << std::setw(8) << gpu_timings.resize_sync_ms << " ms" << std::endl;
        std::cout << "  Scan Candidates:           " << std::setw(8) << gpu_timings.scan_candidates_ms << " ms" << std::endl;
        std::cout << "  Sort:                      " << std::setw(8) << gpu_timings.sort_ms << " ms" << std::endl;
        std::cout << "  Gather:                    " << std::setw(8) << gpu_timings.gather_ms << " ms" << std::endl;
        std::cout << "  Total GPU Batch Logic:     " << std::setw(8) << gpu_timings.total_ms << " ms" << std::endl;
        std::cout << "  Avg candidates per batch:   " << std::setw(8) << gpu_timings.total_candidates << std::endl;
        std::cout << "=====================================================" << std::endl;
    }
    
    // 计算召回率
    if (gt_data != nullptr) {
        const auto& results = pipeline.get_results();
        float recall_1 = compute_recall(results, gt_data, nq, gt_k, 1);
        float recall_10 = compute_recall(results, gt_data, nq, gt_k, 10);
        std::cout << std::endl;
        std::cout << "  Recall@1:           " << std::fixed << std::setprecision(4) << (recall_1 * 100) << "%" << std::endl;
        std::cout << "  Recall@10:          " << std::fixed << std::setprecision(4) << (recall_10 * 100) << "%" << std::endl;
    }
    
    std::cout << "=======================================" << std::endl;
    
    // 清理资源
    delete[] raw; 
    delete[] q_ptr;
    if (gt_data) delete[] gt_data;
    
    std::cout << "\nExiting gracefully..." << std::endl;
    
    // 使用 quick_exit 避免 cuVS/rmm 析构时的段错误
    std::quick_exit(0);
}