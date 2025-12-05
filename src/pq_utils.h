#ifndef PQ_UTILS_H_
#define PQ_UTILS_H_

#include "defs.h"
#include <vector>
#include <cmath>
#include <limits>
#include <random>
#include <algorithm>
#include <omp.h>
#include <cstring>

// 简单的 CPU K-Means
inline void run_kmeans(const float* data, int n, int dim, int k, std::vector<float>& centroids) {
    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> dist(0, n - 1);
    
    // 初始化
    for (int i = 0; i < k; ++i) {
        int idx = dist(rng);
        std::memcpy(centroids.data() + i * dim, data + idx * dim, dim * sizeof(float));
    }

    std::vector<int> assign(n);
    std::vector<int> counts(k);
    
    for (int iter = 0; iter < 20; ++iter) { 
        // Assignment
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            float min_d = std::numeric_limits<float>::max();
            int best_c = 0;
            for (int c = 0; c < k; ++c) {
                float d = 0;
                for(int j=0; j<dim; ++j) {
                    float diff = data[i*dim+j] - centroids[c*dim+j];
                    d += diff*diff;
                }
                if (d < min_d) { min_d = d; best_c = c; }
            }
            assign[i] = best_c;
        }

        // Update
        std::fill(centroids.begin(), centroids.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);
        
        // Accumulate (Thread local reduction omitted for brevity, using atomic or single thread is safer but slower. 
        // For simplicity in this snippet, we iterate serially for update or use critical)
        for (int i = 0; i < n; ++i) {
            int c = assign[i];
            counts[c]++;
            for (int j = 0; j < dim; ++j) centroids[c*dim+j] += data[i*dim+j];
        }

        for (int c = 0; c < k; ++c) {
            if (counts[c] > 0) {
                for (int j = 0; j < dim; ++j) centroids[c*dim+j] /= counts[c];
            }
        }
    }
}

inline void train_pq_codebooks(const point_t<float>* points, int n, std::vector<float>& codebook) {
    // Flatten data for easier access
    const float* flat_data = (const float*)points;
    
    #pragma omp parallel for
    for (int m = 0; m < PQ_M; ++m) {
        std::vector<float> sub_vecs(n * PQ_SUB_DIM);
        for (int i = 0; i < n; ++i) {
            for (int d = 0; d < PQ_SUB_DIM; ++d)
                sub_vecs[i * PQ_SUB_DIM + d] = flat_data[i * DIM + m * PQ_SUB_DIM + d];
        }
        
        std::vector<float> sub_centroids(PQ_K * PQ_SUB_DIM);
        run_kmeans(sub_vecs.data(), n, PQ_SUB_DIM, PQ_K, sub_centroids);
        
        // Copy to global codebook
        size_t offset = m * PQ_K * PQ_SUB_DIM;
        std::copy(sub_centroids.begin(), sub_centroids.end(), codebook.begin() + offset);
    }
}

inline void encode_vectors_to_pq(const point_t<float>* points, int n, const std::vector<float>& codebook, uint8_t* codes) {
    const float* flat_data = (const float*)points;
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int m = 0; m < PQ_M; ++m) {
            float min_d = std::numeric_limits<float>::max();
            uint8_t best_k = 0;
            const float* sub_data = flat_data + i * DIM + m * PQ_SUB_DIM;
            
            for (int k = 0; k < PQ_K; ++k) {
                const float* center = codebook.data() + (m * PQ_K + k) * PQ_SUB_DIM;
                float d = 0;
                for(int j=0; j<PQ_SUB_DIM; ++j) {
                    float diff = sub_data[j] - center[j];
                    d += diff * diff;
                }
                if (d < min_d) { min_d = d; best_k = k; }
            }
            codes[i * PQ_M + m] = best_k;
        }
    }
}

#endif // PQ_UTILS_H_