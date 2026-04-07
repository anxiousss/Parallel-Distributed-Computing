#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cmath>

#define CSC(call)                                           \
do {                                                        \
    cudaError_t res = call;                                 \
    if (res != cudaSuccess) {                               \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",    \
                __FILE__, __LINE__, cudaGetErrorString(res)); \
        exit(1);                                            \
    }                                                       \
} while(0)

bool invert3x3(const float m[9], float inv[9]) {
    float det = m[0]*(m[4]*m[8] - m[5]*m[7]) -
                m[1]*(m[3]*m[8] - m[5]*m[6]) +
                m[2]*(m[3]*m[7] - m[4]*m[6]);
    if (fabs(det) < 1e-6f) return false;
    float invdet = 1.0f / det;
    inv[0] =  (m[4]*m[8] - m[5]*m[7]) * invdet;
    inv[1] = -(m[1]*m[8] - m[2]*m[7]) * invdet;
    inv[2] =  (m[1]*m[5] - m[2]*m[4]) * invdet;
    inv[3] = -(m[3]*m[8] - m[5]*m[6]) * invdet;
    inv[4] =  (m[0]*m[8] - m[2]*m[6]) * invdet;
    inv[5] = -(m[0]*m[5] - m[2]*m[3]) * invdet;
    inv[6] =  (m[3]*m[7] - m[4]*m[6]) * invdet;
    inv[7] = -(m[0]*m[7] - m[1]*m[6]) * invdet;
    inv[8] =  (m[0]*m[4] - m[1]*m[3]) * invdet;
    return true;
}

void computeClassStats(const std::vector<uchar4>& image, int w, int h,
                       const std::vector<std::pair<int,int>>& pixels,
                       float3& mean, float inv_cov[9]) {
    int n = (int)pixels.size();
    if (n == 0) {
        mean = make_float3(0,0,0);
        for (int i=0;i<9;i++) inv_cov[i] = (i%4==0)?1.0f:0.0f;
        return;
    }
    float3 sum = make_float3(0,0,0);
    for (auto& p : pixels) {
        int x = p.first, y = p.second;
        if (x<0 || x>=w || y<0 || y>=h) continue;
        uchar4 c = image[y*w + x];
        sum.x += c.x;
        sum.y += c.y;
        sum.z += c.z;
    }
    mean.x = sum.x / n;
    mean.y = sum.y / n;
    mean.z = sum.z / n;

    float cov[9] = {0};
    for (auto& p : pixels) {
        int x = p.first, y = p.second;
        if (x<0 || x>=w || y<0 || y>=h) continue;
        uchar4 c = image[y*w + x];
        float3 diff = make_float3(c.x - mean.x, c.y - mean.y, c.z - mean.z);
        cov[0] += diff.x * diff.x;
        cov[1] += diff.x * diff.y;
        cov[2] += diff.x * diff.z;
        cov[3] += diff.y * diff.x;
        cov[4] += diff.y * diff.y;
        cov[5] += diff.y * diff.z;
        cov[6] += diff.z * diff.x;
        cov[7] += diff.z * diff.y;
        cov[8] += diff.z * diff.z;
    }
    float norm = (n > 1) ? 1.0f/(n-1) : 1.0f;
    for (int i=0;i<9;i++) cov[i] *= norm;

    if (n < 3) {
        for (int i=0;i<3;i++) {
            for (int j=0;j<3;j++) {
                inv_cov[i*3+j] = (i==j) ? 1.0f / (cov[i*3+i] + 1e-6f) : 0.0f;
            }
        }
    } else {
        if (!invert3x3(cov, inv_cov)) {
            for (int i=0;i<3;i++) {
                for (int j=0;j<3;j++) {
                    inv_cov[i*3+j] = (i==j) ? 1.0f / (cov[i*3+i] + 1e-6f) : 0.0f;
                }
            }
        }
    }
}

__constant__ float const_means[256 * 3];
__constant__ float const_inv_covs[256 * 9];

__global__ void kernel1D(const uchar4* pixels, uchar4* out, int w, int h, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= w * h) return;

    uchar4 pixel = pixels[idx];
    float3 color = make_float3(pixel.x, pixel.y, pixel.z);
    float min_dist2 = 1e30f;
    int best_class = 0;

    for (int c = 0; c < num_classes; ++c) {
        const float* mean = const_means + c * 3;
        const float* inv = const_inv_covs + c * 9;
        float3 diff;
        diff.x = color.x - mean[0];
        diff.y = color.y - mean[1];
        diff.z = color.z - mean[2];
        float dist2 = 0.0f;
        dist2 += diff.x * (inv[0]*diff.x + inv[1]*diff.y + inv[2]*diff.z);
        dist2 += diff.y * (inv[3]*diff.x + inv[4]*diff.y + inv[5]*diff.z);
        dist2 += diff.z * (inv[6]*diff.x + inv[7]*diff.y + inv[8]*diff.z);
        if (dist2 < min_dist2) {
            min_dist2 = dist2;
            best_class = c;
        }
    }
    out[idx] = make_uchar4(pixel.x, pixel.y, pixel.z, (unsigned char)best_class);
}

__global__ void setAlphaZero(uchar4* out, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) out[idx].w = 0;
}

__global__ void kernelGlobal(const uchar4* pixels, uchar4* out, int w, int h,
                             const float3* means, const float* inv_covs, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= w * h) return;

    uchar4 pixel = pixels[idx];
    float3 color = make_float3(pixel.x, pixel.y, pixel.z);
    float min_dist2 = 1e30f;
    int best_class = 0;

    for (int c = 0; c < num_classes; ++c) {
        float3 diff;
        diff.x = color.x - means[c].x;
        diff.y = color.y - means[c].y;
        diff.z = color.z - means[c].z;
        const float* inv = inv_covs + c * 9;
        float dist2 = 0.0f;
        dist2 += diff.x * (inv[0]*diff.x + inv[1]*diff.y + inv[2]*diff.z);
        dist2 += diff.y * (inv[3]*diff.x + inv[4]*diff.y + inv[5]*diff.z);
        dist2 += diff.z * (inv[6]*diff.x + inv[7]*diff.y + inv[8]*diff.z);
        if (dist2 < min_dist2) {
            min_dist2 = dist2;
            best_class = c;
        }
    }
    out[idx] = make_uchar4(pixel.x, pixel.y, pixel.z, (unsigned char)best_class);
}

int main() {
    std::string in_file, out_file;
    if (!(std::cin >> in_file >> out_file)) {
        std::cerr << "Error reading file names" << std::endl;
        return 1;
    }

    std::ifstream in(in_file, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Error opening input file: " << in_file << std::endl;
        return 1;
    }

    int w = 0, h = 0;
    if (!in.read(reinterpret_cast<char*>(&w), sizeof(int))) {
        std::cerr << "Failed to read image width" << std::endl;
        return 1;
    }
    if (!in.read(reinterpret_cast<char*>(&h), sizeof(int))) {
        std::cerr << "Failed to read image height" << std::endl;
        return 1;
    }

    std::cerr << "Read dimensions: w=" << w << ", h=" << h << std::endl;
    size_t pixel_count = static_cast<size_t>(w) * h;

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t needed_mem = pixel_count * sizeof(uchar4) * 2;
    if (needed_mem > free_mem) {
        std::cerr << "Not enough GPU memory. Need " << needed_mem / (1024*1024)
                  << " MB, free " << free_mem / (1024*1024) << " MB" << std::endl;
        return 1;
    }

    std::vector<uchar4> data(pixel_count);
    in.read(reinterpret_cast<char*>(data.data()), pixel_count * sizeof(uchar4));
    if (in.gcount() != static_cast<std::streamsize>(pixel_count * sizeof(uchar4))) {
        std::cerr << "Failed to read pixel data" << std::endl;
        return 1;
    }
    in.close();

    size_t nc;
    if (!(std::cin >> nc)) {
        std::cerr << "Error reading number of classes" << std::endl;
        return 1;
    }

    std::vector<std::vector<std::pair<int,int>>> classes(nc);
    for (size_t i = 0; i < nc; ++i) {
        int n_pairs;
        if (!(std::cin >> n_pairs)) {
            std::cerr << "Error reading number of points for class " << i << std::endl;
            return 1;
        }
        classes[i].reserve(n_pairs);
        for (int j = 0; j < n_pairs; ++j) {
            int x, y;
            if (!(std::cin >> x >> y)) {
                std::cerr << "Error reading point for class " << i << std::endl;
                return 1;
            }
            if (x < 0 || x >= w || y < 0 || y >= h) {
                std::cerr << "Warning: point (" << x << "," << y << ") out of bounds, skipping" << std::endl;
                continue;
            }
            classes[i].emplace_back(x, y);
        }
        if (classes[i].empty()) {
            std::cerr << "Warning: class " << i << " has no valid points" << std::endl;
        }
    }

    std::vector<float3> h_means(nc);
    std::vector<float> h_inv_covs(nc * 9);
    for (size_t i = 0; i < nc; ++i) {
        if (!classes[i].empty()) {
            computeClassStats(data, w, h, classes[i], h_means[i], &h_inv_covs[i*9]);
        } else {
            h_means[i] = make_float3(0,0,0);
            for (int k = 0; k < 9; ++k) h_inv_covs[i*9 + k] = (k % 4 == 0) ? 1.0f : 0.0f;
        }
    }

    uchar4 *dev_pixels = nullptr, *dev_out = nullptr;
    CSC(cudaMalloc(&dev_pixels, pixel_count * sizeof(uchar4)));
    CSC(cudaMalloc(&dev_out, pixel_count * sizeof(uchar4)));
    CSC(cudaMemcpy(dev_pixels, data.data(), pixel_count * sizeof(uchar4), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (pixel_count + blockSize - 1) / blockSize;

    if (nc == 0) {
        CSC(cudaMemcpy(dev_out, dev_pixels, pixel_count * sizeof(uchar4), cudaMemcpyDeviceToDevice));
        setAlphaZero<<<gridSize, blockSize>>>(dev_out, pixel_count);
        CSC(cudaGetLastError());
        CSC(cudaDeviceSynchronize());
    } else {
        const size_t MAX_CONST_CLASSES = 256;
        if (nc <= MAX_CONST_CLASSES) {
            std::vector<float> flat_means(nc * 3);
            for (size_t i = 0; i < nc; ++i) {
                flat_means[i*3 + 0] = h_means[i].x;
                flat_means[i*3 + 1] = h_means[i].y;
                flat_means[i*3 + 2] = h_means[i].z;
            }
            CSC(cudaMemcpyToSymbol(const_means, flat_means.data(), nc * 3 * sizeof(float)));
            CSC(cudaMemcpyToSymbol(const_inv_covs, h_inv_covs.data(), nc * 9 * sizeof(float)));

            kernel1D<<<gridSize, blockSize>>>(dev_pixels, dev_out, w, h, (int)nc);
            CSC(cudaGetLastError());
            CSC(cudaDeviceSynchronize());
        } else {
            float3* dev_means = nullptr;
            float* dev_inv_covs = nullptr;
            CSC(cudaMalloc(&dev_means, nc * sizeof(float3)));
            CSC(cudaMalloc(&dev_inv_covs, nc * 9 * sizeof(float)));
            CSC(cudaMemcpy(dev_means, h_means.data(), nc * sizeof(float3), cudaMemcpyHostToDevice));
            CSC(cudaMemcpy(dev_inv_covs, h_inv_covs.data(), nc * 9 * sizeof(float), cudaMemcpyHostToDevice));

            kernelGlobal<<<gridSize, blockSize>>>(dev_pixels, dev_out, w, h, dev_means, dev_inv_covs, (int)nc);
            CSC(cudaGetLastError());
            CSC(cudaDeviceSynchronize());

            CSC(cudaFree(dev_means));
            CSC(cudaFree(dev_inv_covs));
        }
    }

    CSC(cudaMemcpy(data.data(), dev_out, pixel_count * sizeof(uchar4), cudaMemcpyDeviceToHost));

    CSC(cudaFree(dev_pixels));
    CSC(cudaFree(dev_out));

    std::ofstream out(out_file, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        std::cerr << "Error opening output file: " << out_file << std::endl;
        return 1;
    }
    out.write(reinterpret_cast<const char*>(&w), sizeof(int));
    out.write(reinterpret_cast<const char*>(&h), sizeof(int));
    out.write(reinterpret_cast<const char*>(data.data()), pixel_count * sizeof(uchar4));
    if (!out) {
        std::cerr << "Error writing output file" << std::endl;
        return 1;
    }
    out.close();

    std::cout << "Processing complete. Result saved to " << out_file << std::endl;
    return 0;
}