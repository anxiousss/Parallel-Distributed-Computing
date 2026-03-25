#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>

#define CSC(call)                                           \
do {                                                        \
    cudaError_t res = call;                                 \
    if (res != cudaSuccess) {                               \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",    \
                __FILE__, __LINE__, cudaGetErrorString(res)); \
        exit(1);                                            \
    }                                                       \
} while(0)

__device__ int clamp_idx(int idx, int dim) {
    if (idx < 0) return 0;
    if (idx >= dim) return dim - 1;
    return idx;
}

__device__ float to_yuv(cudaTextureObject_t tex, int x, int y) {
    uchar4 p = tex2D<uchar4>(tex, x, y);
    return 0.299f * p.x + 0.587f * p.y + 0.114f * p.z;
}

__device__ float get_pixel_clamped(cudaTextureObject_t tex, int x, int y, int w, int h) {
    int cx = clamp_idx(x, w);
    int cy = clamp_idx(y, h);
    return to_yuv(tex, cx, cy);
}

__global__ void kernel(cudaTextureObject_t tex, uchar4 *out, int w, int h) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int y = idy; y < h; y += offsety) {
        for (int x = idx; x < w; x += offsetx) {
            float v00 = get_pixel_clamped(tex, x,     y,     w, h);
            float v11 = get_pixel_clamped(tex, x + 1, y + 1, w, h);
            float v10 = get_pixel_clamped(tex, x + 1, y,     w, h);
            float v01 = get_pixel_clamped(tex, x,     y + 1, w, h);

            float g1 = v00 - v11;
            float g2 = v10 - v01;

            float g = sqrtf(g1 * g1 + g2 * g2);
            g = fminf(g, 255.0f);

            out[y * w + x] = make_uchar4(
                    static_cast<unsigned char>(g),
                    static_cast<unsigned char>(g),
                    static_cast<unsigned char>(g),
                    0
            );
        }
    }
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

    if (w <= 0 || h <= 0) {
        std::cerr << "Invalid image dimensions: " << w << "x" << h << std::endl;
        return 1;
    }

    size_t pixel_count = static_cast<size_t>(w) * h;
    std::vector<uchar4> data(pixel_count);

    in.read(reinterpret_cast<char*>(data.data()), pixel_count * sizeof(uchar4));
    if (in.gcount() != static_cast<std::streamsize>(pixel_count * sizeof(uchar4))) {
        std::cerr << "Failed to read pixel data" << std::endl;
        return 1;
    }
    in.close();

    cudaArray *arr = nullptr;
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
    CSC(cudaMallocArray(&arr, &ch, w, h));

    CSC(cudaMemcpy2DToArray(arr, 0, 0, data.data(), w * sizeof(uchar4),
                            w * sizeof(uchar4), h, cudaMemcpyHostToDevice));

    cudaResourceDesc resDesc{};
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = arr;

    cudaTextureDesc texDesc{};
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;

    cudaTextureObject_t tex = 0;
    CSC(cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr));

    uchar4 *dev_out = nullptr;
    CSC(cudaMalloc(&dev_out, pixel_count * sizeof(uchar4)));

    dim3 block(32, 32);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    kernel<<<grid, block>>>(tex, dev_out, w, h);
    CSC(cudaGetLastError());
    CSC(cudaDeviceSynchronize());

    CSC(cudaMemcpy(data.data(), dev_out, pixel_count * sizeof(uchar4), cudaMemcpyDeviceToHost));

    CSC(cudaDestroyTextureObject(tex));
    CSC(cudaFreeArray(arr));
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