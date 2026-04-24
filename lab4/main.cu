#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <iomanip>
#include <cstdio>

#define CSC(call) do { \
    cudaError_t res = call; \
    if (res != cudaSuccess) { \
        exit(1); \
    } \
} while(0)

const double EPSILON = 1e-12;

__global__ void find_and_swap_pivot_gpu(double *AB, int n, int cols, int col) {
    int max_row = col;
    double max_val = fabs(AB[col * cols + col]);

    for (int r = col + 1; r < n; ++r) {
        double val = fabs(AB[r * cols + col]);
        if (val > max_val) {
            max_val = val;
            max_row = r;
        }
    }

    if (max_row != col) {
        for (int c = col; c < cols; ++c) {
            double tmp = AB[col * cols + c];
            AB[col * cols + c] = AB[max_row * cols + c];
            AB[max_row * cols + c] = tmp;
        }
    }
}

__global__ void column_elimination_gpu(double *AB, int n, int cols, int col) {
    int r = col + 1 + blockIdx.y * blockDim.y + threadIdx.y;
    int c = col + 1 + blockIdx.x * blockDim.x + threadIdx.x;

    if (r < n && c < cols) {
        double pivot = AB[col * cols + col];
        if (fabs(pivot) > EPSILON) {
            double factor = AB[r * cols + col] / pivot;
            AB[r * cols + c] -= factor * AB[col * cols + c];
        }
    }
}

int main() {
    int n;
    if (scanf("%d", &n) != 1) return 0;

    int cols = n + 1;
    size_t total_elements = (size_t)n * cols;
    std::vector<double> h_AB(total_elements);

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            scanf("%lf", &h_AB[i * cols + j]);

    for (int i = 0; i < n; ++i)
        scanf("%lf", &h_AB[i * cols + n]);

    double *d_AB;
    CSC(cudaMalloc(&d_AB, total_elements * sizeof(double)));
    CSC(cudaMemcpy(d_AB, h_AB.data(), total_elements * sizeof(double), cudaMemcpyHostToDevice));

    dim3 threads(32, 8);

    for (int col = 0; col < n; ++col) {

        find_and_swap_pivot_gpu<<<1, 1>>>(d_AB, n, cols, col);

        int rows_to_process = n - 1 - col;
        int cols_to_process = cols - 1 - col;

        if (rows_to_process > 0 && cols_to_process > 0) {
            dim3 blocks((cols_to_process + threads.x - 1) / threads.x,
                        (rows_to_process + threads.y - 1) / threads.y);
            column_elimination_gpu<<<blocks, threads>>>(d_AB, n, cols, col);
        }
    }

    CSC(cudaMemcpy(h_AB.data(), d_AB, total_elements * sizeof(double), cudaMemcpyDeviceToHost));

    std::vector<double> x(n);
    for (int i = n - 1; i >= 0; --i) {
        double sum = 0;
        for (int j = i + 1; j < n; ++j) {
            sum += h_AB[i * cols + j] * x[j];
        }
        double pivot = h_AB[i * cols + i];
        x[i] = (fabs(pivot) < EPSILON) ? 0 : (h_AB[i * cols + n] - sum) / pivot;
    }

    for (int i = 0; i < n; ++i) {
        printf("%.10e%c", x[i], (i == n - 1 ? '\n' : ' '));
    }

    cudaFree(d_AB);
    return 0;
}