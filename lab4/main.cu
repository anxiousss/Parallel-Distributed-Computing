#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <iomanip>

#define CSC(call)                                           \
do {                                                        \
    cudaError_t res = call;                                 \
    if (res != cudaSuccess) {                               \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",    \
                __FILE__, __LINE__, cudaGetErrorString(res)); \
        exit(1);                                            \
    }                                                       \
} while(0)


__global__ void replace_zero_gpu(double *AB, int rows, int cols, int column) {
    int idy = threadIdx.y;
    if (fabs(AB[column * cols + column]) <= 1e-4) {
        int targetRow = column;
        for (; targetRow < rows; ++targetRow) {
            if (fabs(AB[targetRow * cols + column]) > 1e-4)
                break;
        }
        if (targetRow < rows && (column + idy) < cols) {
            int zeroPos = column * cols + column + idy;
            int chosenPos = targetRow * cols + column + idy;
            AB[zeroPos] += AB[chosenPos];
        }
    }
}


__global__ void column_elimination_gpu(double *AB, int rows, int cols, int column) {
    int el_row = column + 1 + threadIdx.y + blockIdx.y * blockDim.y;
    int el_col = column + threadIdx.x + blockIdx.x * blockDim.x;

    if (el_row < rows && el_col < cols) {
        int idx = el_row * cols + el_col;
        int upper = column * cols + el_col;
        int pivot = column * cols + column;
        int pivotRow = el_row * cols + column;
        double factor = AB[pivotRow] / AB[pivot];
        AB[idx] -= factor * AB[upper];
    }
}

__global__ void back_substitution_gpu(double *AB, int n, int cols) {
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        double* x = new double[n];
        for (int i = n - 1; i >= 0; --i) {
            double sum = 0.0;
            for (int j = i + 1; j < n; ++j) {
                sum += AB[i * cols + j] * x[j];
            }
            x[i] = (AB[i * cols + n] - sum) / AB[i * cols + i];
        }
        for (int i = 0; i < n; ++i) {
            AB[i * cols + n] = x[i];
        }
        delete[] x;
    }
}


void forwardElimination(double *d_AB, int n, int cols) {
    for (int col = 0; col < n; ++col) {
        dim3 blockReplace(1, cols - col);
        dim3 gridReplace(1, 1);
        replace_zero_gpu<<<gridReplace, blockReplace>>>(d_AB, n, cols, col);
        CSC(cudaGetLastError());
        CSC(cudaDeviceSynchronize());

        int rowsBelow = n - col - 1;
        int colsRight = cols - col;
        if (rowsBelow > 0 && colsRight > 0) {
            const int TILE_X = 16;
            const int TILE_Y = 16;
            dim3 blockElim(TILE_X, TILE_Y);
            dim3 gridElim((colsRight + TILE_X - 1) / TILE_X,
                          (rowsBelow + TILE_Y - 1) / TILE_Y);
            column_elimination_gpu<<<gridElim, blockElim>>>(d_AB, n, cols, col);
            CSC(cudaGetLastError());
            CSC(cudaDeviceSynchronize());
        }
    }
}


void backwardSubstitution(double *d_AB, int n, int cols) {
    back_substitution_gpu<<<1, 1>>>(d_AB, n, cols);
    CSC(cudaGetLastError());
    CSC(cudaDeviceSynchronize());
}


int main() {
    int n;
    std::cin >> n;
    int cols = n + 1;
    size_t totalElements = (size_t)n * cols;

    double *h_AB = new double[totalElements];

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cin >> h_AB[i * cols + j];
        }
    }
    for (int i = 0; i < n; ++i) {
        std::cin >> h_AB[i * cols + n];
    }

    double *d_AB = nullptr;
    size_t bytes = totalElements * sizeof(double);
    CSC(cudaMalloc(&d_AB, bytes));
    CSC(cudaMemcpy(d_AB, h_AB, bytes, cudaMemcpyHostToDevice));

    forwardElimination(d_AB, n, cols);
    backwardSubstitution(d_AB, n, cols);

    CSC(cudaMemcpy(h_AB, d_AB, bytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; ++i) {
        std::cout << std::scientific << std::setprecision(10) << h_AB[i * cols + n] << ' ';
    }

    delete[] h_AB;
    CSC(cudaFree(d_AB));

    return 0;
}