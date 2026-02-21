#include <iostream>
#include <iomanip>

__global__ void diffKernel(double* a, double* b, double* c) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    c[idx] = a[idx] - b[idx];
}

void diff(double* a, double* b, double* c, int n, dim3 threads) {
    int bytes = n * sizeof(double);

    double* aDev = NULL;
    double* bDev = NULL;
    double* cDev = NULL;

    cudaMalloc((void**)&aDev, bytes);
    cudaMalloc((void**)&bDev, bytes);
    cudaMalloc((void**)&cDev, bytes);

    dim3 blocks = dim3((n + threads.x - 1) / threads.x, 1);

    cudaMemcpy(aDev, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(bDev, b, bytes, cudaMemcpyHostToDevice);

    diffKernel<<<blocks, threads>>>(aDev, bDev, cDev);

    cudaMemcpy(c, cDev, bytes, cudaMemcpyDeviceToHost);

    cudaFree(aDev);
    cudaFree(bDev);
    cudaFree(cDev);
}


int main() {
    int n;
    std::cin >> n;

    if (n == 0)
        return 0;

    auto* a = new double[n];
    auto* b = new double[n];
    auto* c = new double[n];

    for (size_t i = 0; i < n; ++i) {
        std::cin >> a[i];
    }

    for (size_t i = 0; i < n; ++i) {
        std::cin >> b[i];
    }

    dim3 threads = dim3(1024 ,1);

    diff(a, b, c, n, threads);

    for (size_t i = 0; i < n; ++i) {
        std::cout << std::scientific << std::setprecision(10) << c[i] << ' ';
    }

    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}