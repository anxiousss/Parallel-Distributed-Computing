#include <cstdlib>
#include <iostream>
#include <iomanip>

void bubble_sort(float* arr, int n) {
    for (size_t i = 0; i < n; ++i) {
        bool flag = false;
        for (size_t j = 0; j < n - 1; ++j) {
            if (arr[j] > arr[j + 1]) {
                std::swap(arr[j], arr[j + 1]);
                flag = true;
            }
        }
        if (!flag)
            break;
    }
}

int main() {
    int n;
    std::cin >> n;

    auto* arr = (float*)(malloc(sizeof(float) * n));
    if (!arr) {
        printf("Memory leak\n");
        return -1;
    }
    for (size_t i = 0; i < n; ++i) {
        std::cin >> arr[i];
    }

    bubble_sort(arr, n);

    for (size_t i = 0; i < n; ++i) {
        std::cout << std::scientific << std::setprecision(6) <<  arr[i] << ' ';
    }
    free(arr);

    return 0;
}