#include <cmath>
#include <iostream>
#include <iomanip>

int main() {
    const float eps = 1e-10;

    float a, b, c;
    std::cin >> a >> b >> c;

    if ( fabs(a) < eps && fabs(b) < eps && fabs(c) < eps) {
        std::cout << "any";
    } else if (fabs(a) < eps && fabs(b) < eps && fabs(c) >= eps) {
        std::cout << "incorrect";
    } else if (fabs(a) < eps && fabs(c) < eps && fabs(b) >= eps) {
        std::cout << std::fixed << std::setprecision(6) << 0.0;
    } else if (fabs(a) < eps) {
        std::cout << std::fixed << std::setprecision(6) << -c / b;
    }
    else  {
        float D = powf(b, 2) - 4 * a * c;
        if (D > 0) {
            std::cout << std::fixed << std::setprecision(6) <<  (-b + sqrtf(D)) / (2 * a) << ' ' << (- b - sqrtf(D)) / (2 * a);
        } else if (fabs(D - 0.0) < eps) {
            std::cout << std::fixed << std::setprecision(6) << -b / (2 * a);
        } else {
            std::cout << "imaginary";
        }
    }

    return 0;
}