#include <iostream>
#include <omp.h>
#include <chrono>
#include <vector>
#include <functional>
#include <future>
#include <numeric>

unsigned long long fib_recursive(unsigned int n) {
    if (n <= 1)
        return n;

    return fib_recursive(n - 1) + fib_recursive(n - 2);
}

unsigned long long fib_iterative(unsigned int n) {
    if (n <= 1)
        return n;

    unsigned long long a = 0, b = 1;
    for (int i = 2; i <= n; ++i) {
        unsigned long long temp = a + b;
        a = b;
        b = temp;
    }

    return b;
}

unsigned long long fib_iterative_omp(unsigned int n) {
    if (n <= 1)
        return n;

    unsigned long long a = 0, b = 1;

    #pragma omp parallel
    {
        #pragma omp single nowait
        for (int i = 2; i <= n; ++i) {
            #pragma omp task shared(a, b)
            {
                unsigned long long temp = a + b;
                a = b;
                b = temp;
            }
        }
    }

    return b;
}

unsigned long long fib_omp(unsigned int n) {
    if (n <= 1) {
        return n;
    }

    unsigned long long r1, r2;
    #pragma omp task shared(r1)
    {
        r1 = fib_omp(n - 1);
    }

    #pragma omp task shared(r2)
    {
        r2 = fib_omp(n - 2);
    }

    #pragma omp taskwait

    return r1 + r2;
}

int main() {
    unsigned int number = 20;
    //std::cin >> number;

    std::vector<std::pair<const char *, std::function<unsigned long long(unsigned int n)>>> funcs{
            {"fib_omp", fib_omp},
            {"fib_recursive", fib_recursive},
            {"fib_iterative", fib_iterative},
            {"fib_iterative_omp", fib_iterative_omp},
    };

    std::cout << "method;result;time" << std::endl;
    for (auto& f : funcs) {
        auto start_time = std::chrono::high_resolution_clock::now();

        for (unsigned int i = 1; i < number; i++)
            f.second(number);

        auto future_result = std::async(std::launch::async, f.second, number);
        auto res = future_result.get();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        std::cout << f.first << ";" << res << ";" << duration.count() << std::endl;
    }

    return 0;
}