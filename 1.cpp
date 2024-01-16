#include <memory>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <thread>
#include <mutex>
#include <vector>
#include <iostream>
#include <chrono>
#include <concepts>
#include "barrier.hpp"
#include <cstring>
#include <functional>
#include <utility>

unsigned get_num_thread();
void set_num_threads(unsigned T);

struct partial_sum_t
{
    alignas(64) double value;
};

typedef struct profiling_results_t
{
    double result, time, speedup, efficiency;
    unsigned T;
} profiling_results_t;

template <class F>
auto run_experiment(F func, const double *v, size_t n)
    requires std::is_invocable_r_v<double, F, const double *, size_t>
{
    std::vector<profiling_results_t> res_table;
    auto Tmax = get_num_thread();
    for (unsigned int T = 1; T <= Tmax; ++T)
    {
        using namespace std::chrono;
        res_table.emplace_back();
        auto &rr = res_table.back();
        set_num_threads(T);
        auto t1 = steady_clock::now();
        rr.result = func(v, n);
        auto t2 = steady_clock::now();
        rr.time = duration_cast<milliseconds>(t2 - t1).count();
        rr.speedup = res_table.front().time / rr.time;
        rr.efficiency = rr.speedup / T;
        rr.T = T;
    }
    return res_table;
}

double average(const double *v, size_t n)
{
    double res = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        res += v[i];
    }
    return res / n;
}

double average_reduce(const double *v, size_t n)
{
    double res = 0.0;
#pragma omp parallel for reduction(+ : res)
    for (size_t i = 0; i < n; ++i)
    {
        res += v[i];
    }
    return res / n;
}

double average_rr(const double *v, size_t n) // Round Robin
{
    double res = 0.0;
#pragma omp parallel
    {
        unsigned t = omp_get_thread_num();
        unsigned T = omp_get_num_threads();
        for (size_t i = t; i < n; i += T)
        {
            res += v[i]; // Гонка
        }
    }
    return res / n;
}

double average_omp(const double *v, size_t n)
{
    double res = 0.0, *partial_sums;
    unsigned T;
#pragma omp parallel shared(T)
    {
        unsigned t = omp_get_thread_num();
#pragma omp single
        {
            T = omp_get_num_threads();
            partial_sums = (double *)calloc(T, sizeof(v[0]));
        }
        for (size_t i = t; i < n; i += T)
        {
            partial_sums[t] += v[i];
        }
    }
    for (size_t i = 1; i < omp_get_num_procs(); ++i)
    {
        partial_sums[0] += partial_sums[i];
    }
    res = partial_sums[0] / n;
    free(partial_sums);
    return res;
}

double average_omp_align(const double *v, size_t n)
{
    double res = 0.0;
    partial_sum_t *partial_sums;
    unsigned T;
#pragma omp parallel shared(T)
    {
        unsigned t = omp_get_thread_num();
#pragma omp single
        {
            T = omp_get_num_threads();
            partial_sums = (partial_sum_t *)calloc(T, sizeof(partial_sum_t));
        }
        for (size_t i = t; i < n; i += T)
        {
            partial_sums[t].value += v[i];
        }
    }
    for (size_t i = 1; i < T; ++i)
    {
        partial_sums[0].value += partial_sums[i].value;
    }
    res = partial_sums[0].value / n;
    free(partial_sums);
    return res;
}

double average_omp_mtx(const double *v, size_t n)
{
    double res = 0.0;
#pragma omp parallel
    {
        unsigned int t = omp_get_thread_num();
        unsigned int T = omp_get_num_threads();
        for (size_t i = t; i < n; i += T)
        {
#pragma omp critial
            {
                res += v[i];
            }
        }
    }
    return res / n;
}

double average_omp_mtx_opt(const double *v, size_t n)
{
    double res = 0.0;
#pragma omp parallel
    {
        double partial = 0.0;
        unsigned int t = omp_get_thread_num();
        unsigned int T = omp_get_num_threads();
        for (size_t i = t; i < n; i += T)
        {

            partial += v[i];
        }
#pragma omp critical
        {
            res += partial;
        }
    }
    return res / n;
}

double average_cpp_mtx(const double *v, size_t n)
{
    double res = 0.0;
    unsigned T = get_num_thread();
    std::vector<std::thread> workers;
    std::mutex mtx;
    auto worker_proc = [&mtx, T, n, &res, v](unsigned t)
    {
        double partial_result = 0.0;
        for (std::size_t i = t; i < n; i += T)
        {
            partial_result += v[i];
        }
        std::scoped_lock l{mtx};
        res += partial_result;
    };
    for (unsigned t = 1; t < T; ++t)
    {
        workers.emplace_back(worker_proc, t);
    }
    worker_proc(0);
    for (auto &w : workers)
    {
        w.join();
    }
    return res / n;
}

double average_cpp_partial_align(const double *v, size_t n)
{
    double res = 0.0;
    unsigned T = get_num_thread();
    std::vector<std::thread> workers;
    partial_sum_t *partial_sums = (partial_sum_t *)calloc(T, sizeof(partial_sum_t));
    auto worker_proc = [v, n, T, &res, partial_sums](size_t t)
    {
        for (size_t i = t; i < n; i += T)
        {
            partial_sums[t].value += v[i];
        }
    };
    for (unsigned t = 1; t < T; ++t)
    {
        workers.emplace_back(worker_proc, t);
    }
    worker_proc(0);
    for (auto &w : workers)
    {
        w.join();
    }
    for (size_t i = 1; i < T; ++i)
    {
        partial_sums[0].value += partial_sums[i].value;
    }
    res = partial_sums[0].value / n;
    free(partial_sums);
    return res;
}

double average_mtx_local(const double *v, size_t n)
{
    double res = 0.0;

    unsigned T = get_num_thread();
    std::mutex mtx;
    size_t e = n / T;
    size_t b = n % T;
    std::vector<std::thread> workers;
    auto worker_proc = [v, n, T, &res, &mtx](size_t t, size_t e, size_t b)
    {
        double local = 0.0;
        if (t < b)
        {
            b = t * ++e;
        }
        else
        {
            b += t * e;
        }
        e += b;
        for (size_t i = b; i < e; ++i)
        {
            local += v[i];
        }
        std::scoped_lock l{mtx};
        res += local;
    };
    for (unsigned t = 1; t < T; ++t)
    {
        workers.emplace_back(worker_proc, t, e, b);
    }
    worker_proc(0, e, b);
    for (auto &w : workers)
    {
        w.join();
    }
    return res / n;
}

double average_cpp_reduce(const double *v, size_t n)
{
    std::vector<std::thread> workers;
    double res;
    unsigned T = get_num_thread();
    size_t e = n / T;
    size_t b = n % T;
    barrier bar(T);
    std::vector<double> partial(T, 0.0);
    auto fill_proc = [v, n, T, &partial](size_t t, size_t e, size_t b)
    {
        if (t < b)
        {
            b = t * ++e;
        }
        else
        {
            b += t * e;
        }
        e += b;
        for (size_t i = b; i < e; ++i)
        {
            partial[t] += v[i];
        }
    };
    auto worker_proc = [&partial, &T, &bar, &fill_proc, e, b](size_t t)
    {
        fill_proc(t, e, b);
        for (size_t step = 1, next = 2; step < T; step = next, next += next)
        {
            bar.arrive_and_wait();
            if (((t & (next - 1)) == 0 && t + step < T))
            {
                partial[t] += partial[t + step];
            }
        }
    };

    for (unsigned t = 1; t < T; ++t)
    {
        workers.emplace_back(worker_proc, t);
    }
    worker_proc(0);
    for (auto &w : workers)
    {
        w.join();
    }
    return partial[0] / n;
}

int main()
{
    size_t N = 1u << 25;

    auto buf = std::make_unique<double[]>(N);
    for (size_t i = 0; i < N; ++i)
        buf[i] = i;

    const char *p = "%s,%u,%f,%f,%f\n";

    std::vector<profiling_results_t> res;

    std::vector<std::pair<const char *, std::function<double(const double *v, size_t n)>>> funcs{
        {"omp", average_omp},
        {"rr", average_rr},
        {"reduce", average_reduce},
        {"omp_aligned", average_omp_align},
        {"cpp_aligned", average_cpp_partial_align},
        {"omp_mtx", average_omp_mtx_opt},
        {"cpp_mtx", average_cpp_mtx},
        {"cpp_mtx_local", average_mtx_local},
        {"cpp_reduction", average_cpp_reduce}};

    auto file = fopen("file1.csv", "w");
    fprintf(file, "method,number,result,speedup,efficiency\n");
    for (auto &f : funcs)
    {
        res = run_experiment(f.second, buf.get(), N);
        for (auto &i : res)
        {
            fprintf(file, p, f.first, i.T, i.result, i.speedup, i.efficiency);
        }
    }
    return 0;
}
