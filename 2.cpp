#include <concepts>
#include <iostream>
#include <cstdint>
#include <memory>
#include <vector>
#include <thread>
#include <mutex>
#include <ranges>
#include <cmath>
#include <algorithm>

// Рандомизация данных с помощью
// Линейного конгруэнтного генератора
// Рекуррентных афиинных преобразований

unsigned get_num_thread();
void set_num_threads(unsigned T);

template <class T, std::unsigned_integral U>
auto my_pow(T x, U n)
{
    T r = T(1);
    while (n > 0)
    {
        if (n & 1)
        {
            r *= x;
        }
        x *= x;
        n >>= 1;
    }
    return r;
}

static const uint32_t A = 22695477u;
static const uint32_t B = 1u;

class lc_t
{
    uint32_t A, B;

public:
    lc_t(uint32_t a = 1, uint32_t b = 0)
    {
        this->A = a;
        this->B = b;
    }

    lc_t &operator*=(const lc_t &r)
    {
        this->B += this->A * r.B;
        this->A *= r.A;

        return *this;
    }

    auto operator()(uint32_t seed) const
    {
        return this->A * seed + this->B;
    }

    auto operator()(uint32_t seed, uint32_t min, uint32_t max) const
    {
        if (max == min - 1)
        {
            return this->operator()(seed);
        }

        return (*this)(seed) % (max - min) + min;
    }
};

double randomize_vector(uint32_t *V, size_t n, uint32_t seed,
                        uint32_t min_val = 0, uint32_t max_val = UINT32_MAX)
{
    double res = 0.0;
    if (min_val > max_val)
    {
        exit(__LINE__);
    }

    lc_t g(A, B);
    lc_t g_(A, B);

    for (size_t i = 0; i < n; i++)
    {
        g *= g_;
        V[i] = g(seed, min_val, max_val);
        res += V[i];
    }
    return res / n;
}

double randomize_vector_par(uint32_t *V, size_t n, uint32_t seed,
                            uint32_t min_val = 0, uint32_t max_val = UINT32_MAX)
{
    double res = 0.0;

    std::vector<std::thread> workers;
    std::mutex mtx;
    unsigned T = get_num_thread();
    size_t e = n / T;
    size_t b = n % T;
    auto workers_proc = [V, n, T, seed, min_val, max_val, &res, &mtx](unsigned t, size_t e, size_t b)
    {
        double partial = 0.0;

        if (t < b)
            b = t * ++e;
        else
            b += t * e;
        e += b;

        lc_t g_ = lc_t(A, B);
        lc_t g = my_pow(g_, b + 1);
        for (size_t i = b; i < e; ++i)
        {
            g *= g_;
            V[i] = g(seed, min_val, max_val);
            partial += V[i];
        }
        std::scoped_lock l{mtx};
        res += partial;
    };
    for (unsigned t = 1; t < T; ++t)
        workers.emplace_back(workers_proc, t, e, b);
    workers_proc(0, e, b);
    for (auto &worker : workers)
        worker.join();
    return res / n;
}

bool randomize_test(size_t n)
{
    const size_t seed = time(NULL);

    std::vector<uint32_t> v1(n), v2(n);
    double avg1 = randomize_vector(v1.data(), n, seed);
    double avg2 = randomize_vector_par(v2.data(), n, seed);
    if (floor(avg1) != floor(avg2))
        return false;
    auto pr = std::ranges::mismatch(v1, v2);
    return pr.in1 == v1.end() && pr.in2 == v2.end();
}

typedef struct profiling_results_t
{
    bool result;
    double time, speedup, efficiency;
    unsigned T;
} profiling_results_t;

auto run_experiment(size_t n)
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
        rr.result = randomize_test(n);
        auto t2 = steady_clock::now();
        rr.time = duration_cast<milliseconds>(t2 - t1).count();
        rr.speedup = res_table.front().time / rr.time;
        rr.efficiency = rr.speedup / T;
        rr.T = T;
    }
    return res_table;
}

int main()
{
    size_t n = 1u << 25;
    const char *p = "%u,%u,%f,%f\n";

    auto res = run_experiment(n);

    auto file = fopen("file2.csv", "w");
    fprintf(file, "number,result,speedup,efficiency\n");

    for (auto &i : res)
    {
        fprintf(file, p, i.T, i.result, i.speedup, i.efficiency);
    }

    fclose(file);
}
