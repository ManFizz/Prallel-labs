#include <mutex>
#include <condition_variable>

class latch
{
private:
    unsigned T;
    std::mutex mtx;
    std::condition_variable cv;

public:
    void arrive_and_wait();
    latch(unsigned T);
};

class barrier
{
private:
    unsigned lock_id = 0;
    unsigned T, Tmax;
    std::mutex mtx;
    std::condition_variable cv;

public:
    barrier(unsigned threads);
    void arrive_and_wait();
};
