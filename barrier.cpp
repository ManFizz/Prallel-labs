#include "barrier.hpp"

latch::latch(unsigned T) : T(T) {}
void latch::arrive_and_wait()
{
    std::unique_lock l(mtx);
    if (--T)
        do
        {
            cv.wait(l);
        } while (T > 0);
    else
        cv.notify_all();
}

barrier::barrier(unsigned threads) : T(threads), Tmax(threads) {}
void barrier::arrive_and_wait()
{
    std::unique_lock l(mtx);
    if (--T)
    {
        unsigned my_lock_id = lock_id;
        while (my_lock_id == lock_id)
            cv.wait(l);
    }
    else
    {
        ++lock_id;
        T = Tmax;
        cv.notify_all();
    }
}
