import IBDWT as ibdwt
import time

def probable_prime(power):
    start = time.time()
    s = 3
    for i in range(power):
        # s *= s
        # s = s % ((1 << power) - 1)
        s = ibdwt.squaremod_with_ibdwt(s)
        if i % 100 == 0:
            time_elapsed = time.time() - start
            print("Time elapsed at iteration ", i, ": ", time_elapsed, ". S = ", s)
    if s == 9:
        return True
    return False
