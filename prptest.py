import config
from gerbicz import update_gec_save, rollback
import IBDWT as ibdwt


def probable_prime(power):
    L = config.GEC_iterations
    L_2 = L*L

    s = 3
    for i in range(power):
        if config.GEC_enabled:
            # Every L iterations, update d and prev_d
            if i != 0 and i % L == 0:
                gerbicz.prev_d = d
                gerbicz.d = (d * s) % n
            # Every L^2 iterations, check the current d value with and independently calculated d
            if (i != 0 and i % L_2 == 0) or (i+ L > power):
                check_value = (3 * (gerbicz.prev_d ** (2 ** L))) % n
                if d != check_value:
                    i, s = rollback()
                else:
                    update_gec_save(i, s)
        # s *= s
        # s = s % ((1 << power) - 1)
        s = ibdwt.squaremod_with_ibdwt(s)
    if s == 9:
        return True
    return False
