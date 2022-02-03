import config

def rollback():
    i = 0

def update_save():
    i = 0
    
def probable_primeGEC(power):
    p = config.EXPONENT_TO_CHECK
    n = 2 ** p - 1
    s = 3

    # GEC setup
    L = config.CONSTANT
    d = 3

    for i in range(power):
        # Every L iterations, update d and prev_d
        if i != 0 and i % L == 0:
            prev_d = d
            d = (d * s) % n
        # Every L^2 iterations, check the current d value with and independently calculated d
        if (i != 0 and i % (L ** 2) == 0) or (i+ L > p):
            check_value = (3 * (prev_d ** (2 ** L))) % n
            if d != check_value:
                rollback()
            else:
                update_save()
        s *= s
        s = s % ((1 << power) - 1)

    if s == 9:
        return True
    return False