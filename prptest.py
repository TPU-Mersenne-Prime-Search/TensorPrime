import IBDWT as ibdwt

def probable_prime(power):
    s = 3
    for i in range(power):
        # s *= s
        # s = s % ((1 << power) - 1)
        s = ibdwt.squaremod_with_ibdwt(s)
    if s == 9:
        return True
    return False
