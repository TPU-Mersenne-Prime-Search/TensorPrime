def probable_prime(power):
    s = 3
    for i in range(power):
        s *= s
        s = s % ((1 << power) - 1)
    if s == 9:
        return True
    return False
