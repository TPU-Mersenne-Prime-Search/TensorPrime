# Naive implementation of the Lucas-Lehmer primality test
def naive_lucas_lehmer(power):
  if power == 2:
    return True
  s = 4
  for i in range(1, power-1):
    ns = ((s**2) - 2) % ((1 << power) - 1)
    s = ns
  return s == 0