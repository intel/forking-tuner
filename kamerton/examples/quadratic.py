#!/usr/bin/env python

"""
A simple quadratic example that minimizes the `(x - 5) ** 2 + (x - 7) ** 2`
function and provides sample callback usage..
"""

from kamerton import nelder_mead


# minimizes a quadratic function
def main():
  for attempt in nelder_mead([22, 2], [11, 1], threshold=1e-8, cb=print):
    # the best attempt's objective value, will be close  to 0
    print((attempt[0] - 5) ** 2 + (attempt[1] - 7) ** 2)


if __name__ == '__main__':
  main()
