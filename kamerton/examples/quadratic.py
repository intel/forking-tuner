#!/usr/bin/env python

"""
A simple quadratic example that minimizes the `(x - 5) ** 2 + (x - 7) ** 2`
function.
"""

from kamerton import nelder_mead


# minimizes a quadratic function
def main():
  for attempt in nelder_mead(lambda x: None, [22, 2], [11, 1], threshold=1e-8):
    # the best attempt's simplex
    print(attempt)
    # the best attempt's objective value, will be close  to 0
    print((attempt[0] - 5) ** 2 + (attempt[1] - 7) ** 2)


if __name__ == '__main__':
  main()
