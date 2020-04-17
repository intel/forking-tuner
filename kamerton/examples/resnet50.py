#!/usr/bin/env python

"""
The ResNet50 example.

It will likely not converge due to the variance in clock-to-clock CPU usage.

You may wish to increase the number of predictions in the timeit loop, though
that will increase the runtime considerably.
"""

import logging
import os
import timeit
from collections import namedtuple

import numpy as np

from kamerton import nelder_mead, set_log_level
from kamerton.util import set_threading


def main():
  # this will silence tensorflow's XLA compiler's informational messages
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # noqa

  # this needs to be imported after the tf's log level environ is set
  from tensorflow.keras.applications.resnet50 import ResNet50  # noqa

  threading = namedtuple('threading', ['intra_op', 'inter_op'])

  print("This will take a while...")
  set_log_level(logging.INFO)
  for attempt in nelder_mead(threading(22, 2), [11, 1], threshold=0.02,
                             iterations=10):
    set_threading(attempt)
    # N.B. ResNet50 creation has to happen inside this loop because kamerton
    # sets the threading model and that has to happen before any tensors
    # are instantiated
    res = ResNet50()
    print(attempt)
    print(timeit.timeit(lambda: res.predict(np.random.rand(32, 224, 224, 3),
                                            32), number=1))


if __name__ == '__main__':
  main()
