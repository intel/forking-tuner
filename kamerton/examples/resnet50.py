#!/usr/bin/env python

"""
The ResNet50 example.

It will likely not converge due to the variance in clock-to-clock CPU usage.

You may wish to increase the number of predictions in the timeit loop, though
that will increase the runtime considerably.
"""

import numpy as np
import timeit

from kamerton import kamerton, set_threading


def main():
  # this will silence tensorflow's XLA compiler's informational messages
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # noqa

  # this needs to be imported after the tf's log level environ is set
  from tensorflow.keras.applications.resnet50 import ResNet50  # noqa

  for attempt in kamerton(set_threading, [22, 2], [11, 1],
                          threshold=0.02, iterations=10):
    # N.B. ResNet50 creation has to happen inside this loop because kamerton
    # sets the threading model and that has to happen before any tensors
    # are instantiated
    res = ResNet50()
    print(attempt)
    print(timeit.timeit(lambda: res.predict(np.random.rand(32, 224, 224, 3),
                                            32), number=1))


if __name__ == '__main__':
  main()