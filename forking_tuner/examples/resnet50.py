#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
The ResNet50 example.

It will likely not converge due to the variance in clock-to-clock CPU usage.

You may wish to increase the number of predictions in the timeit loop, though
that will increase the runtime considerably.
"""

import logging
import os
import sys
import timeit
from collections import namedtuple

try:
  import numpy as np
except ImportError:
  print("Please install numpy and tensorflow to run this example.")
  sys.exit(-1)


from forking_tuner import nelder_mead, set_log_level
from forking_tuner.tf import set_threading


def main():
  # this will silence tensorflow's XLA compiler's informational messages
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # noqa

  # this needs to be imported after the tf's log level environ is set
  try:
    from tensorflow.keras.applications.resnet50 import ResNet50  # noqa
  except ImportError:
    print("Please install tensorflow to run this example.")
    sys.exit(-1)

  threading = namedtuple('threading', ['intra_op', 'inter_op'])

  print("This will take a while...")
  set_log_level(logging.INFO)
  for attempt in nelder_mead(threading(22, 2), [11, 1], threshold=0.02,
                             iterations=10):
    set_threading(attempt)
    # N.B. ResNet50 creation has to happen inside this loop because
    # 'forking-tuner' sets the threading model and that has to happen before any
    # tensors are instantiated
    res = ResNet50()
    print(f"Optimal configuration: {int(attempt[0])} intra-op threads, "
          f"{int(attempt[1])} inter-op threads.")
    print(timeit.timeit(lambda: res.predict(np.random.rand(32, 224, 224, 3),
                                            32), number=1))


if __name__ == '__main__':
  main()
