#!/usr/bin/env python

"""
TensorFlow utility configuration callbacks for kamerton.
"""

from typing import List

import tensorflow as tf

__all__ = ['set_threading']


def set_threading(params: List[float]) -> None:
  """
  Sets the intra- and inter-op parallelism threads.
  """

  tf.config.threading.set_intra_op_parallelism_threads(
      max([int(params[0]), 1]))
  tf.config.threading.set_inter_op_parallelism_threads(
      max([int(params[1]), 1]))
