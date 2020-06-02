import tensorflow as tf

from kamerton.tf import set_threading


def test_set_threading():
  set_threading([3, 4])
  assert tf.config.threading.get_intra_op_parallelism_threads() == 3
  assert tf.config.threading.get_inter_op_parallelism_threads() == 4
