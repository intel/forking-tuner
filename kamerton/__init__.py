#!/usr/bin/env python

import os
import sys
from statistics import stdev
from typing import List, Callable, Tuple, Any, Generator
import logging


Callback = Callable[[List[float]], None]
SimplexWithObjectives = List[List[List[float]]]


logger = logging.getLogger(__name__)
_logger_handler = logging.StreamHandler()
_logger_handler.setLevel(logging.DEBUG)
logger.addHandler(_logger_handler)


def make_simplex(vertex: List[float],
                 step_sizes: Any = None) -> List[List[float]]:
  dim = len(vertex)
  step_sizes = step_sizes or [1 for i in range(dim)]
  assert dim == len(step_sizes)
  simplex = []
  # vertex + unit vector along each axis
  for i in range(dim):
    vert = [float(v) for v in vertex]
    vert[i] += step_sizes[i]
    simplex.append(vert)
  # and the original vertex
  simplex.append([float(v) for v in vertex])
  return simplex


def do_fork(cb: Callback, params: List[float]) -> Tuple[bool, Any]:
  r, w = os.pipe()
  pid = os.fork()

  # parent
  if pid > 0:
    os.close(w)
    read = os.fdopen(r)
    line = None
    for line in read:
      pass
    line = float(line.strip())
    return (True, line)

  # child
  os.close(r)
  write = os.fdopen(w, 'w')
  sys.stdout = write
  cb(params)
  return (False, None)


def centroid(simplex: SimplexWithObjectives) -> List[float]:
  sim = [s[1] for s in simplex]
  points_count = len(sim) - 1
  dim = len(sim[0])
  ret = [0.0] * points_count
  for i in range(points_count):
    for j in range(dim):
      ret[j] += sim[i][j]
  for j in range(dim):
    ret[j] /= points_count
  return ret


def reflect(simplex: SimplexWithObjectives, center: List[float]) -> List[float]:
  return [2 * i - j for i, j in zip(center, simplex[-1][1])]


def expand(reflected: List[float], center: List[float]) -> List[float]:
  return [i - j for i, j in zip(reflected, center)]


def contract(simplex: SimplexWithObjectives,
             center: List[float]) -> List[float]:
  return [0.5 * i + 0.5 * j for i, j in zip(center, simplex[-1][1])]


def shrink(simplex: SimplexWithObjectives) -> None:
  for i in range(1, len(simplex)):
    simplex[i][1] = [0.5 * i + 0.5 * j for i, j in
                     zip(simplex[0][1], simplex[i][1])]


def kamerton(cb: Callback, vertex: List[float],
             step_sizes: Any = None, iterations: int = 200,
             threshold: float = 1e-2) -> Generator:
  sim = make_simplex(vertex, step_sizes)
  # zip with objectives
  simplex = [list(i) for i in zip([0.0] * len(sim), sim)]  # type: List
  len_simplex = len(simplex)

  # compute initial vertices' objectives
  for index in range(len_simplex):
    is_parent, value = do_fork(cb, simplex[index][1])
    if not is_parent:
      yield simplex[index][1]
      raise StopIteration
    simplex[index][0] = value

  # https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method
  for _ in range(iterations):
    # 1. Order, check early termination
    simplex = sorted(simplex)
    logger.info('objectives, simplexes: ' + str(simplex))
    if stdev([s[0] for s in simplex]) < threshold:
      break

    # 2. Compute centroid
    center = centroid(simplex)

    # 3. Reflection
    reflected = reflect(simplex, center)
    is_parent, value = do_fork(cb, reflected)
    if not is_parent:
      yield reflected
      raise StopIteration
    if value > simplex[0][0] and value < simplex[1][0]:
      simplex[-1] = [value, reflected]
      continue

    # 4. Expansion
    if value < simplex[0][0]:
      expanded = expand(reflected, center)
      is_parent, value_expanded = do_fork(cb, expanded)
      if not is_parent:
        yield expanded
        raise StopIteration
      if value_expanded < value:
        simplex[-1] = [value_expanded, expanded]
        continue
      simplex[-1] = [value, reflected]
      continue

    # 5. Contraction
    contracted = contract(simplex, center)
    is_parent, value = do_fork(cb, contracted)
    if not is_parent:
      yield contracted
      raise StopIteration
    if value < simplex[-1][0]:
      simplex[-1] = [value, contracted]
      continue

    # 6. Shrink
    shrink(simplex)
    for i in range(1, len(simplex)):
      is_parent, value = do_fork(cb, simplex[i][1])
      if not is_parent:
        yield simplex[i][1]
        raise StopIteration
      simplex[i][0] = value

  # parent cleanup
  cb(simplex[0][1])
  yield simplex[0][1]
  raise StopIteration


def set_threading(params):
  import tensorflow as tf

  tf.config.threading.set_intra_op_parallelism_threads(
      max([int(params[0]), 1]))
  tf.config.threading.set_inter_op_parallelism_threads(
      max([int(params[1]), 1]))


def main():
  """
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # noqa

  import numpy as np
  from tensorflow.keras.applications.resnet50 import ResNet50
  import timeit

  for attempt in kamerton(set_threading, [22, 2], [11, 1],
                          threshold=0.02, iterations=10):
    res = ResNet50()
    print(attempt)
    print(timeit.timeit(
      lambda: res.predict(np.random.rand(32, 224, 224, 3), 32), number=10))
  """

  logger.setLevel(logging.INFO)
  for attempt in kamerton(lambda x: None, [22, 2], [11, 1], threshold=1e-8):
    print(attempt)
    print((attempt[0] - 5) ** 2 + (attempt[1] - 7) ** 2)


if __name__ == '__main__':
  main()
