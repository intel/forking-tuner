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
Kamerton: the forking tuner.
"""

import os
import sys
from statistics import stdev
from typing import List, Callable, Tuple, Any, Generator, Sequence, Optional
import logging

__all__ = ['logger', 'nelder_mead', 'set_log_level']


Callback = Callable[[List[List[Any]]], None]
SimplexWithObjectives = List[List[List[float]]]


logger = logging.getLogger(__name__)
_logger_handler = logging.StreamHandler()
_logger_handler.setLevel(logging.DEBUG)
logger.addHandler(_logger_handler)


def _make_simplex(vertex: Sequence,
                  step_sizes: Optional[List[int]] = None) -> List[List[float]]:
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


def _do_fork() -> Tuple[bool, Any]:
  r, w = os.pipe()
  pid = os.fork()

  # parent
  if pid > 0:
    os.close(w)
    read = os.fdopen(r)
    line = None
    for line in read:
      pass
    return (True, float(line.strip()))

  # child
  os.close(r)
  write = os.fdopen(w, 'w')
  sys.stdout = write
  return (False, None)


def _centroid(simplex: SimplexWithObjectives) -> List[float]:
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


def _reflect(simplex: SimplexWithObjectives,
             center: List[float]) -> List[float]:
  return [2 * i - j for i, j in zip(center, simplex[-1][1])]


def _expand(reflected: List[float], center: List[float]) -> List[float]:
  return [i - j for i, j in zip(reflected, center)]


def _contract(simplex: SimplexWithObjectives,
              center: List[float]) -> List[float]:
  return [0.5 * i + 0.5 * j for i, j in zip(center, simplex[-1][1])]


def _shrink(simplex: SimplexWithObjectives) -> None:
  for i in range(1, len(simplex)):
    simplex[i][1] = [0.5 * i + 0.5 * j for i, j in
                     zip(simplex[0][1], simplex[i][1])]


def nelder_mead(vertex: Sequence, step_sizes: Optional[List[int]] = None,
                iterations: int = 200, threshold: float = 1e-2,
                cb: Optional[Callback] = None) -> Generator:
  """
  The Nelder-Mead-based forking tuner.  See the project README.md or
  `kamerton.examples` for details.
  """
  VertexType = type(vertex)  # type: Any
  try:
    VertexType = VertexType._make
  except AttributeError:
    pass
  sim = _make_simplex(vertex, step_sizes)
  # zip with objectives
  simplex = [list(i) for i in zip([0.0] * len(sim), sim)]  # type: List
  len_simplex = len(simplex)

  # compute initial vertices' objectives
  for index in range(len_simplex):
    is_parent, value = _do_fork()
    if not is_parent:
      yield VertexType(simplex[index][1])
      return
    simplex[index][0] = value

  # https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method
  for _ in range(iterations):
    # 1. Order, check early termination
    simplex = sorted(simplex)
    typed_simplex = [[VertexType(v), o] for o, v in simplex]
    if cb is not None:
      cb(typed_simplex)
    logger.info('simplexes with objectives:')
    for vertex, objective in typed_simplex:
      logger.info(f'\t{vertex}: {objective}')
    if stdev([s[0] for s in simplex]) < threshold:
      break

    # 2. Compute centroid
    center = _centroid(simplex)

    # 3. Reflection
    reflected = _reflect(simplex, center)
    is_parent, value = _do_fork()
    if not is_parent:
      yield VertexType(reflected)
      return
    if value > simplex[0][0] and value < simplex[1][0]:
      simplex[-1] = [value, reflected]
      continue

    # 4. Expansion
    if value < simplex[0][0]:
      expanded = _expand(reflected, center)
      is_parent, value_expanded = _do_fork()
      if not is_parent:
        yield VertexType(expanded)
        return
      if value_expanded < value:
        simplex[-1] = [value_expanded, expanded]
        continue
      simplex[-1] = [value, reflected]
      continue

    # 5. Contraction
    contracted = _contract(simplex, center)
    is_parent, value = _do_fork()
    if not is_parent:
      yield VertexType(contracted)
      return
    if value < simplex[-1][0]:
      simplex[-1] = [value, contracted]
      continue

    # 6. Shrink
    _shrink(simplex)
    for i in range(1, len(simplex)):
      is_parent, value = _do_fork()
      if not is_parent:
        yield VertexType(simplex[i][1])
        return
      simplex[i][0] = value

  # parent cleanup
  yield VertexType(simplex[0][1])
  return


def set_log_level(level):
  """
  Sets the kamerton log level.  Try `logging.INFO` for detailed iteration logs.
  """
  logger.setLevel(level)
