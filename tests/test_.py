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

import logging
from copy import deepcopy

from mock import MagicMock, sentinel, call
from pytest import fixture, mark

from kamerton import logger, nelder_mead, set_log_level
from kamerton import _make_simplex, _do_fork, _centroid, _reflect, _expand
from kamerton import _contract, _shrink


simplex = [[1, [2, 3]], [2, [4, 7]], [3, [9, 11]]]
centroid = [3.0, 5.0]
values = [[1.5], [0.5, 0.3], [0.5, 0.7], [2.5, 2.5], [2.5, 3.5, 1.9, 2.9]]


@fixture
def fork(patch):
  return patch(_do_fork)


@fixture
def os(patch):
  os_ = patch('os')
  os_.pipe.return_value = (MagicMock(), MagicMock())
  return os_


@fixture
def simp(patch):
  patch(_make_simplex).return_value = [sim[1] for sim in simplex]


@fixture
def stdev(patch):
  patch('stdev').side_effect = [5, 0]


def test_set_log_level():
  set_log_level(logging.DEBUG)
  assert logger.level == logging.DEBUG


def test_make_simplex():
  assert _make_simplex([3, 4]) == [[4, 4], [3, 5], [3, 4]]


def test_do_fork_parent(os):
  os.fdopen.return_value = ['a\n', 'b\n', '3\n']
  os.fork.return_value = 1
  assert _do_fork() == (True, 3.0)


def test_do_fork_child(patch, os):
  sys = patch('sys')
  os.fdopen.return_value = sentinel.write
  os.fork. return_value = 0
  assert _do_fork() == (False, None)
  assert sys.stdout == sentinel.write


def test_centroid():
  assert _centroid(simplex) == centroid


def test_reflect():
  assert _reflect(simplex, centroid) == [-3.0, -1.0]


def test_expand():
  assert _expand([2.0, 3.0], centroid) == [-1.0, -2.0]


def test_contract():
  assert _contract(simplex, centroid) == [6.0, 8.0]


def test_shrink():
  sim = deepcopy(simplex)
  _shrink(sim)
  assert sim[0] == simplex[0]
  assert sim[1] == [2, [3.0, 5.0]]
  assert sim[2] == [3, [5.5, 7.0]]


@mark.parametrize('value', values)
def test_nelder_mead_parent(patch, simp, stdev, fork, value):
  fork.side_effect = [(True, v) for v in [1, 2, 3, *value]]
  cb = MagicMock()
  for attempt in nelder_mead(sentinel.vertex, sentinel.step_sizes, cb=cb):
    pass
  fork.assert_has_calls([call()] * (len(value) + 3))


@mark.parametrize('value', values)
def test_nelder_mead_child(patch, simp, stdev, fork, value):
  fork.side_effect = [(True, v) for v in
                      [1, 2, 3, *value[:-1]]] + [(False, value[-1])]
  cb = MagicMock()
  for attempt in nelder_mead(sentinel.vertex, sentinel.step_sizes, cb=cb):
    pass
  fork.assert_has_calls([call()] * (len(value) + 3))


def test_nelder_mead_child_one(patch, simp, stdev, fork):
  fork.side_effect = [(False, 1)]
  cb = MagicMock()
  for attempt in nelder_mead(sentinel.vertex, sentinel.step_sizes, cb=cb):
    pass
  fork.assert_has_calls([call()])
