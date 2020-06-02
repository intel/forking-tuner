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
