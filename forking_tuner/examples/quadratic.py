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
A simple quadratic example that minimizes the `(x - 5) ** 2 + (x - 7) ** 2`
function and provides sample callback usage..
"""

from forking_tuner import nelder_mead


# minimizes a quadratic function
def main():
  # the callback will print the simplexes and their objectives at every step
  for attempt in nelder_mead([22, 2], [11, 1], threshold=1e-8, cb=print):
    # the best attempt's objective value, will be close  to 0
    print(f"Optimal parameters for (x - 5) ^ 2 + (y - 7) ^ 2 are: {attempt}.")
    print((attempt[0] - 5) ** 2 + (attempt[1] - 7) ** 2)


if __name__ == '__main__':
  main()
