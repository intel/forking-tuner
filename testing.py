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

import os
import inspect

from mock import MagicMock


def raise_unmocked(name):
  def wrapper(*args, **kw):
    raise RuntimeError("using unmocked " + name + "!")  # pragma: no cover

  return wrapper


# do not call this directly -- assumes the fixture's caller is two stacks up
# and will correspondingly guess the module path to patch
def patch_setattr(module_names, module_replace, monkeypatch, path, m=None):
  """
  `path` can be:
      1. an object, if it's defined in the module you're testing
      2. a name, if it's imported in the module you're testing
      3. a full path a la traditional monkeypatch
  """
  m = m if m is not None else MagicMock()
  if hasattr(path, '__module__'):
    # object
    monkeypatch.setattr('.'.join((path.__module__, path.__name__)), m)
    return m
  if any(path.startswith(i + '.') for i in module_names):
    # full path.  OK.
    monkeypatch.setattr(path, m)
    return m
  # assume we're patching stuff in the file the test file is supposed to
  # be testing
  fn = inspect.getouterframes(inspect.currentframe())[2][1]
  fn = os.path.splitext(os.path.relpath(fn))[0]
  module = None
  if not fn.endswith('test_'):
    module = fn.replace(os.path.sep, '.').replace('test_', '')\
               .replace(*module_replace)
  else:
    module = '.'.join(fn.split(os.path.sep)[:-1])
    module = module.replace(*module_replace)
  try:
    monkeypatch.setattr('.'.join((module, path)), m)
  except AttributeError:
    # maybe it's a builtin?
    monkeypatch.setattr("{}.{}".format('builtins', path), m)
  return m
