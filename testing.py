# -*- coding: utf-8 -*-
#
# Copyright (c) 2016-2017 Intel Corporation.
#
# Redistribution.  Redistribution and use in binary form, without
# modification, are permitted provided that the following conditions are
# met:
#
# * Redistributions must reproduce the above copyright notice and the
#   following disclaimer in the documentation and/or other materials
#   provided with the distribution.
# * Neither the name of Intel Corporation nor the names of its suppliers
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
# * No reverse engineering, decompilation, or disassembly of this software
#   is permitted.
#
# Limited patent license.  Intel Corporation grants a world-wide,
# royalty-free, non-exclusive license under patents it now or hereafter
# owns or controls to make, have made, use, import, offer to sell and
# sell ("Utilize") this software, but solely to the extent that any
# such patent is necessary to Utilize the software alone, or in
# combination with an operating system licensed under an approved Open
# Source license as listed by the Open Source Initiative at
# http://opensource.org/licenses.  The patent license shall not apply to
# any other combinations which include this software.  No hardware per
# se is licensed hereunder.
#
# DISCLAIMER.  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,
# BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.
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
