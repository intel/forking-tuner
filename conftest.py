# -*- coding: utf-8 -*-
#
# Copyright (c) 2016-2019 Intel Corporation.
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

import logging

import pytest
from mock import MagicMock

from testing import patch_setattr, raise_unmocked


class FakeLogger(object):
  call_count = 0

  def __init__(self):
    self.__class__.call_count = 0

  def makeRecord(self, *argv, **kw):
    self.__class__.call_count += 1
    return MagicMock()


logging.getLoggerClass = MagicMock(return_value=FakeLogger)
logging.basicConfig = MagicMock()
logging.setLoggerClass = MagicMock()


MODULES = ('kamerton',)
MODULES_REPLACE = ('tests', 'kamerton')


@pytest.fixture
def patch(monkeypatch):
  def wrapper(path, mock=None):
    return patch_setattr(MODULES, MODULES_REPLACE, monkeypatch, path, mock)

  return wrapper


@pytest.fixture
def patch_session(monkeypatch):
  def wrapper(path=None):
    session = session_mock()
    patch_setattr(MODULES, MODULES_REPLACE, monkeypatch, path or 'Session',
                  MagicMock(return_value=session))
    return session

  return wrapper


@pytest.fixture
def patch_scoped_session(monkeypatch):
  def wrapper(path=None):
    session = session_mock()
    session.__enter__ = session
    patch_setattr(MODULES, MODULES_REPLACE, monkeypatch,
                  path or 'scoped_session', MagicMock(return_value=session))
    return session

  return wrapper
