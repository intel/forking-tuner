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
