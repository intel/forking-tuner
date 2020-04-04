#!/usr/bin/env python
# coding: utf-8

from setuptools import setup, find_packages


NAME = "kamerton"
VERSION = "0.1.0"


setup(
  name=NAME,
  version=VERSION,
  description="The forking tuner for tensorflow.",
  author_email="igor.kaplounenko@intel.com",
  url="",
  keywords=["tensorflow"],
  packages=find_packages(),
  long_description=""
)
