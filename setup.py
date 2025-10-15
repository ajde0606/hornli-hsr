#!/usr/bin/env python

from setuptools import setup, find_packages, Extension

import numpy as np

scripts = []

ext_modules = []

setup(name="hsr",
      packages=find_packages(),
      scripts=scripts,
      ext_modules=ext_modules)
