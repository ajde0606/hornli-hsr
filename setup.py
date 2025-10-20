#!/usr/bin/env python

from setuptools import setup, find_packages, Extension

scripts = []

ext_modules = []

setup(name="hsr",
      packages=find_packages(),
      scripts=scripts,
      ext_modules=ext_modules,
      install_requires=[
          "numpy",
      ])
