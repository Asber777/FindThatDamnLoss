# Copyright (c) 2018-present, Royal Bank of Canada and other authors.
# See the AUTHORS.txt file for a list of contributors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from setuptools import setup
from setuptools import find_packages


with open(os.path.join(os.path.dirname(__file__), 'advertorch/VERSION')) as f:
    version = f.read().strip()


setup(name='advertorch',
      version=version,
      url='https://github.com/BorealisAI/advertorch',#TBM
      package_data={'advertorch_examples': ['*.ipynb', 'trained_models/*.pt']},
      install_requires=['gym'],# according to envs 
      include_package_data=True,
      packages=find_packages())
