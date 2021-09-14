# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import os

with open(os.path.join(os.path.dirname(__file__), 'VERSION')) as f:
    __version__ = f.read().strip()

from . import attacks  # noqa: F401
from . import defenses  # noqa: F401


# FOR GYM ENV
import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='attack-v0',
    entry_point='advertorch.envs:AttackEnv',
    # timestep_limit=1000,#?
    # reward_threshold=1.0,
    # nondeterministic = True,# TBD
)