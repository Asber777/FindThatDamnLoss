# Copyright (c) 2018-present, Royal Bank of Canada and other authors.
# See the AUTHORS.txt file for a list of contributors.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# This .py file is created by Asber, if you have any question, plz
# email xq.chen@siat.ac.cn

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
import torch.nn as nn

from advertorch.utils import clamp
from advertorch.utils import normalize_by_pnorm
from advertorch.utils import clamp_by_pnorm
from advertorch.utils import is_float_or_torch_tensor
from advertorch.utils import batch_multiply
from advertorch.utils import batch_clamp
from advertorch.utils import replicate_input
from advertorch.utils import batch_l1_proj
from advertorch.utils import topk_from_logits
from advertorch.utils import predict_from_logits

from .base import Attack
from .base import LabelMixin
from .utils import rand_init_delta



class MultiOPAttack(Attack, LabelMixin):
    """
    Embedding Attack ; First Test With PGD.
    """
    def __init__(self, predict, attack_list=None, clip_min=0., clip_max=1.):
        self.attack_list = attack_list
        super(MultiOPAttack, self).__init__(
            predict=predict, loss_fn=None, clip_min=clip_min,clip_max=clip_max)

    def perturb(self, x, y=None):
        x, y = self._verify_and_process_inputs(x, y)
        delta = torch.zeros_like(x)
        delta = nn.Parameter(delta) # make delta trainable.
        if self.rand_init:
            rand_init_delta(
                delta, x, self.ord, self.eps, self.clip_min, self.clip_max)
            delta.data = clamp(
                x + delta.data, min=self.clip_min, max=self.clip_max) - x
        pass
        return rval.data