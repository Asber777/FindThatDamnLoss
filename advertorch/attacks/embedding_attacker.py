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
    Embedding Attack with order=Linf; First Test With PGD.
    """
    def __init__(self, predict, attack_list=None, rand_init=True, clip_min=0., clip_max=1.):
        ord = np.inf
        self.attack_list = attack_list
        super(MultiOPLinfPGDAttack, self).__init__(
            predict=predict, loss_fn=loss_fn, eps=eps, nb_iter=nb_iter,
            eps_iter=eps_iter, rand_init=rand_init, clip_min=clip_min,
            clip_max=clip_max, ord=ord, early_stop=True)

    def perturb(self, x, y=None):
        x, y = self._verify_and_process_inputs(x, y)
        delta = torch.zeros_like(x)
        delta = nn.Parameter(delta) # make delta trainable.
        if self.rand_init:
            rand_init_delta(
                delta, x, self.ord, self.eps, self.clip_min, self.clip_max)
            delta.data = clamp(
                x + delta.data, min=self.clip_min, max=self.clip_max) - x
        nth_logit = topk_from_logits(self.predict(x), k=self.k)
        no_nb_iter = self.nb_iter//(self.k-1)
        for i in range(1,self.k):
            ith_y = nth_logit[:,i]
            rval = perturb_iterative(
                x, ith_y, self.predict, nb_iter=no_nb_iter,
                eps=self.eps, eps_iter=self.eps_iter,
                loss_fn=self.loss_fn, minimize=self.targeted,
                ord=self.ord, clip_min=self.clip_min,
                clip_max=self.clip_max, delta_init=delta,
                l1_sparsity=self.l1_sparsity,early_stop=self.early_stop
            )
        return rval.data