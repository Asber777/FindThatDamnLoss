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

from advertorch.attacks.utils import batch_attack_success_checker

from .base import Attack
from .base import LabelMixin
from .utils import rand_init_delta



class MultiOPAttack(Attack, LabelMixin):
    """
    Embedding Attack ; First Test With PGD.
    """
    def __init__(self, predict, attack_list=[], eps=0.3, clip_min=0., clip_max=1.,device="cuda",norm="Linf"):
        super(MultiOPAttack, self).__init__(
            predict=predict, loss_fn=None, clip_min=clip_min,clip_max=clip_max)
        # init subattacker here
        self.targeted = False
        self.device = device
        self.eps = eps
        self.norm = norm
        assert len(attack_list) > 0
        self.attack_list = []
        for attack_class,attack_kwargs in attack_list:
            self.attack_list.append(attack_class(self.predict, **attack_kwargs))

    def update_advx(self, x, y, last_globalloser_idx, advx):
        localOKlist, localloserlist = batch_attack_success_checker(self.predict,x,y)
        localloser_idx = torch.tensor(localloserlist,dtype=torch.long).to(self.device)
        x_loser = torch.index_select(x,dim=0,index=localloser_idx)
        y_loser = torch.index_select(y,dim=0,index=localloser_idx)
        globalloser_idx = np.delete(last_globalloser_idx,localOKlist) 
        globalOKidx = sorted(list(set(last_globalloser_idx)-set(globalloser_idx)))
        advx[globalOKidx] = x[localOKlist] #update success advx
        assert len(globalloser_idx)==len(x_loser)
        return globalloser_idx, x_loser, y_loser
    
    def perturb(self, x, y=None):

        # first limit delta in [-eps,eps] then limit data in [clip_min,clip_max] 
        x, y = self._verify_and_process_inputs(x, y)
        x_oral = x.clone()
        globalloser_list = np.array(range(len(x)))# store not success attacked idx
        advx = torch.zeros_like(x)# store success attacked advx
        globalloser_list, x,y= self.update_advx(x,y,globalloser_list,advx)
        for attack in self.attack_list:
            adv = attack.perturb(x,y)
            #if per is out of eps than clipped (or rescaled which is not achieved)
            if self.norm=="Linf":
                per = batch_clamp(self.eps, adv - x_oral[globalloser_list])
                adv = clamp(x_oral[globalloser_list] + per,self.clip_min,self.clip_max)
            globalloser_list, x,y= self.update_advx(adv,y,globalloser_list,advx)
            if len(x)==0: break# if all data is attacked successfully, break
        # deal with loser_list data
        advx[globalloser_list] = x[:]
        return advx

    # def perturb(self, x, y=None):
    #     x, y = self._verify_and_process_inputs(x, y)
    #     advx = torch.zeros_like(x)
    #     for attack in self.attack_list:
    #         adv = attack.perturb(x,y)
    #     return adv