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
    def __init__(self, predict, attack_list=None, clip_min=0., clip_max=1.):
        self.attack_list = attack_list
        super(MultiOPAttack, self).__init__(
            predict=predict, loss_fn=None, clip_min=clip_min,clip_max=clip_max)

    '''
    Param: TODO
    '''
    def find_winner_loser(x, y, model, advx):
        winner_list = batch_attack_success_checker(model,x,y)
        loser_list = list(set(range(len(x))) - set(winner_list))
        loser_list.sort()
        loser_idx = torch.tensor(loser_list,dtype=torch.long)
        #也可以得到winner_idx 但是winner_idx的顺序是输入x里的idx 所以缺少一个映射
        x_loser = torch.index_select(x,dim=0,index=loser_idx)
        y_loser = torch.index_select(y,dim=0,index=loser_idx)
        return x_loser,y_loser, winner_list

    def perturb(self, x, y=None):
        x, y = self._verify_and_process_inputs(x, y)
        batch_size = len(x)
        loser_list = np.array(range(batch_size))# store not success attacked idx
        advx = torch.zeros_like(x)# store success attacked advx

        x_loser, y_loser, win = self.find_winner_loser(x,y,self.predict)
        next_loser = np.delete(loser_list,win) #delete idx of success advx in loser_list
        win_idx = list(set(loser_list)-set(next_loser))
        win_idx.sort()
        advx[win_idx] = x[win] #update success advx
        x, y, loser_list= x_loser, y_loser, next_loser

        for attack in self.attack_list:
            adv = attack.perturb(x,y)
            x_loser,y_loser,win = self.find_winner_loser(x,y,self.predict)

            if len(win):
                next_loser = np.delete(loser_list,win) #delete idx of success advx in loser_list
                win_idx = list(set(loser_list)-set(next_loser))
                win_idx.sort()
                advx[win_idx] = x[win] #update success advx
                x, y, loser_list= x_loser, y_loser, next_loser

            if len(x_loser)==0: break# if all data is attacked successfully, break
        return #rval.data