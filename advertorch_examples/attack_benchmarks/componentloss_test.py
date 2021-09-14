from torch.nn.modules.loss import SoftMarginLoss
from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import get_mnist_lenet5_clntrained
from advertorch_examples.utils import get_mnist_lenet5_advtrained
from advertorch_examples.benchmark_utils import get_benchmark_sys_info
import torch.nn as nn
from advertorch.loss import CompositeLoss
from advertorch.oplib import getMNISTop
from advertorch.attacks import LinfPGDAttack

from advertorch_examples.benchmark_utils import benchmark_attack_success_rate
import torch as t

batch_size = 100
device = "cuda"

v2vlist,T2Tlist,tv2vlist,T2vlist,op = getMNISTop()
# Construct CE loss CW loss DLR loss by CompositeLoss.
comloss = CompositeLoss(T2Tlist,T2vlist,v2vlist,tv2vlist,K=2,M=1,N=0,reduction="sum")
'''
Z ->     Remain(Z):Z    -> GetLabellogit(Z):z_y -> reverse(z_y):-z_y \
Z -> exponential(Z):e^Z -> Sum(e^Z):sumeZ       -> logarithm(sumeZ)  ->addtensor(-z_y,logSumEZ) ->CELoss
'''
CEloss = op['Remain']+op['exponential']+op['GetLabellogit']+op['Sum']+op['reverse']+op['logarithm']+op['addtensor']
print(CEloss)

comloss.getLoss(CEloss)
comloss.visualization()
# CE(x,y) = -\log p_y = -z_y + \log(\sum^K_{j=1}e^{z_j})    
# CW(x, y) = −z_y + \max_{i\neq y} z_i
# DLR(x,y) = - \frac{z_y - \max_{i\neq y} z_i} {z_{\pi_1}-z_{\pi_3}}
lst_attack = [
    (LinfPGDAttack, dict(
        loss_fn=comloss, eps=0.3,
        nb_iter=40, eps_iter=0.01, rand_init=False,
        clip_min=0.0, clip_max=1.0, targeted=False)),
]  # each element in the list is the tuple (attack_class, attack_kwargs)

mnist_clntrained_model = get_mnist_lenet5_clntrained().to(device)
mnist_advtrained_model = get_mnist_lenet5_advtrained().to(device)
mnist_test_loader = get_mnist_test_loader(batch_size=batch_size)

lst_setting = [
    (mnist_clntrained_model, mnist_test_loader),
    (mnist_advtrained_model, mnist_test_loader),
]

info = get_benchmark_sys_info()

lst_benchmark = []
for model, loader in lst_setting:
    for attack_class, attack_kwargs in lst_attack:
        lst_benchmark.append(benchmark_attack_success_rate(
            model, loader, attack_class, attack_kwargs, device="cuda"))

print(info)
for item in lst_benchmark:
    print(item)
