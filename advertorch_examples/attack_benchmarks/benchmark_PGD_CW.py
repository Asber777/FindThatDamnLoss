'''
Test in 8/27 by changing loss function.
# Automatically generated benchmark report (screen print of running this file)
#
# sysname: Linux
# release: 5.4.0-80-generic
# version: #90~18.04.1-Ubuntu SMP Tue Jul 13 19:40:02 UTC 2021
# machine: x86_64
# python: 3.7.11
# torch: 1.9.0
# torchvision: 0.10.0
# advertorch: 0.2.4

# attack type: LinfPGDAttack
# attack kwargs: loss_fn=CrossEntropyLoss()
#                eps=0.3
#                nb_iter=40
#                eps_iter=0.01
#                rand_init=False
#                clip_min=0.0
#                clip_max=1.0
#                targeted=False
# data: mnist_test, 10000 samples
# model: MNIST LeNet5 standard training
# accuracy: 98.76%
# attack success rate: 100.0%

# attack type: LinfPGDAttack
# attack kwargs: loss_fn=CWLoss()
#                eps=0.3
#                nb_iter=40
#                eps_iter=0.01
#                rand_init=False
#                clip_min=0.0
#                clip_max=1.0
#                targeted=False
# data: mnist_test, 10000 samples
# model: MNIST LeNet5 standard training
# accuracy: 98.76%
# attack success rate: 100.0%

# attack type: LinfPGDAttack
# attack kwargs: loss_fn=ZeroOneLoss()
#                eps=0.3
#                nb_iter=40
#                eps_iter=0.01
#                rand_init=False
#                clip_min=0.0
#                clip_max=1.0
#                targeted=False
# data: mnist_test, 10000 samples
# model: MNIST LeNet5 standard training
# accuracy: 98.76%
# attack success rate: 100.0%

# attack type: LinfPGDAttack
# attack kwargs: loss_fn=LogitMarginLoss()
#                eps=0.3
#                nb_iter=40
#                eps_iter=0.01
#                rand_init=False
#                clip_min=0.0
#                clip_max=1.0
#                targeted=False
# data: mnist_test, 10000 samples
# model: MNIST LeNet5 standard training
# accuracy: 98.76%
# attack success rate: 100.0%

# attack type: LinfPGDAttack
# attack kwargs: loss_fn=SoftLogitMarginLoss()
#                eps=0.3
#                nb_iter=40
#                eps_iter=0.01
#                rand_init=False
#                clip_min=0.0
#                clip_max=1.0
#                targeted=False
# data: mnist_test, 10000 samples
# model: MNIST LeNet5 standard training
# accuracy: 98.76%
# attack success rate: 100.0%

# attack type: LinfPGDAttack
# attack kwargs: loss_fn=CrossEntropyLoss()
#                eps=0.3
#                nb_iter=40
#                eps_iter=0.01
#                rand_init=False
#                clip_min=0.0
#                clip_max=1.0
#                targeted=False
# data: mnist_test, 10000 samples
# model: MNIST LeNet 5 PGD training according to Madry et al. 2018
# accuracy: 98.64%
# attack success rate: 6.79%

# attack type: LinfPGDAttack
# attack kwargs: loss_fn=CWLoss()
#                eps=0.3
#                nb_iter=40
#                eps_iter=0.01
#                rand_init=False
#                clip_min=0.0
#                clip_max=1.0
#                targeted=False
# data: mnist_test, 10000 samples
# model: MNIST LeNet 5 PGD training according to Madry et al. 2018
# accuracy: 98.64%
# attack success rate: 6.7%

# attack type: LinfPGDAttack
# attack kwargs: loss_fn=ZeroOneLoss()
#                eps=0.3
#                nb_iter=40
#                eps_iter=0.01
#                rand_init=False
#                clip_min=0.0
#                clip_max=1.0
#                targeted=False
# data: mnist_test, 10000 samples
# model: MNIST LeNet 5 PGD training according to Madry et al. 2018
# accuracy: 98.64%
# attack success rate: 6.7%

# attack type: LinfPGDAttack
# attack kwargs: loss_fn=LogitMarginLoss()
#                eps=0.3
#                nb_iter=40
#                eps_iter=0.01
#                rand_init=False
#                clip_min=0.0
#                clip_max=1.0
#                targeted=False
# data: mnist_test, 10000 samples
# model: MNIST LeNet 5 PGD training according to Madry et al. 2018
# accuracy: 98.64%
# attack success rate: 6.7%

# attack type: LinfPGDAttack
# attack kwargs: loss_fn=SoftLogitMarginLoss()
#                eps=0.3
#                nb_iter=40
#                eps_iter=0.01
#                rand_init=False
#                clip_min=0.0
#                clip_max=1.0
#                targeted=False
# data: mnist_test, 10000 samples
# model: MNIST LeNet 5 PGD training according to Madry et al. 2018
# accuracy: 98.64%
# attack success rate: 6.79%


'''
import torch.nn as nn
from torch.nn.modules.loss import SoftMarginLoss
from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import get_mnist_lenet5_clntrained
from advertorch_examples.utils import get_mnist_lenet5_advtrained
from advertorch_examples.benchmark_utils import get_benchmark_sys_info
from advertorch.attacks import LinfPGDAttack
from advertorch.loss import CWLoss, ZeroOneLoss, LogitMarginLoss, SoftLogitMarginLoss
from advertorch_examples.benchmark_utils import benchmark_attack_success_rate

batch_size = 100
device = "cuda"
lst_attack = []

lst_attack = [
    (LinfPGDAttack, dict(
        loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
        nb_iter=40, eps_iter=0.01, rand_init=False,
        clip_min=0.0, clip_max=1.0, targeted=False)),
]  # each element in the list is the tuple (attack_class, attack_kwargs)

lst_attack.append(
    (LinfPGDAttack, dict(
        loss_fn=CWLoss(),eps=0.3,
        nb_iter=40, eps_iter=0.01, rand_init=False,
        clip_min=0.0, clip_max=1.0, targeted=False))
)

lst_attack.append(
    (LinfPGDAttack, dict(
        loss_fn=ZeroOneLoss(),eps=0.3,
        nb_iter=40, eps_iter=0.01, rand_init=False,
        clip_min=0.0, clip_max=1.0, targeted=False))
)

lst_attack.append(
    (LinfPGDAttack, dict(
        loss_fn=LogitMarginLoss(),eps=0.3,
        nb_iter=40, eps_iter=0.01, rand_init=False,
        clip_min=0.0, clip_max=1.0, targeted=False))
)

lst_attack.append(
    (LinfPGDAttack, dict(
        loss_fn=SoftLogitMarginLoss(),eps=0.3,
        nb_iter=40, eps_iter=0.01, rand_init=False,
        clip_min=0.0, clip_max=1.0, targeted=False))
)

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
