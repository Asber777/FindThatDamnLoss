from numpy.core.fromnumeric import clip
import torch.nn as nn
from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import get_mnist_lenet5_clntrained
from advertorch_examples.utils import get_mnist_lenet5_advtrained
from advertorch_examples.benchmark_utils import get_benchmark_sys_info
from advertorch.attacks import MultiOPAttack, TargetLinfPGDAttack, LinfPGDAttack
from advertorch_examples.benchmark_utils import benchmark_attack_success_rate, benchmark_margin

batch_size = 100
device = "cuda"

sub_attack = [
    (LinfPGDAttack, dict(
        loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
        nb_iter=40, eps_iter=0.01, rand_init=False,
        clip_min=0.0, clip_max=1.0, targeted=False, early_stop=False)),
    (TargetLinfPGDAttack, dict(
        loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
        nb_iter=40, eps_iter=0.01, rand_init=False,
        clip_min=0.0, clip_max=1.0, early_stop=False, n=10)),
    # (TargetLinfPGDAttack, dict(
    #     loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
    #     nb_iter=40, eps_iter=0.01, rand_init=False,
    #     clip_min=0.0, clip_max=1.0, early_stop=False, n=3)),
    # (TargetLinfPGDAttack, dict(
    #     loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
    #     nb_iter=40, eps_iter=0.01, rand_init=False,
    #     clip_min=0.0, clip_max=1.0, early_stop=False, n=4)),
    # (TargetLinfPGDAttack, dict(
    #     loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
    #     nb_iter=40, eps_iter=0.01, rand_init=False,
    #     clip_min=0.0, clip_max=1.0, early_stop=False, n=5)),
    # (TargetLinfPGDAttack, dict(
    #     loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
    #     nb_iter=40, eps_iter=0.01, rand_init=False,
    #     clip_min=0.0, clip_max=1.0, early_stop=True, n=6)),
]  # each element in the list is the tuple (attack_class, attack_kwargs)

lst_attack = [(MultiOPAttack, dict(attack_list=sub_attack, clip_min=0.0, clip_max=1.0,device=device)),]

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
        lst_benchmark.append(benchmark_margin(
            # model, loader, attack_class, attack_kwargs, device="cuda",save_adv=True,num_batch=1,norm="Linf")) #for test
            model, loader, attack_class, attack_kwargs, device="cuda",norm="Linf")) #for all

print(info)
for item in lst_benchmark:
    print(item)

