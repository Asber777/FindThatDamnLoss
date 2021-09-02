import torch.nn as nn
from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import get_mnist_lenet5_clntrained
from advertorch_examples.utils import get_mnist_lenet5_advtrained
from advertorch_examples.benchmark_utils import get_benchmark_sys_info
from advertorch.attacks import MultiOPAttack
from advertorch_examples.benchmark_utils import benchmark_attack_success_rate

batch_size = 100
device = "cuda"

lst_attack = [
    (MultiOPAttack, dict(
        loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
        nb_iter=1000, eps_iter=0.01, rand_init=False,
        clip_min=0.0, clip_max=1.0)),
]  # each element in the list is the tuple (attack_class, attack_kwargs)


mnist_clntrained_model = get_mnist_lenet5_clntrained().to(device)
mnist_advtrained_model = get_mnist_lenet5_advtrained().to(device)
mnist_test_loader = get_mnist_test_loader(batch_size=batch_size)


lst_setting = [
    (mnist_clntrained_model, mnist_test_loader),
    # (mnist_advtrained_model, mnist_test_loader),
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
