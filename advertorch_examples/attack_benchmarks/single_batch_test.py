
# the code below is tested OK
from numpy.core.fromnumeric import clip
import torch.nn as nn
from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import get_mnist_lenet5_clntrained
from advertorch_examples.utils import get_mnist_lenet5_advtrained
from advertorch_examples.benchmark_utils import get_benchmark_sys_info
from advertorch.attacks import MultiOPAttack, TargetLinfPGDAttack, LinfPGDAttack
from advertorch_examples.benchmark_utils import benchmark_attack_success_rate


batch_size = 8*5
device = "cuda"
from advertorch.utils import predict_from_logits
from torchvision.utils import save_image

# model = get_mnist_lenet5_clntrained().to(device)
model = get_mnist_lenet5_advtrained().to(device)
loader = get_mnist_test_loader(batch_size=batch_size)
attack_list = [
    (LinfPGDAttack, dict(
        loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
        nb_iter=40, eps_iter=0.01, rand_init=False,
        clip_min=0.0, clip_max=1.0, targeted=False, early_stop=True)),
    (TargetLinfPGDAttack, dict(
        loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
        nb_iter=40, eps_iter=0.01, rand_init=False,
        clip_min=0.0, clip_max=1.0, early_stop=False, n=2)),
]
adversary = MultiOPAttack(model,attack_list)

device="cuda"
for data, label in loader:
    save_image(data,"oral.png")
    data, label = data.to(device), label.to(device)
    adv = adversary.perturb(data, label)
    advpred = predict_from_logits(adversary.predict(adv))
    pred = predict_from_logits(adversary.predict(data))
    print(advpred,pred)
    save_image(adv,"adv.png")
    break
