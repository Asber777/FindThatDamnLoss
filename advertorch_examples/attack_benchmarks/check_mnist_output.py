
import torch.nn as nn
from advertorch.attacks import LinfPGDAttack
from advertorch.attacks.utils import multiple_mini_batch_attack
from advertorch_examples.utils import get_mnist_test_loader
loader = get_mnist_test_loader(batch_size=100)

for data, label in loader:
    data, label = data.to("cuda"), label.to("cuda")
    print(label)
    print(data)
    break

# Accuracy: 98.53%, Robust Accuracy: 92.51%
