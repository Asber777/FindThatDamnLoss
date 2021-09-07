
from numpy.lib.npyio import save
import torch.nn as nn
from advertorch.attacks import LinfPGDAttack
from advertorch.attacks.utils import multiple_mini_batch_attack
from advertorch_examples.utils import get_mnist_test_loader
loader = get_mnist_test_loader(batch_size=10)
from torchvision.utils import save_image
for data, label in loader:
    save_image(data,"../test.png",nrow=10)
    print(label)
    print(data)
    break

# Accuracy: 98.53%, Robust Accuracy: 92.51%
