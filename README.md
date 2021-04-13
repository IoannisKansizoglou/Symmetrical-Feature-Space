# Symmetrical Feature Space

This repository contains the implementation of a hidden layer before the classification one that secures complete symmetry among the weight vectors during the entire training procedure. 

If you find this repository useful in your research, please consider citing:


    @article{kansizoglou2021do,
      title={Do Neural Network Weights account for Classes Centers?},
      author={Kansizoglou, Ioannis and Bampis, Loukas and Gasteratos, Antonios},
      journal={arXiv},
      year={2021}
    }

### Import Libraries

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from datasets import load_cifar10
from train import train_model
from model import resnet110

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)
```

### Load Dataset

```python
dataloaders, dataset_sizes = load_cifar10(batch_size=256)
```

### Setup Model

```python
net = resnet110()
net = net.to(device)
optimizer = optim.SGD(
    net.parameters(), 
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4)
```

### Train Model

```python
net = train_model(net,dataloaders,dataset_sizes,optimizer,device)
```


