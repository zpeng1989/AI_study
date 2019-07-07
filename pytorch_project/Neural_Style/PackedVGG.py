import torch
import torch.nn as nn
from torchvision.models import vgg16
from collections import namedtuple


class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16)