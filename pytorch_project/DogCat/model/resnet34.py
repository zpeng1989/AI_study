from .basic_module import BasicModule
from torch import nn
from torch.nn import functional as F

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride = 1, shortcut = None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.COnv2d(inchannel, outchannel, 3, stride, 1, bias = False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace = True),
            nn.COnv2d(outchannel, outchannel, 3, 1, 1, bias = False),
            nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut

    