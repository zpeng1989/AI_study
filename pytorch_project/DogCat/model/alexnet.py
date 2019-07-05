from torch import nn
from .basic_model import BasicModule


class AlexNet(BasicModule):
    def __init__(self, num_classes = 2):
        super(AlexNet, self).__init__()
        self.model_name = 'alexnet'
        self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size = 11, stride = 4, padding = 2),
                nn.ReLU(inplace = True),
                nn.MaxPool2d(kernel_size = 3, stride = 2),
                nn.Conv2d(64, 192, kernel_size = 5, padding = 2),
                nn.ReLU(inplace = True),
                nn.MaxPool2d(kernel_size = 3, stride = 2),
                nn.Conv2d(192, 384, kernel_size = 3, padding = 1),
                nn.ReLU(inplace = True),
                nn.Conv2d(384, 256, kernel_size = 3, padding = 1),
                nn.ReLU(inplace = True),
                nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
                nn.ReLU(inplace = True),
                nn.MaxPool2d(kernel_size = 3, stride = 2)
            )

