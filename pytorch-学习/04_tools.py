import torch as t
from torch.utils import data

import os
from PIL import Image
import numpy as np

class DogCat(data.Dataset):
    def __init__(self, root):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, img) for img in imgs]

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = 1 if 'dog' in img_path.split('/')[-1] else 0
        pil_img = Image.open(img_path)
        array = np.asarray(pil_img)
        data = t.from_numpy(array)
        return data, label
    def __len__(self):
        return len(self.imgs)


dataset = DogCat('./data/dogcat/')
img, label = dataset[0]
for img, label in dataset:
    print(img.size(), img.float().mean(), label)

import os
from PIL import Image
import numpy as np
from torchvision import transforms as T

transform = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.Normalize(mean = [.5, .5, .5], std = [.5, .5, .5])
])

class DogCat(data.Dataset):
    def __init__(self, root, transfroms = None):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, img) for img in imgs]
        self.transforms = transfroms
    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = 0 if 'dog' in img_path.split('/')[-1] else 1
        data = Image.open(img_path)
        if self.transforms:
            data = self.transforms(data)
        return data, label
    def __len__(self):
        return len(self.imgs)

dataset = DogCat('./data/dogcat/', transforms = transform)
img, label = dataset[0]
for img, label in dataset:
    print(img.size(), label)


from torchvision.datasets import ImageFolder
dataset = ImageFolder('data/dogcat_2/')

print(dataset.class_to_idx)

print(dataset.imgs)

from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=3, shuffle = True, num_workers=0, drop_last=False)

dataiter = iter(dataloader)
imgs, labels = next(dataiter)
print(imgs.size())





