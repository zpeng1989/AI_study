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


transforms = T.Compose([
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean = [.5, .5, .5], std = [.5, .5, .5])
])

class DogCat(data.Dataset):
    def __init__(self, root, transforms = None):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, img) for img in imgs]
        #self.transforms = transfroms
        self.transforms = transforms
    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = 0 if 'dog' in img_path.split('/')[-1] else 1
        data = Image.open(img_path)
        if self.transforms:
            data = self.transforms(data)
            #data = self.transforms(data)
        return data, label
    def __len__(self):
        return len(self.imgs)

#dataset = DogCat('./data/dogcat/', transforms = transform)
dataset = DogCat('./data/dogcat/', transforms = transforms)
img, label = dataset[0]
for img, label in dataset:
    print(img.size(), label)


from torchvision.datasets import ImageFolder
dataset = ImageFolder('data/dogcat_2/')

print(dataset.class_to_idx)

print(dataset.imgs)

#from torch.utils.data import DataLoader


normalize = T.Normalize(mean=[0.4, 0.4, 0.4], std=[0.2, 0.2, 0.2])
transform  = T.Compose([
                    T.RandomResizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize,
                    ])

dataset = ImageFolder('data/dogcat_2/', transform=transform)
to_img = T.ToPILImage()
to_img(dataset[0][0] * 0.2 + 0.4)

from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=3, shuffle = True, num_workers=0, drop_last=False)
dataiter = iter(dataloader)
imgs, labels = next(dataiter)
print(imgs.size())

class NewDogCat(DogCat):
    def __getitem__(self, index):
        try:
            return super(NewDogCat, self).__getitem__(index)
        except:
            return None, None

from torch.utils.data.dataloader import default_collate

def my_collate_fn(batch):
    batch = list(filter(lambda x:x[0] is not None, batch))
    if len(batch) == 0: return t.Tensor()
    return default_collate(batch)

dataset = NewDogCat('./data/dogcat_wrong/', transforms = transform)

#print(dataset[5])


dataloader = DataLoader(dataset, 2, collate_fn = my_collate_fn, num_workers = 1, shuffle = True)
for batch_datas, batch_labels in dataloader:
    print(batch_datas.size(), batch_labels.size())


dataset = DogCat('data/dogcat/', transforms = transform)
weights = [2 if label == 1 else 1 for data, label in dataset]
print(weights)

from torch.utils.data.sampler import WeightedRandomSampler
sampler = WeightedRandomSampler(weights, num_samples = 9, replacement = True)
dataloader = DataLoader(dataset, batch_size = 3, sampler = sampler)

for datas, labels in dataloader:
    print(labels.tolist())

from torchvision import models
from torch import nn

resnet34 = models.squeezenet1_1(pretrained = True, num_classes = 1000)

resnet34.fc = nn.Linear(512, 10)

from torchvision import datasets

dataset = datasets.MNIST('/home/zhangp/Documents/data/', download = True, train = False, transform = transform)

from torchvision import transforms
to_pil = transforms.ToPILImage()
#to_pil(t.randn(3, 64, 64)).show()


dataloader = DataLoader(dataset, shuffle = True, batch_size = 64)
from torchvision.utils import make_grid, save_image

dataiter = iter(dataloader)
img = make_grid(next(dataiter)[0], 8)
#to_img(img).show()

##mul GPU

a = t.Tensor(3, 4)
if t.cuda.is_available():
    a = a.cuda(1) 
    t.save(a,'a.pth')                                       
    b = t.load('a.pth')
    c = t.load('a.pth', map_location=lambda storage, loc: storage)
    d = t.load('a.pth', map_location={'cuda:1':'cuda:0'})















