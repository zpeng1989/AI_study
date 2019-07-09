import torch as t
from torch.utils import data

import os
from PIL import Image
import torchvision as tv
import numpy as np

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def create_collate_fn(padding, eos, max_length = 50):
    def collate_fn(img_cap):
        img_cap.sort(key = lambda p: len(p[1]), reverse = True)
        imgs, caps, indexs = zip(*img_cap)
        imgs = t.cat([img.unsqueeze(0) for img in imgs], 0)
        lengths = [min(len(c) + 1, max_length) for c in caps]
        batch_length = max(lengths)
        cap_tensor = t.LongTensor(batch_length, len(caps)).fill_(padding)
        for i, c in enumerate(caps):
            end_cap = lengths[i] - 1
            if end_cap < batch_length:
                cap_tensor[end_cap, i] = eos
            cap_tensor[:end_cap, i].copy_(c[:end_cap])
        return (imgs, (cap_tensor, lengths), indexs)
    return collate_fn

class CaptionDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        data = t.load(opt.caption_data_path)
        word2ix = data['word2ix']
        self.captions = data['caption']
        self.padding = word2ix.get(data.get('padding'))
        self.end = word2ix.get(data.get('end'))
        self._data = data
        self.ix2id = data['ix2id']
        self.all_imgs = t.load(opt.img_feature_path)

    def __getitem(self, index):
        img = self.all_imgs[index]
        caption = self.captions[index]
        rdn_index = np.random.choice(len(caption),1)[0]
        caption = caption[rdn_index]
        return img, t.LongTensor(caption), index

    def __len__(self):
        return len(self,ix2id)


def get_dataloader(opt):
    dataset = CaptionDataset(opt)
    dataloader = data.DataLoader(dataset,
        batch_size = opt.batch_size, 
        shuffle = opt.shuffle,
        num_workers = opt.num_workers,
        collate_fn = create_collate_fn(dataset.padding, dataset.end)
    )
    return dataloader

if __name__ == '__main__':
    from config import Config
    opt = Config()
    dataloader = get_dataloader(opt)
    for ii, data in ensumerate(dataloader):
        print(ii, data)
        break