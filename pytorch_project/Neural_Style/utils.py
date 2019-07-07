from itertooks import chain
import visdom
import torch as t
import time
import torchvision as tv
import numpy as np

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)/(ch * h * w)
    return gram

class Visualizer():
    def __init__(self, env = 'default', **kwargs):
        import visdom
        self.vis = visdom.Visdom(env = env, use_incoming_socker = False, **kwargs)
        self.index = {}
        self.log_text = ''

    def reinit(self, env = 'default', **kwargs):
        self.vis = visdom.Visdom(env = env, use_incoming_socket = False, **kwargs)
        return self

        


