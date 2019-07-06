from itertools import chain
import visdom
import torch
import time
import torchvision as tv
import numpy as np

class Visualizer():
    def __init__(self, env = 'default', **kwargs):
        import visdom
        self.vis = visdom.Visdom(env = env, use_incoming_socket = False, **kwargs)
        self.index = {}
        self.log_text = ''
    def reinit(self, env = 'default', **kwargs):
        self.vis = visdom.Visdom(env = env, use_incoming_socket = False, **kwargs)
        return self
    
    def plot_many(self, d):
        for k, v in d.items():
            self.plot(k, v)
    
    def img_many(self, d):
        for k,v in d.items():
            self.img(k, v)

    def plot(self, name, y):
        
