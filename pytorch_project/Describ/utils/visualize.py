import visdom

import time
import torchvision as tv
import numpy as np

class Visualizer():
    def __init__(self, env = 'default', **kwargs):
        self.vis = visdom.Visdom(env = env, **kwargs)
        self.index = {}
        self.log_text = ''

    def reinit(self, env = 'default', **kwargs):
        self.vis = visdom.Visdom(env = env, **kwargs)
        return self

    def plot_mangy(self, d):
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y):
        

