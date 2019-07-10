import sys, os
import torch as t
import data import get_data
from model import PoetryModel
from torch import nn
from utils import Visualizer
import tqdm
from torchnet import meter
import ipdb

class Config(object):
    data_path = ''
    pickle_path = 'tang.npz'
    author = None
    constrain = None
    category = 'poet.tang'
    lr = 1e-3
    weight_decay = 1e-4
    use_gpu = True
    epoch = 20
    batch_size = 128
    maxlen = 125
    plot_every = 20
    env = 'poetry'
    max_gen_len = 200
    debug_file = ''
    model_path = None
    prefix_words = '细雨鱼儿出,微风燕子斜。'
    start_words = '闲云潭影日悠悠'
    acrostic = False
    model_prefix = 'checkpoints/tang'

opt = Config()

def generate(model, start_words, ix2word, word2ix, prefix_words = None):
    results = list(start_words)
    start_word_len = len()

