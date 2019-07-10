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
    input = t.Tensor([word2ix['<START>']]).view(1, 1).long()
    if opt.use_gpu: input = input.cuda()
    hidden = None

    if prefix_words:
        for word in prefix_words:
            output, hidden = model(input, hidden)
            input = input.data.new([word2ix[word]]).view(1,1)
    
    for i in range(opt.max_gen_len):
        output,hidden = model(input, hidden)

        if i < start_word_len:
            w = results[i]
            input = input.data.new([word2ix[w]]).view(1,1)
        else:
            top_index = output.data[0].topk(1)[1][0].item()
            w = ix2word[top_index]
            results.append(w)
            input = input.data.new([top_index]).view(1,1)
        if w == '<EOP>':
            del results[-1]
            break
    return results

def gen_acrostic(model, start_words, ix2word, word2ix, prefix_words = None):
    results = []
    start_word_len = len(start_words)
    input = (t.Tensor([word2ix['<START>']]).view(1,1).long())
    if opt.use_gpu: input = input.cuda()
    hidden = None

    index = 0
    pre_word = '<START>'

    if prefix_words:
        for word in prefix_words:
            output, hidden = model(input, hidden)
            input = (input.data.new([word2ix[word]])).view(1,1)
    for i in range(opt.max_gen_len):
        output, hidden = model(input,hidden)
        top_index = output.data[0].topk(1)[1][0].item()
        w = ix2word[top_index]

        if (pre_word in {u'。', u'！','<START>'}):
            if index == start_word_len:
                break
            else:
                w = start_words[index]
                index += 1
                input = (input.data.new([word2ix[w]])).view(1,1)
        else:
            input = (input.data.new([word2ix[w]])).view(1,1)
        results.append(w)
        pre_word = w

    return results

    






