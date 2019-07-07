import warnings
import torch as t

class DefaultConfig(object):
    env = 'default'
    vis_port = 8097
    model = 'SqueezeNet'

    train_data_root = '/home/zhangp/Documents/data/DogCat/train/train/'
    test_data_root = '/home/zhangp/Documents/data/DogCat/test1/test1'
    load_model_path = None

    batch_size = 32
    use_gpu = True
    num_workers = 4
    num_workers = 4
    print_freq = 20

    debug_file = '/home/zhangp/Documents/data/DogCat/temp/'
    reslut_file = 'result.csv'

    max_epoch = 10
    lr = 0.001
    lr_decay = 0.5
    weight_decay = 0e-5

    def _parse(self, kwargs):
        for k,v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("waring: opt has not attribut %s" %k)
            setattr(self, k, v)
        opt.device = t.device('cuda') if opt.use_gpu else t.device('cpu')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))

opt = DefaultConfig()
