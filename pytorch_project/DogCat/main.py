from config import opt
import os
import torch as t
import models
from data.dataset import DogCat
from torch.utils.data import DataLoader
from torchnet import meter
from utils.visualize import Visualizer
from tqdm import tqdm

@t.no_grad()

def test(**kwargs):
    opt._parse(kwargs)
    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device)
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device)

    train_data = DogCat(opt.test_data_root, test = True)
    test_dataloader = DataLpader(train_data, batch_size = opt.batch_size, shuffle = False, num_workers = opt.num_workers)
    results = []
    for ii, (data, path) in tqdm(enumerate(test_dataloader)):
        input = data.to(opt.device)
        score = model(input)
        probability = t.nn.functional.softmax(score, dim = 1)[:,0].detach().tolist()
        batch_results = [(path_.item(), probability_) for path_, probability_ in zip(path, probability)]
        results += batch_results
    write_csv(results, opt.results_file)
    return results


def write_csv(results, file_name):
    import csv
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(results)

def train(**kwargs):
    opt._parse(kwargs)
    vis = Visualizer(opt.env, port = opt.vis_port)
    print(opt.model)
    #print(getattr(models))
    #print(getattr(models, 'alexnet'))
    #model = getattr(models, opt.model)().eval()
    #model = models.SqueezeNet
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device)
    print(opt.train_data_root)
    train_data = DogCat(root = opt.train_data_root, train = True)
    val_data = DogCat(root = opt.train_data_root, train = False)
    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle = True, num_workers = opt.num_workers)
    val_dataloader = DataLoader(val_data, opt.batch_size, shuffle = False, num_workers = opt.num_workers)
    criterion = t.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = model.get_optimizer(lr, opt.weight_decay)
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e10

    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        confusion_matrix.reset()
        for ii, (data, label) in tqdm(enumerate(train_dataloader)):
            input = data.to(opt.device)
            target = label.to(opt.device)
            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()
            loss_meter.add(loss.item())
            confusion_matrix.add(score.detach(), target.detach())
            #if (ii + 1)%opt.print_freq == 0:
            #    vis.plot('loss', loss_meter.value()[0])
                #if os.path.exists(opt.debug_file):
                #    import ipdb;
                #    ipdb.set_trace()
    
    model.save()
    val_cm, val_accuracy = val(model, val_dataloader)
    vis.plot('val_accuracy', val_accuracy)
    vis.log('epoch:{epoch}, lr:{lr}, train_cm:{train_cm}, val_cm:{val_cm}'.format(
        epoch = epoch, loss = loss_meter.value()[0], val_cm = str(val_cm.value()),train_cm = str(confusion_matrix.value()), lr = lr
    ))
    if loss_meter.value()[0] > previous_loss:
        lr = lr * opt.lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    previous_loss = loss_meter.value()[0]

@t.no_grad()
def val(model, dataloader):
    model.eval()
    confusion_matrix = meter.ConfusionMeter(2)
    for ii, (val_input, label) in tqdm(enumerate(dataloader)):
        val_input = val_input.to(opt.device)
        score = model(val_input)
        confusion_matrix.add(score.detach().squeeze(), label.type(t.LongTensor))
    model.train()
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    return confusion_matrix, accuracy

def help():
    print("""
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:""".format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)

if __name__ == '__main__':
    import fire
    fire.Fire()

    


