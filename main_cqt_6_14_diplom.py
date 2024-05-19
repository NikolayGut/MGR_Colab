import argparse
import os
import shutil
import time
import random
import pickle
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchview

from model_cqt_6_14 import model_audio
from dataloader_cqt_6_14 import datatype
from config import Config
from center_loss import CenterLoss
from torchviz import make_dot


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test-batch', default=120, type=int, metavar='N',
                    help='test batchsize (default: 200)')
check = './checkpoint/checkpoint.pth.tar'
parser.add_argument('-c', '--checkpoint', default='', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg16',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('--lr', '--learning-rate', default=0.004, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--exponential', type=float, default=0.94, help='LR is multiplied by gamma on schedule.')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
#state['lr'] = Config["normal_config"]["initial_lr"]
ff = Config["normal_config"]['fold_index']
fold = ff# 1#3#2

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class DictMeter(object):

    def __init__(self, classes):
        self.classes = classes
        self.reset()

    def reset(self):
        self.dict_my = dict()
        for i in self.classes:
            self.dict_my[i] = dict()

    def update(self, dict_test):
        for genre,value in dict_test.items():
            for musicName,b in value.items():
                self.dict_my[genre][musicName] = dict_test[genre][musicName]

def main():
    global best_acc
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    opt = args
    args.lr = Config["normal_config"]["initial_lr"]
    checkpoint_dir = Config["normal_config"]["checkpoint"]
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    train_dataloader = datatype().instance_a_loader(t='train')
    test_dataloader = datatype().instance_a_loader(t='test')
    #test_dataloader = train_dataloader

    model = model_audio()
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    criterion = nn.CrossEntropyLoss().cuda()
    criterion_cent = CenterLoss(num_classes=8, feat_dim=1024, use_gpu=True)
    
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.9, eps=1.0, weight_decay=0.00004, momentum=0.9,
                              centered=False)
    optimizer_cent = optim.RMSprop(model.parameters(), lr=args.lr)

    best_acc = 63
    start_epoch = 0

    #if args.resume:
    checkpoint = torch.load(Config["cqt_6_14"]["cqt_model_best"].format(ff))
    start_epoch = checkpoint['epoch'] + 1
    best_acc = checkpoint['acc']
    #best_acc = 86
    print('==> Resuming from checkpoint..(trained from {} epochs,lr:{}\tacc:{})'.format(start_epoch-1,checkpoint['lr'],best_acc))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_acc = 64
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    #if args.evaluate:
    print('\nEvaluation only')
    test_loss, test_acc, test_dict = test(test_dataloader, model, criterion, criterion_cent, start_epoch, use_cuda, 'test_dict_cqt.csv')
    train_loss, train_acc, train_dict = test(train_dataloader, model, criterion, criterion_cent, start_epoch, use_cuda, 'train_dict_cqt.csv')
    print('test_loss:{}, test_acc:{}'.format(test_loss, test_acc))
    print('train_loss:{}, train_acc:{}'.format(train_loss, train_acc))
    #return
    epoches = Config["normal_config"]["epoch_num"]

    # Создайте пример входных данных
    example_input = torch.randn(1, 1, 32, 32)

    # После обучения модели и перед завершением функции main() добавьте следующий код:
    graph = torchview.draw_graph(model, example_input)  
    torchview.draw_graph_to_file(graph, "neural_network_graph_1.png")  # Сохранение схемы в файл

    print("Схема нейронной сети сохранена в файл neural_network_graph.png")



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(train_loader, model, criterion, criterion_cent, optimizer, optimizer_cent, epoch, use_cuda):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    for batch_idx, ss  in enumerate(train_loader):
        melspectrogram  = ss[1].unsqueeze(1)
        path = ss[0]
        targets = torch.LongTensor(ss[2])
        inputs = melspectrogram
        batch_size1 = inputs.shape[0]

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        optimizer.zero_grad()
        optimizer_cent.zero_grad()

        outputs, features = model(inputs)
        loss_soft = criterion(outputs, targets)
        loss_cent = criterion_cent(features, targets)
        loss = loss_soft + 1e-4*loss_cent

        
        prec1 = accuracy(outputs.data, targets.data, topk=(1,))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0].item(), inputs.size(0))

        
        line = 'train:{}\tbatch:{}\ttop1:{}\tloss:{}\tlr:{}\n'.format(batch_idx,
                Config["normal_config"]["batch_size"],top1.avg,losses.avg,state['lr'])
        
        if batch_idx%50==0:
            print('train:{}/top1:{}/loss:{}'.format(batch_idx,top1.avg,losses.avg))
        loss.backward()
        optimizer.step()

        for param in criterion_cent.parameters():
            param.grad.data *= (1. / 1e-4)
        optimizer_cent.step()

    return (losses.avg, top1.avg)

def feat_to_dict(output, path,dict_test):
    nb_data = output.shape[0]

    for j in range(nb_data):
        genre,musicName, seg = path[j].split('-')

        if musicName not in dict_test[genre].keys():
            dict_test[genre][musicName] = output[j,:]
        else:
            dict_test[genre][musicName] += output[j,:]


def output_to_dict(output, path,dict_test):
    nb_data = output.shape[0]
    classes = Config["normal_config"]["classes"]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    for j in range(nb_data):
        genre,musicName, seg = path[j].split('-')
        
        if musicName not in dict_test[genre].keys():
            dict_test[genre][musicName] = output[j,:]
        else:
            #dict_test[genre][musicName] += output[j,:]
            #changed 06.05.24
            dict_test[genre][musicName] = output[j,:]
            #print('{} is a error'.format(musicName))
    total = 0
    sum_t = 0
    for key,value in dict_test.items():
        for a,b in value.items():
            total +=1
            b = b.tolist()
            predict = int(b.index(max(b)))
            if class_to_idx[key] == predict:
                sum_t +=1
    acc = float(sum_t)/total * 100

    return acc

def write_to_pickle(dict_test,path1):
    with open(path1,'wb') as f:
        pickle.dump(dict_test,f)
    f.close()

# created 6.05.24
def output_to_csv(dict_test, f):
    #f = open(file_name, "w")
    f.write("target; track_id; predict; ")
    for i in range (0, 14):
        f.write("ft" + str(i) + ";")
    f.write("\n")
    for key,value in dict_test.items():
        for a,b in value.items():
            #f.write('{};{};',key,a)
            a_new = int(a.replace('.mp3', '').lstrip('0'))
            b = b.tolist()
            predict = int(b.index(max(b)))
            f.write("{};{};{};".format(key,a_new,predict))
            for x in b:
                f.write("{};".format(x))
            f.write('\n')
    #f.close()
# created 6.05.24

import torch
from torchviz import make_dot

def test(val_loader, model, criterion, criterion_cent, epoch, use_cuda, file_name='xxx.csv'):
    f = open(file_name, "w")
    losses = AverageMeter()
    top1 = AverageMeter()
    model.eval()
    dict_test = dict()
    feat_test = dict()
    classes = Config["normal_config"]["classes"]
    classes.sort()
    dictMeter = DictMeter(classes)
    print('clear_dict')
    for i in classes:
        dict_test[i] = dict()
        feat_test[i] = dict()
    
    conv_layer = None
    graph_created = False  # Флаг для отслеживания создания графа

    # Поиск первого сверточного слоя в модели
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            print(name)
            conv_layer = module
            break  # Прерываем цикл после нахождения первого сверточного слоя

    for batch_idx, ss in enumerate(val_loader):
        
        melspectrogram  = ss[1].unsqueeze(1)
        path = ss[0]
        targets = torch.LongTensor(ss[2])
        inputs = melspectrogram
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs =  torch.autograd.Variable(inputs, requires_grad=False)
            targets = torch.autograd.Variable(targets, requires_grad=False)
        
        with torch.no_grad():
            outputs, feat = model(inputs)
            loss_soft = criterion(outputs, targets)
            loss_cent = criterion_cent(feat, targets)
            loss = loss_soft + 1e-4 * loss_cent

        acc1 = output_to_dict(outputs.data.cpu().numpy(), path, dict_test)
        
        loss = criterion(outputs, targets)
        prec1 = accuracy(outputs.data, targets.data, topk=(1,))
        
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1[0].item(), inputs.size(0))
        dictMeter.update(dict_test)     

        if conv_layer is not None and not graph_created:  # Создаем граф только если он еще не был создан
            weights = conv_layer.weight.data
            print("Weights of the convolutional layer:")
            print(weights.size())

            x = torch.randn((1,) + tuple(inputs.shape[1:])).requires_grad_(True)
            y = model(x)
            dot = make_dot(y, params=dict(list(model.named_parameters()) + [('input', x)]))
            dot.render("neural_network_graph", format="png")  # Сохраняем граф в файл
            graph_created = True  # Устанавливаем флаг в True после создания графа

        if batch_idx % 10 == 0:
            print('test: {}/top1: {}/loss: {}'.format(batch_idx, top1.avg, losses.avg))

    output_to_csv(dict_test, f)
    f.close()

    mode = 'test'
    out_w = Config["cqt_6_14"]["save_result"].format(mode, ff)
    feat_w = Config["cqt_6_14"]["extract_feature"].format(mode, ff)

    return (losses.avg, acc1, dict_test)

def adjust_learning_rate(optimizer, epoch):
    global statei
    state['lr'] = (args.lr) * (args.exponential ** int(epoch / 2))
    # state['lr'] =  (args.lr) * (args.exponential ** int((epoch+19)/2))
    if state['lr'] <= 0.0001:
        state['lr'] = 0.0001
    for param_group in optimizer.param_groups:
        param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()

