import argparse
import csv
import os
import shutil
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import visdom
import numpy as np

from tensorboardX import SummaryWriter
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger, VisdomLogger

from custom.folder import ImageFolder

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
print(model_names)
vis = visdom.Visdom()
writer = SummaryWriter('runs')

datasets = {
    'LEAVES1': '/mnt/sdb1/dataset/leaves/leaves1',
    'SOYBEAN1': '/mnt/sdb1/dataset/soybean/soybean1_split',
    'SOYBEAN2': '/mnt/sdb1/dataset/soybean/soybean2',
    'SOYBEAN3': '/mnt/sdb1/dataset/soybean/soybean3',
    'EGGS_8': '/mnt/sdb1/dataset/parasites/eggs_8',
    'EGGS_9': '/mnt/sdb1/dataset/parasites/eggs_9',
    'EGGS_16': '/mnt/sdb1/dataset/parasites/eggs_16',
    'LARVAE_2': '/mnt/sdb1/dataset/parasites/larvae_2',
    'PROTOZOAN_6': '/mnt/sdb1/dataset/parasites/protozoan_6',
    'PROTOZOAN_7': '/mnt/sdb1/dataset/parasites/protozoan_7',
    'PROTOZOAN_12': '/mnt/sdb1/dataset/parasites/protozoan_12',
    'PARASITES_15': '/mnt/sdb1/dataset/parasites/parasites_15',
    'PARASITES_16': '/mnt/sdb1/dataset/parasites/parasites_16',
    'PARASITES_30': '/mnt/sdb1/dataset/parasites/parasites_30',
    'MAMMOSET_5-2': '/mnt/sdb1/dataset/mammoset/exp5-2/',
    'MAMMOSET_5-2_B': '/mnt/sdb1/dataset/mammoset/exp5-2_B/',
    'MAMMOSET_5-2_AUG': '/mnt/sdb1/dataset/mammoset/exp5-2_aug'
}

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--dataset', '-ds', metavar='NAME', default='MAMMOSET_5-2_AUG',
                    help='nome do dataset (LEAVES1, SOYBEAN1, SOYBEAN2, SOYBEAN3, PARASITES1)')

# parser.add_argument('--datadir', metavar='DIR', default=datasets['SOYBEAN1'],  help='caminho para o dataset')

parser.add_argument('--model', '-m', metavar='MODEL', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')

parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--pretrained', dest='pretrained', action='store_true', default=True,
                    help='use pre-trained model')

parser.add_argument('-tl', '--freezed', dest='freezed', action='store_true', default=True,
                    help='use transfer learning')

parser.add_argument('-op', '--optimizer', dest='optimizer', default='Adam',
                    help='choose optimizer')

parser.add_argument('--imagedistortions', dest='imagedistortions', action='store_true', default=False,
                    help='use distoced images')


def visualize_model(model, dataloaders, num_images=6):
    images_so_far = 0
    fig = plt.figure()

    for i, data in enumerate(dataloaders['val']):
        inputs, labels = data
        inputs, labels = torch.autograd.Variable(
            inputs.cuda()), torch.autograd.Variable(labels.cuda())

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images // 2, 2, images_so_far)
            ax.axis('off')
            # ax.set_title('predicted: {}'.format(class_names[preds[j]]))
            plt.imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                return


def save_results_on_csv(epoch, acc_train, loss_train, acc_val, loss_val, correct_per_class):
    with open('results_mammoset.csv', 'a') as f:
        csv_writer = csv.writer(
            f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if not os.path.exists('results_mammoset.csv'):
            csv_writer.writerow(['dataset', 'model', 'pre_trained', 'weight_freezed', 'optimizer',
                                 'image_distortions', 'batch_size', 'epochs', 'learning_rate', 'momentum', 'train_loss',
                                 'train_acc', 'val_loss', 'val_acc', 'correct_per_class'])
        csv_writer.writerow([args.dataset, args.model, args.pretrained, args.freezed, args.optimizer,
                             args.imagedistortions, args.batch_size, epoch, args.lr, args.momentum, acc_train,
                             loss_train, acc_val, loss_val, correct_per_class])


def visualize_init(acc=0.0, acc_train=0.0, loss=0.0, loss_train=0.0, epoch=0):
    legend_acc, legend_loss, x, y_acc, y_loss = vis_params(
        acc, acc_train, epoch, loss, loss_train)
    win_loss = vis.line(Y=y_loss, X=x, opts=legend_loss)
    win_acc = vis.line(Y=y_acc, X=x, opts=legend_acc)
    return win_acc, win_loss


def visualize(acc, acc_train, loss, loss_train, epoch, win_acc, win_loss):
    legend_acc, legend_loss, x, y_acc, y_loss = vis_params(
        acc, acc_train, epoch, loss, loss_train)
    vis.line(win=win_loss,
             update='append',
             Y=y_loss,
             X=x,
             opts=legend_loss)
    vis.line(win=win_acc,
             update='append',
             Y=y_acc,
             X=x,
             opts=legend_acc)


def vis_params(acc, acc_train, epoch, loss, loss_train):
    x = np.array([epoch])
    y_loss = np.column_stack((np.array(loss_train), np.array(loss)))
    y_acc = np.column_stack((np.array(acc_train), np.array(acc)))
    legend_loss = dict(title=str(args.model) + ' - lr:' + ('%.2E' % args.lr) + ' - batch_size:' + str(args.batch_size),
                       xtick=True, xlabel='Epochs', ylabel='Loss', legend=['train', 'val'])
    # legend_loss = dict(legend=['train_loss', 'val_loss'])
    legend_acc = dict(title=str(args.model) + ' - lr:' + ('%.2E' % args.lr) + ' - batch_size:' + str(args.batch_size),
                      xtick=True, xlabel='Epochs', ylabel='Accuracy', legend=['train', 'val'])
    # legend_acc = dict(legend=['train_acc', 'val_acc'])
    return legend_acc, legend_loss, x, y_acc, y_loss


def train_and_eval(criterion, model, optimizer, class_to_id, train_loader, val_loader):
    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    monitor = {'previous_loss': 0.0, 'best_acc': 0.0,
               'best_epoch': 0, 'non_improving_counter': 0}

    win_acc, win_loss = visualize_init()

    for epoch in range(args.start_epoch, args.epochs):

        # treina por uma epoca no conjunto de treino
        acc_train, loss_train = train(
            train_loader, model, criterion, optimizer, epoch)

        # valida no conjunto de validacao a cada epoca
        acc, loss, acc_per_class = validate(
            val_loader, model, criterion, epoch, class_to_id)

        visualize(acc, acc_train, loss, loss_train,
                  epoch + 1, win_acc, win_loss)

        # decaimento da taxa de aprendizado
        adjust_learning_rate(optimizer, epoch, 0.7, 2)

        # e para apos 3 epocas consecutivas com acuracias piores
        if loss < monitor['previous_loss']:
            monitor['non_improving_counter'] = 0
        else:
            monitor['non_improving_counter'] += 1
            if monitor['non_improving_counter'] > 5:
                break
        monitor['previous_loss'] = acc

        # salva o modelo e parametros da melhor acuracia da validacao
        if acc > monitor['best_acc']:
            monitor['best_epoch'] = epoch
            monitor['best_acc'] = acc
            save_checkpoint({
                'model': args.model,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_acc': monitor['best_acc'],
                'epoch': monitor['best_epoch'],
            }, True)

        save_results_on_csv(epoch, acc_train, loss_train, acc, loss, acc_per_class)

    print('#### Best Acc %.2f at Epoch %d' %
          (monitor['best_acc'], monitor['best_epoch']))


def load_model(classes):
    optim_parameters = None
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.model))
        model = models.__dict__[args.model](pretrained=True)
        optim_parameters = model.parameters()
        if args.freezed:
            for p in model.parameters():
                p.requires_grad = False
            if args.model.startswith('inception') or args.model.startswith('resnet'):
                model.fc = nn.Linear(model.fc.in_features, len(classes))
                optim_parameters = model.fc.parameters()
            elif args.model.startswith('alexnet'):
                model.classifier = nn.Linear(
                    model.classifier[1].in_features, len(classes))
                optim_parameters = model.classifier.parameters()
            elif args.model.startswith('squeezenet'):
                model.classifier = nn.Linear(12, len(classes))
                optim_parameters = model.classifier.parameters()
            else:
                model.classifier = nn.Linear(
                    model.classifier[0].in_features, len(classes))
                optim_parameters = model.classifier.parameters()
    else:
        print("=> creating model '{}'".format(args.model))
        model = models.__dict__[args.model]()
        optim_parameters = model.parameters()

    if args.model.startswith('alexnet') or args.model.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adagrad(optim_parameters, args.lr)

    return criterion, model, optimizer


def load_dataset():
    # Normalizacao utilizada no paper da ResNet https://arxiv.org/abs/1512.03385
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    scale = 299 if args.model == 'inception_v3' else 224

    transform = transforms.Compose([
        # transforms.RandomResizedCrop(200),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(45),
        transforms.Resize([scale, scale]),
        # transforms.CenterCrop(scale),
        transforms.ToTensor(),
        normalize,
    ])

    train_dir = os.path.join(args.datadir, 'train')
    val_dir = os.path.join(args.datadir, 'test')

    # train_data = torchvision.datasets.ImageFolder(train_dir, transform)
    # val_data = torchvision.datasets.ImageFolder(val_dir, transform)

    train_data, val_data = ImageFolder(
        args.datadir,
        transform).get_split(train_perc=0.8)

    print('Tamanho conjuto de treino %s' % len(train_data))
    print('Tamanho conjuto de teste %s' % len(val_data))
    # print('Quantidade por Classes %s' % train_data.class_counts)
    print('Classes Ids %s' % train_data.class_to_idx)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=1, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)
    return train_data.classes, train_data.class_to_idx, train_loader, val_loader


def train(train_loader, model, criterion, optimizer, epoch):
    time_meter = AverageMeter()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    model.train()

    end = time.time()

    # para cada batch de imagens
    for i, (input, target) in enumerate(train_loader):

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # Computa a saida do modelo
        output = model(input_var)
        if args.model == 'inception_v3':
            # get from final classifier not auxiliary
            output = output[0]

        # Computa o erro de acordo com funcao definida para o criterion
        loss = criterion(output, target_var)

        # Mede a acuracia e o erro
        pred, prec5 = accuracy(output.data, target, topk=(1, 5))
        loss_meter.update(loss.data[0], input.size(0))
        acc_meter.update(pred[0], input.size(0))

        # Computa os gradientes e faz o backpropagation reajustando os parametros (pesos) da rede
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
            # 'Time {time.val:.3f} ({time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.2f} ({acc.avg:.2f})'.format(
                epoch + 1, i + 1, len(train_loader),
                time=time_meter, loss=loss_meter, acc=acc_meter))
    print('Epoch: [{0}][TRAIN]\t\t'
          'Loss {loss.avg:.4f}\t\t'
          'Acc {acc.avg:.2f}\n'.format(
        epoch + 1, args.epochs, loss=loss_meter, acc=acc_meter))
    return acc_meter.avg, loss_meter.avg


def validate(val_loader, model, criterion, epoch, class_to_id):
    time_meter = AverageMeter()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    model.eval()

    end = time.time()

    acc_per_class = defaultdict(lambda: 0)

    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # Computa a saida do modelo
        output = model(input_var)

        # Computa o erro de acordo com funcao definida para o criterion
        loss = criterion(output, target_var)

        # Mede a acuracia e o erro
        pred, prec5 = accuracy(output.data, target, topk=(1, 5))
        loss_meter.update(loss.data[0], input.size(0))
        acc_meter.update(pred[0], input.size(0))
        acc_per_class[target.cpu().numpy()[0]] += int(acc_meter.val / 100)

        time_meter.update(time.time() - end)
        end = time.time()

    acc_per_class = map_id_to_class(class_to_id, acc_per_class)
    print('Epoch: [{0}][VAL]\t\t'
          'Loss {loss.avg:.4f}\t\t'
          'Acc {acc.avg:.2f}\n'.format(
        epoch + 1, args.epochs, loss=loss_meter, acc=acc_meter))

    return acc_meter.avg, loss_meter.avg, acc_per_class


def map_id_to_class(class_to_id, correct_per_class):
    aux_dict = {}
    for i, c in enumerate(sorted(class_to_id, key=class_to_id.get)):
        aux_dict[c] = correct_per_class[i]
    return aux_dict


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'best_model.pth.tar')


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


def adjust_learning_rate(optimizer, epoch, factor, epochs_to_decay):
    lr = args.lr * (factor ** (epoch // epochs_to_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    global args

    args = parser.parse_args()

    print('CUDA disponivel' if torch.cuda.is_available() else '')


    # args.datadir = datasets[args.dataset]
    args.datadir = datasets['MAMMOSET_5-2_AUG']

    args.batch_size = 8
    args.freezed = False
    args.epochs = 25
    args.model = 'resnet18'
    iter_to_search = 20
    args.lr = 0.001

    print(args)

    classes, class_to_id, train_loader, val_loader = load_dataset()
    criterion, model, optimizer = load_model(classes)

    for i in range(iter_to_search):
        args.lr = 10**np.random.uniform(-4, -3)
        train_and_eval(criterion, model, optimizer, class_to_id, train_loader, val_loader)

    # Grid Search
    # for args.model in model_names:
    # print(0.1 * (0.9 ** (5 // 5)))
    # for args.batch_size in [8, 16, 32, 64]:
    #     for args.lr in [0.05, 0.01, 0.005, 0.001, 0.005, 0.0001]:
    #         for args.pretrained in [True]:
    #             if args.pretrained:
    #                 for args.freezed in [True]:
    #                     classes, class_to_id, train_loader, val_loader = load_dataset()
    #                     criterion, model, optimizer = load_model(classes)
    #                     train_and_eval(criterion, model, optimizer, class_to_id, train_loader, val_loader)
    #             else:
    #                 args.freezed = False
    #                 classes, class_to_id, train_loader, val_loader = load_dataset()
    #                 criterion, model, optimizer = load_model(classes)
    #                 train_and_eval(criterion, model, optimizer, class_to_id, train_loader, val_loader)
