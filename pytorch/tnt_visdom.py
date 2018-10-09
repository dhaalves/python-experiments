""" Run MNIST example and log to visdom
    Notes:
        - Visdom must be installed (pip works)
        - the Visdom server must be running at start!

    Example:
        $ python -m visdom.server -port 8097 &
        $ python mnist_with_visdom.py
"""
import torch
import torch.nn.functional as F
import torch.optim
import torchnet as tnt
from torch.autograd import Variable
from torch.nn.init import kaiming_normal
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger, VisdomLogger
from torchvision import models
from torchvision.transforms import transforms
from tqdm import tqdm
import numpy as np

from pytorch.custom import ImageFolder


def get_imager_folder(folder='/mnt/sdb1/datasets/mammoset/exp5-2_aug'):
    # Normalizacao utilizada no paper da ResNet https://arxiv.org/abs/1512.03385
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    scale = 224

    transform = transforms.Compose([
        # transforms.RandomResizedCrop(200),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(45),
        transforms.Resize([scale, scale]),
        # transforms.CenterCrop(scale),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        normalize,
    ])

    train_data, val_data = ImageFolder(
        folder,
        transform).get_split(train_perc=0.8)

    num_classes = len(train_data.class_counts)

    print('Tamanho conjuto de treino %s' % len(train_data))
    print('Tamanho conjuto de teste %s' % len(val_data))
    print('Classes Ids %s' % train_data.class_to_idx)
    print('Classes Counts %s' % train_data.class_counts)

    return train_data, val_data, num_classes


def get_iterator(train_data, val_data, mode):
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True, num_workers=4)

    val_loader = torch.utils.data.DataLoader(val_data, batch_size=8, shuffle=False, num_workers=4)

    return train_loader if mode else val_loader


def conv_init(ni, no, k):
    return kaiming_normal(torch.Tensor(no, ni, k, k))


def linear_init(ni, no):
    return kaiming_normal(torch.Tensor(no, ni))


def f(params, inputs, mode):
    o = inputs.view(inputs.size(0), 1, 28, 28)
    o = F.conv2d(o, params['conv0.weight'], params['conv0.bias'], stride=2)
    o = F.relu(o)
    o = F.conv2d(o, params['conv1.weight'], params['conv1.bias'], stride=2)
    o = F.relu(o)
    o = F.conv2d(o, params['conv2.weight'], params['conv2.bias'], stride=2)
    o = F.relu(o)
    o = o.view(o.size(0), -1)
    o = F.linear(o, params['linear2.weight'], params['linear2.bias'])
    o = F.relu(o)
    o = F.linear(o, params['linear3.weight'], params['linear3.bias'])
    return o


def main():
    train_data, val_data, num_classes = get_imager_folder()

    params = {
        'conv0.weight': conv_init(1, 50, 5), 'conv0.bias': torch.zeros(50),
        'conv1.weight': conv_init(50, 50, 5), 'conv1.bias': torch.zeros(50),
        'conv2.weight': conv_init(50, 50, 5), 'conv2.bias': torch.zeros(50),
        'linear2.weight': linear_init(800, 512), 'linear2.bias': torch.zeros(512),
        'linear3.weight': linear_init(512, num_classes), 'linear3.bias': torch.zeros(num_classes),
    }
    params = {k: Variable(v, requires_grad=True) for k, v in params.items()}


    model = models.resnet50(pretrained=True)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                          bias=False)
    # for p in model.parameters():
    #     p.requires_grad = False
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.cuda()
    params = model.parameters()
    optimizer = torch.optim.Adagrad(params)

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    classerr = tnt.meter.ClassErrorMeter(accuracy=True)
    confusion_meter = tnt.meter.ConfusionMeter(num_classes, normalized=True)

    port = 8097
    train_loss_logger = VisdomPlotLogger(
        'line', port=port, opts={'title': 'Train Loss'})
    train_err_logger = VisdomPlotLogger(
        'line', port=port, opts={'title': 'Train Acc'})
    test_loss_logger = VisdomPlotLogger(
        'line', port=port, opts={'title': 'Test Loss'})
    test_err_logger = VisdomPlotLogger(
        'line', port=port, opts={'title': 'Test Acc'})
    confusion_logger = VisdomLogger('heatmap', port=port, opts={'title': 'Confusion matrix',
                                                                'columnnames': list(range(num_classes)),
                                                                'rownames': list(range(num_classes))})
    acc_logger = VisdomPlotLogger(
        'line', port=port, opts={'xlabel': 'Epochs', 'ylabel': 'Accuracy', 'legend': ['train', 'val']})
    loss_logger =  VisdomPlotLogger(
        'line', port=port, opts={'xlabel': 'Epochs', 'ylabel': 'Accuracy', 'legend': ['train', 'val']})

    def h(sample):
        inputs = Variable(sample[0].cuda())  # .float() / 255.0)
        targets = Variable(sample[1].cuda())
        outputs = model(inputs)
        # if model.training: #and model.__all__[1] == 'inception_v3':
        #     outputs = outputs[0]
        # o = f(params, inputs, sample[2])
        return F.cross_entropy(outputs, targets), outputs

    def reset_meters():
        classerr.reset()
        meter_loss.reset()
        confusion_meter.reset()

    def on_sample(state):
        state['sample'].append(state['train'])

    def on_forward(state):
        classerr.add(state['output'].data,
                     torch.LongTensor(state['sample'][1]))
        confusion_meter.add(state['output'].data,
                            torch.LongTensor(state['sample'][1]))
        meter_loss.add(state['loss'].data[0])

    def on_start_epoch(state):
        model.train()
        reset_meters()
        state['iterator'] = tqdm(state['iterator'])

    def on_end_epoch(state):
        print('Training loss: %.4f, accuracy: %.2f%%' % (meter_loss.value()[0], classerr.value()[0]))
        train_loss_logger.log(state['epoch'], meter_loss.value()[0])
        train_err_logger.log(state['epoch'], classerr.value()[0])

        # train_acc = classerr.value()[0]
        # train_loss = meter_loss.value()[0]
        # do validation at the end of each epoch
        reset_meters()
        model.eval()
        engine.test(h, get_iterator(train_data, val_data, False))
        # y_acc = np.column_stack((np.array(train_acc), np.array(classerr.value()[0])))
        # y_loss = np.column_stack((np.array(train_acc), np.array(classerr.value()[0])))
        # acc_logger.log(state['epoch'], np.column_stack((np.array(train_acc), np.array(classerr.value()[0]))))
        # loss_logger.log(state['epoch'], np.column_stack((np.array(train_loss), np.array(meter_loss.value()[0]))))
        test_loss_logger.log(state['epoch'], meter_loss.value()[0])
        test_err_logger.log(state['epoch'], classerr.value()[0])
        confusion_logger.log(confusion_meter.value())
        print('Testing loss: %.4f, accuracy: %.2f%%' % (meter_loss.value()[0], classerr.value()[0]))

    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.train(h, get_iterator(train_data, val_data, True), maxepoch=50, optimizer=optimizer)


if __name__ == '__main__':
    for i in range(5):
        main()

