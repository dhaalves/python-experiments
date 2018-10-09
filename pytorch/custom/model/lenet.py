import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, num_classes=12):
        self.features = nn.Sequential(
            nn.Conv2d(3, 18, kernel_size=5, stride=1, padding=2),
            nn.Threshold(0.618, 0, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(18, 48, kernel_size=5, stride=1, padding=2),
            nn.Threshold(0.618, 0, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(48 * 5 * 5, 360),
            nn.Threshold(0.618, 0, inplace=True),
            # nn.Dropout(),
            nn.Linear(360, 252),
            nn.Threshold(0.618, 0, inplace=True),
            nn.Linear(252, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 48 * 5 * 5)
        x = self.classifier(x)
        return x



class LeNet2(nn.Module):
    def __init__(self, num_classes=12, img_scale=32):
        super(LeNet, self).__init__()
        self.threshold = 0
        self.conv1 = nn.Conv2d(3, 18, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(18, 48, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(48 * self.get_linear_input(img_scale), 360)
        self.fc2 = nn.Linear(360, 252)
        self.fc3 = nn.Linear(252, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.threshold(self.conv1(x), self.threshold, 0), 2)
        x = F.dropout(x)
        x = F.max_pool2d(F.threshold(self.conv2(x), self.threshold, 0), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.dropout(x)
        x = F.threshold(self.fc1(x), self.threshold, 0)
        x = F.threshold(self.fc2(x), self.threshold, 0)
        # x = self.fc1(x)
        # x = self.fc2(x)
        x = self.fc3(x)
        # x = F.softmax(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def get_linear_input(self, x, convs=2, pool=2, stride=1, padding=0):
        linear_input_size = x
        for i in range(convs):
            linear_input_size = (linear_input_size - ((2 - padding) * 2)) / pool * stride
        return int(linear_input_size * linear_input_size)
