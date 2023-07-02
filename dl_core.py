import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import enum

E = 0.0001 # EPSILON


class NetTypes2D(enum.Enum):
    Resnet50 = 'resnet50'
    Densenet121 = 'densenet121'
    Resnext50 = 'resnext50'
    Inception3 = 'inception_v3'



class DeepLearningModel2D(nn.Module):
    def __init__(self, model_type):
        super().__init__()

        if model_type == NetTypes2D.Resnet50:
            fe = models.resnet50(pretrained=False)
            fe.conv1 = nn.Conv2d(1, 64, (3, 3), (2, 2), (1, 1), bias=False)
        elif model_type == NetTypes2D.Densenet121:
            fe = models.densenet121(pretrained=False)
            fe.features.conv0 = nn.Conv2d(1, 64, (3, 3), (2, 2), (1, 1), bias=False)
        elif model_type == NetTypes2D.Resnext50:
            fe = models.resnext50_32x4d(pretrained=False)
            fe.conv1 = nn.Conv2d(1, 64, (3, 3), (2, 2), (1, 1), bias=False)
        elif model_type == NetTypes2D.Inception3:
            fe = models.inception_v3(pretrained=False, transform_input=False)
            fe.Conv2d_1a_3x3 = models.inception.BasicConv2d(1, 32, kernel_size=3, stride=2)
        else:
            raise NotImplementedError

        self.model_type = model_type
        self.feature_extractor = fe
        in_features = list(self.feature_extractor.modules())[-1].out_features
        self.dropout1 = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(in_features=in_features, out_features=1, bias=True)
        self.fc2 = nn.Linear(in_features=in_features, out_features=1, bias=True)
        #nn.init.xavier_normal_(self.fc1.weight)
        self.apply(he_init)

    def forward(self, x):
        x1 = self.feature_extractor(x)
        if isinstance(x1, models.inception.InceptionOutputs):
            x1 = x1[0]

        x2 = self.dropout1(x1)
        out1 = self.fc1(x2)
        out2 = self.fc2(x2)
        return out2

def he_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
