import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from model import grad_reverse
class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        model_load = models.resnet101(pretrained=True)
        mod=list(model_load.children())
        mod.pop()
        self.feature=nn.Sequential(*mod)
        self.dim=2048


    def forward(self, x,mode):
       x=self.feature(x)
       x=x.view(x.size(0),self.dim)
       return x


class Predictor(nn.Module):
    def __init__(self, num_classes=13,num_layer = 2,num_unit=2048,prob=0.5,middle=1000):
        super(Predictor, self).__init__()
        layers=[]
        layers.append(nn.Dropout(p=prob))
        layers.append(nn.Linear(num_unit,middle))
        layers.append(nn.BatchNorm1d(middle,affine=True))
        layers.append(nn.ReLU(inplace=True))
        for i in range(num_layer-1):
            layers.append(nn.Dropout(p=prob))
            layers.append(nn.Linear(middle, middle))
            layers.append(nn.BatchNorm1d(middle, affine=True))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(middle,num_classes))
        self.classfier=nn.Sequential(*layers)

    # def set_lambda(self, lambd):
    #     self.lambd = lambd

    def forward(self, x,mode,reverse=False):

        # if mode!='ad_drop':
        #     if reverse:
        #         rev=grad_reverse.grad_reverse()
        #         x=rev(x)
        # else:
            # x = F.relu(self.bn1_fc(self.fc1(x)))
            # x = F.dropout(x, training=self.training, p=self.prob)
        x=self.classfier(x)
        # x = F.relu(self.bn2_fc(self.fc2(x)))
        # if mode == 'ad_drop':
        #     x = F.dropout(x, training=self.training, p=self.prob)
        # x = self.fc3(x)
        return x