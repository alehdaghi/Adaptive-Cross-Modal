from focalnet import focalnet_tiny_lrf
import torch
import torch.nn as nn
from torch.nn import init
from resnet import resnet50, resnet18
import torchvision
import copy
import torch.nn.functional as F

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)


def make_bn_new(model):
    '''
    copies all batch normalizations layers in the model to new ones
    Returns:

    '''
    modelNew = model
    modules = list(modelNew.modules())
    for i, m in enumerate(model.modules()):
        if m.__class__.__name__ == "BatchNorm2d":
            #modules[i]
            modules[i].weight = nn.Parameter(m.weight.clone())
            modules[i].bias = nn.Parameter(m.bias.clone())
    return modelNew

def make_conv_params_same(model1, model2):
    '''
    copies all batch normalizations layers in the model to new ones
    Returns:

    '''

    for m1, m2 in zip(model1.modules(), model2.modules()):
        if m1.__class__.__name__ == "Conv2d":
            #modules[i]
            m1.weight = m2.weight
            m1.bias = m2.bias



class embed_net(nn.Module):
    def __init__(self,  class_num):
        super(embed_net, self).__init__()
        backbone = focalnet_tiny_lrf(True)


        self.thermal_module = backbone.patch_embed
        self.visible_module = copy.deepcopy(backbone.patch_embed)
        self.gray_module = copy.deepcopy(backbone.patch_embed)


        self.shared_module = backbone

        self.pool_dim = 512

        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(self.pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        self.classifier = nn.Linear(self.pool_dim, class_num, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x1, x2, x3=None, modal=0, with_feature = False):
        if modal == 0:
            x1, H, W = self.visible_module(x1)
            x2, _, _ = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)
            if x3 is not None :
                x3, _, _ = self.gray_module(x3)
                x = torch.cat((x, x3), 0)
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)
        elif modal == 3:
            x = self.gray_module(x3)

        # shared block
        x = self.pos_drop(x)

        for layer in self.shared_module.layers:
            x, H, W = layer(x, H, W)
        x = self.shared_module.norm(x)  # B L C
        x_pool = self.shared_module.avgpool(x.transpose(1, 2))  # B C 1
        x_pool = torch.flatten(x, 1)

        feat = self.bottleneck(x_pool)
        score = self.classifier(feat)

        if self.training :
            return x_pool, score
        else:
            return self.l2norm(x_pool), self.l2norm(feat)

    def getPoolDim(self):
        return self.pool_dim

    def count_params(self):
        ids = set(map(id, self.parameters()))
        params = filter(lambda p: id(p) in ids, self.parameters())
        return sum(p.numel() for p in params if p.requires_grad)

