import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import init
from torch.nn import functional as F

from model import ShallowModule, make_conv_params_same, base_resnet,\
    Non_local, Normalize, weights_init_kaiming,weights_init_classifier
from resnet import resnet50, resnet18
import torchvision
import copy

from sup_con_loss import SupConLoss


class embed_net(nn.Module):
    def __init__(self,  class_num, no_local= 'on', gm_pool = 'on', arch='resnet50', camera_num=6):
        super(embed_net, self).__init__()
        if arch == 'resnet50':
            self.pool_dim = 2048
        else:
            self.pool_dim = 512

        self.thermal_module = ShallowModule(arch=arch)
        self.visible_module = ShallowModule(arch=arch)
        self.gray_module = ShallowModule(arch=arch)

        self.base_resnet = base_resnet(arch=arch)
        self.cameraFeat_module = copy.deepcopy(self.base_resnet.resnet_part2[2]) # layer4

        self.dim = 0
        # self.part_num = 5


        activation = nn.Sigmoid()

        self.non_local = no_local
        if self.non_local =='on':
            layers=[3, 4, 6, 3]
            non_layers=[0,2,3,0]
            self.NL_1 = nn.ModuleList(
                [Non_local(256) for i in range(non_layers[0])])
            self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
            self.NL_2 = nn.ModuleList(
                [Non_local(512) for i in range(non_layers[1])])
            self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
            self.NL_3 = nn.ModuleList(
                [Non_local(1024) for i in range(non_layers[2])])
            self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
            self.NL_4 = nn.ModuleList(
                [Non_local(2048) for i in range(non_layers[3])])
            self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])



        self.l2norm = Normalize(2)

        self.bottleneck = nn.BatchNorm1d(self.pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        nn.init.constant_(self.bottleneck.bias, 0)
        self.classifier = nn.Linear(self.pool_dim, class_num, bias=False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

        self.camera_bottleneck = nn.BatchNorm1d( self.pool_dim)
        self.camera_bottleneck.bias.requires_grad_(False)  # no shift
        nn.init.constant_(self.camera_bottleneck.bias, 0)
        self.camera_classifier = nn.Linear(self.pool_dim, camera_num, bias=False)
        self.camera_bottleneck.apply(weights_init_kaiming)
        self.camera_classifier.apply(weights_init_classifier)

        # self.bottleneck_parts = [nn.BatchNorm1d(self.pool_dim) for i in range(self.part_num)]
        # for btl in self.bottleneck_parts:
        #     btl.bias.requires_grad_(False)  # no shift
        #     nn.init.constant_(btl.bias, 0)
        #     btl.apply(weights_init_kaiming)
        #
        # self.classifier_parts = [nn.Linear(self.pool_dim, class_num, bias=False) for i in range(self.part_num)]
        # for cls in self.classifier_parts:
        #     cls.apply(weights_init_classifier)


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gm_pool = gm_pool

        self.adaptors = nn.Parameter(torch.ones(6, 2, 3))
        # self.adaptors.register_hook(self.parameters_hook)


    def parameters_hook(self, grad):
        print("grad:", grad)

    def forward(self, x1, x2, x3=None, modal=0, with_feature = False, with_camID=False):
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)
            view_size = 2
            if x3 is not None :
                x3 = self.gray_module(x3)
                x = torch.cat((x, x3), 0)
                view_size = 3
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)
        elif modal == 3:
            x = self.gray_module(x3)

        # shared block
        if self.non_local == 'on':
            NL1_counter = 0
            if len(self.NL_1_idx) == 0: self.NL_1_idx = [-1]
            # for i in range(len(self.base_resnet.base.layer1)):
            #     x = self.base_resnet.layer1[i](x)
            #     if i == self.NL_1_idx[NL1_counter]:
            #         _, C, H, W = x.shape
            #         x = self.NL_1[NL1_counter](x)
            #         NL1_counter += 1
            # Layer 2
            NL2_counter = 0
            if len(self.NL_2_idx) == 0: self.NL_2_idx = [-1]
            for i in range(len(self.base_resnet.resnet_part2[0])):
                x = self.base_resnet.resnet_part2[0][i](x)
                if i == self.NL_2_idx[NL2_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_2[NL2_counter](x)
                    NL2_counter += 1
            # Layer 3
            NL3_counter = 0
            if len(self.NL_3_idx) == 0: self.NL_3_idx = [-1]
            for i in range(len(self.base_resnet.resnet_part2[1])):
                x = self.base_resnet.resnet_part2[1][i](x)
                if i == self.NL_3_idx[NL3_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_3[NL3_counter](x)
                    NL3_counter += 1
            # Layer 4
            cameraFeat = self.cameraFeat_module(x)
            NL4_counter = 0
            if len(self.NL_4_idx) == 0: self.NL_4_idx = [-1]
            for i in range(len(self.base_resnet.resnet_part2[2])):
                x = self.base_resnet.resnet_part2[2][i](x)
                if i == self.NL_4_idx[NL4_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_4[NL4_counter](x)
                    NL4_counter += 1
        else:
            x = self.base_resnet.resnet_part2[0](x) # layer2
            x = self.base_resnet.resnet_part2[1](x) # layer3
            if self.training:
                cameraFeat = self.cameraFeat_module(x)
            x = self.base_resnet.resnet_part2[2](x) # layer4



        if (self.training):
            person_mask = self.compute_mask(x)
            cameraFeat = (1-person_mask) * cameraFeat

        camera_global = self.gl_pool(cameraFeat)


        feat_pool = self.gl_pool(x)

        feat = self.bottleneck(feat_pool)

        if with_feature:
            return feat_pool, feat, x, camera_global, cameraFeat

        cam_feat = self.camera_bottleneck(camera_global)

        if not self.training and with_camID:
            return self.l2norm(feat), self.l2norm(feat_pool), self.camera_classifier(cam_feat)
        if not self.training :
            return self.l2norm(feat), self.l2norm(feat_pool)
        return feat_pool, self.classifier(feat), camera_global, self.camera_classifier(cam_feat)


    def gl_pool(self, x):
        if self.gm_pool  == 'on':
            b, c, h, w = x.shape
            x = x.view(b, c, -1)
            p = 3.0
            x_pool = (torch.mean(x**p, dim=-1) + 1e-12)**(1/p)
        else:
            x_pool = self.avgpool(x)
            x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))

        return x_pool

    def extractPartsMask(self, x):
        return self.maskDetector(x)

    def getPoolDim(self):
        return self.pool_dim

    @staticmethod
    def compute_mask(feat):
        batch_size, fdim, h, w = feat.shape
        norms = torch.norm(feat, p=2, dim=1).view(batch_size, h*w)

        norms -= norms.min(dim=-1, keepdim=True)[0]
        norms /= norms.max(dim=-1, keepdim=True)[0] + 1e-12
        mask = norms.view(batch_size, 1, h, w)

        return mask.detach()


    def count_params(self):
        ids = set(map(id, self.parameters()))
        params = filter(lambda p: id(p) in ids, self.parameters())
        return sum(p.numel() for p in params if p.requires_grad)



