import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import init
from torch.nn import functional as F

from model import ShallowModule, make_conv_params_same, base_resnet,\
    Non_local, Normalize, weights_init_kaiming,weights_init_classifier
from Adaptive.Networks import Decoder, MLP, assign_adain_params, get_num_adain_params
from resnet import resnet50, resnet18
import torchvision
import copy

from sup_con_loss import SupConLoss


class ModelAdaptive(nn.Module):
    def __init__(self,  class_num, no_local= 'on', gm_pool = 'on', arch='resnet50', camera_num=6):
        super(ModelAdaptive, self).__init__()
        self.person_id = embed_net(class_num, no_local, gm_pool, arch)
        self.camera_id = Camera_net(camera_num, arch)
        self.adaptor = Decoder(n_upsample=4, n_res=2, dim=1024, output_dim=3,res_norm='adain', activ='relu', pad_type='reflect')
        self.mlp = MLP(2048, get_num_adain_params(self.adaptor), 128, 1, norm='none', activ='relu')

    def forward(self, xRGB, xIR, modal=0, with_feature = False, with_camID=False, epoch=0):

        b = xRGB.shape[0]
        if not self.training:
            return self.person_id(xRGB=xRGB, xIR=xIR, modal=modal, with_feature=with_feature)
        feat_pool, id_score , x3, person_mask = self.person_id(xRGB=xRGB, xIR=xIR, modal=modal, with_feature=with_feature)
        cam_feat, cam_score = self.camera_id(x3, person_mask)
        adain_params = self.mlp(cam_feat[b:])
        assign_adain_params(adain_params, self.adaptor)
        alpha = (max(epoch, 30)+1)/31
        xAdapt = (1-alpha) * self.adaptor(x3[:b]) + alpha * torch.rand(b, 3 , 1 ,1).cuda()
        xNorm = xAdapt / (xAdapt.sum(dim=1, keepdim=True)+1e-5).detach()
        xAdapt = (xNorm * xRGB).sum(dim=1, keepdim=True).expand(-1,3,-1,-1)

        # self.freez_person()
        feat_poolAdapt, id_scoreAdapt, x3Adapt, person_maskAdapt = self.person_id(xRGB =None, xIR = None, xZ=xAdapt, modal=3, with_feature=with_feature)
        cam_featAdapt, cam_scoreAdapt = self.camera_id(x3Adapt, person_mask[:b])
        # self.unFreez_person()


        return torch.cat((feat_pool, feat_poolAdapt), 0), torch.cat((id_score, id_scoreAdapt), 0),\
               torch.cat((cam_feat, cam_featAdapt), 0), torch.cat((cam_score, cam_scoreAdapt), 0)

    def freez_person(self):
        self.person_id.base_resnet.train(False)
        self.person_id.NL_1.train(False)
        self.person_id.NL_2.train(False)
        self.person_id.NL_3.train(False)
        self.person_id.NL_4.train(False)
        self.person_id.bottleneck.train(False)
        self.person_id.classifier.train(False)

    def unFreez_person(self):
        self.person_id.base_resnet.train(True)
        self.person_id.NL_1.train(True)
        self.person_id.NL_2.train(True)
        self.person_id.NL_3.train(True)
        self.person_id.NL_4.train(True)
        self.person_id.bottleneck.train(True)
        self.person_id.classifier.train(True)
class Camera_net(nn.Module):
    def __init__(self, camera_num=6, arch='resnet50'):

        super(Camera_net, self).__init__()
        if arch == 'resnet50':
            self.pool_dim = 2048
        else:
            self.pool_dim = 512

        #self.encoder = base_resnet(arch=arch).resnet_part2[2]  # layer4
        self.encoder = base_resnet(arch=arch).resnet_part2[2]  # layer4

        self.dim = 0
        # self.part_num = 5

        activation = nn.Sigmoid()

        self.l2norm = Normalize(2)

        self.camera_bottleneck = nn.BatchNorm1d(self.pool_dim)
        self.camera_bottleneck.bias.requires_grad_(False)  # no shift
        nn.init.constant_(self.camera_bottleneck.bias, 0)
        self.camera_classifier = nn.Linear(self.pool_dim, camera_num, bias=False)
        self.camera_bottleneck.apply(weights_init_kaiming)
        self.camera_classifier.apply(weights_init_classifier)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


        # self.adaptors = nn.Parameter(torch.ones(6, 2, 3))
        # self.adaptors.register_hook(self.parameters_hook)

    def forward(self, x, person_mask):
        cameraFeat = self.encoder(x)

        if (self.training):
            cameraFeat = (1-person_mask) * cameraFeat

        camera_global = embed_net.gl_pool(cameraFeat, 'off')
        cam_feat = self.camera_bottleneck(camera_global)

        return cam_feat, self.camera_classifier(cam_feat)


class embed_net(nn.Module):
    def __init__(self,  class_num, no_local= 'on', gm_pool = 'on', arch='resnet50', camera_num=6):
        super(embed_net, self).__init__()
        if arch == 'resnet50':
            self.pool_dim = 2048
        else:
            self.pool_dim = 512

        self.thermal_module = ShallowModule(arch=arch)
        self.visible_module = ShallowModule(arch=arch)

        self.z_module = ShallowModule(arch=arch)

        self.base_resnet = base_resnet(arch=arch)


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

        # self.adaptors = nn.Parameter(torch.ones(6, 2, 3))
        # self.adaptors.register_hook(self.parameters_hook)


    def parameters_hook(self, grad):
        print("grad:", grad)

    def forward(self, xRGB, xIR, xZ = None, modal=0, with_feature = False, with_camID=False):
        if modal == 0:
            x1 = self.visible_module(xRGB)
            x2 = self.thermal_module(xIR)
            x = torch.cat((x1, x2), 0)
            view_size = 2
        elif modal == 1:
            x = self.visible_module(xRGB)
        elif modal == 2:
            x = self.thermal_module(xIR)
        elif modal == 3:
            x = self.z_module(xZ)


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
            x3 = x
            # Layer 4

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
            x3 = x
            x = self.base_resnet.resnet_part2[2](x) # layer4

        person_mask = self.compute_mask(x)
        feat_pool = self.gl_pool(x, self.gm_pool)
        feat = self.bottleneck(feat_pool)

        if with_feature:
            return feat_pool, feat, x

        if not self.training :
            return self.l2norm(feat), self.l2norm(feat_pool)
        return feat_pool, self.classifier(feat), x3, person_mask


    @staticmethod
    def gl_pool(x, gm_pool):
        if gm_pool  == 'on':
            b, c, h, w = x.shape
            x = x.view(b, c, -1)
            p = 3.0
            x_pool = (torch.mean(x**p, dim=-1) + 1e-12)**(1/p)
        else:
            x_pool = F.adaptive_avg_pool2d(x, (1,1))
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


