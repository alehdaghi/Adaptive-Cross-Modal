import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
import math


def assign_adain_params(adain_params, model):
    # assign the adain_params to the AdaIN layers in model
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            mean = adain_params[:, :m.num_features]
            std = adain_params[:, m.num_features:2*m.num_features]
            m.bias = mean.contiguous().view(-1)
            m.weight = std.contiguous().view(-1)
            if adain_params.size(1) > 2*m.num_features:
                adain_params = adain_params[:, 2*m.num_features:]


def get_num_adain_params(model):
    # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            num_adain_params += 2*m.num_features
    return num_adain_params

class Generator(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, mod_invar_enc):
        super(Generator, self).__init__()

        input_dim = 3
        middle_dim = 32
        style_dim = 128
        activ = 'relu'
        pad_type = 'reflect'

        mlp_dim = 256
        n_res = 2
        n_downsample = 1

        # modality sepcific encoder
        # self.mod_spec_enc = ModSpecEncoderr(4, input_dim, middle_dim, style_dim, norm='none', activ=activ, pad_type=pad_type)
        # modality invariant encoder
        # self.mod_invar_enc = mod_invar_enc
        # decoder
        self.dec = Decoder(n_downsample, n_res, self.mod_invar_enc.output_dim, input_dim, res_norm='adain', activ=activ, pad_type=pad_type)
        # MLP to generate AdaIN parameters in decoder
        self.mlp = MLP(style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, norm='none', activ=activ)


    # def encode(self, images, togray=False):
    #     # encode an image to its modality-specific and modality-invariant codes
    #     content, _, _ = self.mod_invar_enc(images, togray=togray, sl_enc=True)
    #     style_fake, predict = self.mod_spec_enc(images)
    #     return content, style_fake, predict

    def decode(self, content, style):
        # decode modality-specific and modality-invariant codes to an image
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, self.dec)
        images = self.dec(content)
        return images




class Decoder(nn.Module):
    def __init__(self, output_dim=3):
        super(Decoder, self).__init__()

        # self.model = []
        # # AdaIN residual blocks
        # self.model += [ResBlocks(n_res, dim, res_norm, activ, pad_type=pad_type)]
        # # upsampling blocks
        # for i in range(n_upsample):
        #     self.model += [nn.Upsample(scale_factor=2),
        #                    Conv2dBlock(dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type=pad_type)]
        #     dim //= 2
        # # use reflection padding in the last conv layer
        # self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type=pad_type)]
        # self.model = nn.Sequential(*self.model)

        outplanes = [64, 256, 512, 1024]


        layers = [3, 4, 6, 3] # resnet50
        block = DeBottleneck
        self.stylize4 = AdaptiveInstanceNorm2d(512 * block.expansion)
        self.layer4 = self._make_layer(block, 512, layers[3], outplanes[3])

        self.stylize3 = AdaptiveInstanceNorm2d(256 * block.expansion)
        self.layer3 = self._make_layer(block, 256, layers[2], outplanes[2], upsample=nn.Upsample(scale_factor=2))

        self.stylize2 = AdaptiveInstanceNorm2d(128 * block.expansion)
        self.layer2 = self._make_layer(block, 128, layers[1], outplanes[1], upsample=nn.Upsample(scale_factor=2))

        self.stylize1 = AdaptiveInstanceNorm2d(64 * block.expansion)
        self.layer1 = self._make_layer(block, 64,  layers[0], outplanes[0])

        self.stylize0 = AdaptiveInstanceNorm2d(64)
        self.layer0 = nn.Sequential(
            nn.Upsample(scale_factor=4),
            nn.Conv2d(64, output_dim, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True)
        )


    def forward(self, x, transfer_style=True):
        if transfer_style:
            x = self.stylize4(x)
        x = self.layer4(x)
        if transfer_style:
            x = self.stylize3(x)
        x = self.layer3(x)
        if transfer_style:
            x = self.stylize2(x)
        x = self.layer2(x)
        if transfer_style:
            x = self.stylize1(x)
        x = self.layer1(x)
        if transfer_style:
            x = self.stylize0(x)
        x = self.layer0(x)

        return x

    def _make_layer(self, block, planes, blocks, outplanes, upsample=None):

        layers = []

        for i in range(1, blocks):
            layers.append(block(planes * block.expansion, planes ))

        lastModule = None
        if outplanes != planes * block.expansion:
            lastModule = nn.Sequential(
                nn.Conv2d(planes * block.expansion, outplanes,
                          kernel_size=1, bias=False),
                nn.BatchNorm2d(outplanes),
            )

        layers.append(block(outplanes, planes, lastModule, upsample))


        return nn.Sequential(*layers)

class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm, activation, pad_type):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm, activ):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')] # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))




##################################################################################
# Basic Blocks
##################################################################################
class DeBottleneck(nn.Module):
    expansion = 4
    def __init__(self, out_channels, mid_channels, lastModule=None, upsample=None):
        super(DeBottleneck, self).__init__()


        self.stride = 1

        self.conv1 = nn.Conv2d(mid_channels * self.expansion, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # original padding is 1; original dilation is 1

        self.lastModule = lastModule
        self.upsample = upsample

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):

        out = x

        if self.upsample is not None:
            out = self.upsample(out)
        if self.lastModule is not None:
            residual = self.lastModule(out)
        else:
            residual = out

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)


        out += residual

        out = self.relu(out)

        return out


class ResBlock(nn.Module):
    def __init__(self, dim, norm, activation, pad_type):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out



class Conv2dBlock(nn.Module):

    def __init__(self, input_dim ,output_dim, kernel_size, stride, padding, norm, activation, pad_type):
        super(Conv2dBlock, self).__init__()

        self.use_bias = True

        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x



class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm, activation):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        if norm == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out



##################################################################################
# Normalization layers
##################################################################################

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)



class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x



class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'
