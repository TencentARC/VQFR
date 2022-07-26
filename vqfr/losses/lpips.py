"""Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from torchvision import models

from vqfr.utils.registry import LOSS_REGISTRY

VGG_PRETRAIN_PATH = 'experiments/pretrained_models/vgg16-397923af.pth'
VGG_LPIPS_PRETRAIN_PATH = 'experiments/pretrained_models/lpips/vgg.pth'


@LOSS_REGISTRY.register()
class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self,
                 perceptual_weight=1.0,
                 style_weight=0.0,
                 inp_range=(-1, 1),
                 use_dropout=True,
                 style_measure='L1'):
        super().__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.style_measure = style_measure
        self.scaling_layer = ScalingLayer(inp_range)
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()

        requires_grad = False
        if not requires_grad:
            self.eval()
            for param in self.parameters():
                param.requires_grad = False
        else:
            self.train()
            for param in self.parameters():
                param.requires_grad = True

    def load_from_pretrained(self):
        ckpt = VGG_LPIPS_PRETRAIN_PATH
        self.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu')), strict=False)
        print('loaded pretrained LPIPS loss from {}'.format(ckpt))

    def _gram_mat(self, x):
        """Calculate Gram matrix.
        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).
        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram

    def forward(self, input, target):
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]

        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk])**2

        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for layer in range(1, len(self.chns)):
            val += res[layer]
        percep_loss = self.perceptual_weight * torch.mean(val)

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for kk in range(len(self.chns)):
                if self.style_measure == 'L1':
                    style_loss += F.l1_loss(self._gram_mat(feats0[kk]), self._gram_mat(feats1[kk]))
                elif self.style_measure == 'L2':
                    style_loss += F.mse_loss(self._gram_mat(feats0[kk]), self._gram_mat(feats1[kk]))
                else:
                    raise NotImplementedError
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss


class ScalingLayer(nn.Module):

    def __init__(self, inp_range):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

        if min(inp_range) == -1 and max(inp_range) == 1:
            self.register_buffer('mean', torch.Tensor([0., 0., 0.])[None, :, None, None])
            self.register_buffer('std', torch.Tensor([1., 1., 1.])[None, :, None, None])
        elif min(inp_range) == 0 and max(inp_range) == 1:
            self.register_buffer('mean', torch.Tensor([0.5, 0.5, 0.5])[None, :, None, None])
            self.register_buffer('std', torch.Tensor([0.5, 0.5, 0.5])[None, :, None, None])
        else:
            raise NotImplementedError

    def forward(self, inp):
        inp = (inp - self.mean) / self.std
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [
            nn.Dropout(),
        ] if (use_dropout) else []
        layers += [
            nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),
        ]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):

    def __init__(self, pretrained=True, requires_grad=False):
        super(vgg16, self).__init__()
        vgg_pretrained = models.vgg16(pretrained=False)
        vgg_pretrained.load_state_dict(torch.load(VGG_PRETRAIN_PATH, map_location='cpu'))
        vgg_pretrained_features = vgg_pretrained.features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple('VggOutputs', ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)
