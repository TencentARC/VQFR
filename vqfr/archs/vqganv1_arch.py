import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from vqfr.archs.quantizer_arch import build_quantizer
from vqfr.utils.registry import ARCH_REGISTRY


class UpDownSample(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor, direction):
        super().__init__()
        self.scale_factor = scale_factor
        self.direction = direction
        if not self.scale_factor == 1:
            self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        assert direction in ['up', 'down']

    def forward(self, x):
        if not self.scale_factor == 1:
            _, _, h, w = x.shape
            if self.direction == 'up':
                new_h = int(self.scale_factor * h)
                new_w = int(self.scale_factor * w)
                x = self.conv(x)
                x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
            else:
                new_h = int(h / self.scale_factor)
                new_w = int(w / self.scale_factor)
                x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
                x = self.conv(x)
        return x


class ResnetBlock(nn.Module):

    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.act = nn.SiLU(inplace=True)

        if self.in_channels != self.out_channels:
            self.shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = self.act(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.shortcut_conv(x)
        return x + h


class VQGANEncoder(nn.Module):

    def __init__(self, base_channels, proj_patch_size, resolution_scale_rates, channel_multipliers, encoder_num_blocks,
                 quant_level):
        super(VQGANEncoder, self).__init__()
        self.log_size = int(math.log(proj_patch_size, 2))
        self.channel_dict = {}
        self.resolution_scalerate_dict = {}

        for idx, scale in enumerate(range(self.log_size + 1)):
            self.channel_dict['Level_%d' % 2**scale] = channel_multipliers[idx] * base_channels
            self.resolution_scalerate_dict['Level_%d' % 2**scale] = resolution_scale_rates[idx]

        self.conv_in = torch.nn.Conv2d(3, self.channel_dict['Level_%d' % 2**0], kernel_size=3, stride=1, padding=1)
        self.encoder_dict = nn.ModuleDict()
        self.pre_downsample_dict = nn.ModuleDict()

        for scale in range(self.log_size + 1):
            in_channel = self.channel_dict['Level_%d' % 2**scale] if scale == 0 else self.channel_dict['Level_%d' %
                                                                                                       2**(scale - 1)]
            stage_channel = self.channel_dict['Level_%d' % 2**scale]
            downsample_rate = self.resolution_scalerate_dict['Level_%d' % 2**scale]
            self.encoder_dict['Level_%d' % 2**scale] = nn.Sequential(
                *[ResnetBlock(stage_channel, stage_channel) for _ in range(encoder_num_blocks)])
            self.pre_downsample_dict['Level_%d' % 2**scale] = UpDownSample(
                in_channel, stage_channel, downsample_rate, direction='down')

        self.quant_level = quant_level
        self.enc_convout_dict = nn.ModuleDict()
        for level_name in self.quant_level:
            channel = self.channel_dict[level_name]
            self.enc_convout_dict[level_name] = nn.Sequential(
                nn.GroupNorm(num_groups=32, num_channels=channel, eps=1e-6, affine=True), nn.SiLU(inplace=True),
                nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        enc_res_dict = {}
        x = self.conv_in(x)
        for scale in range(self.log_size + 1):
            x = self.pre_downsample_dict['Level_%d' % 2**scale](x)
            x = self.encoder_dict['Level_%d' % 2**scale](x)
            if 'Level_%d' % 2**scale in self.quant_level:
                enc_res_dict['Level_%d' % (2**scale)] = x
        for level_name, level_feat in enc_res_dict.items():
            enc_res_dict[level_name] = self.enc_convout_dict[level_name](level_feat)
        return enc_res_dict


class VQGANDecoder(nn.Module):

    def __init__(self, base_channels, proj_patch_size, resolution_scale_rates, channel_multipliers, decoder_num_blocks):
        super(VQGANDecoder, self).__init__()
        self.log_size = int(math.log(proj_patch_size, 2))

        self.channel_dict = {}
        self.resolution_scalerate_dict = {}

        resolution_scale_rates = resolution_scale_rates[::-1]

        for idx, scale in enumerate(range(self.log_size + 1)):
            self.channel_dict['Level_%d' % 2**scale] = channel_multipliers[idx] * base_channels
            self.resolution_scalerate_dict['Level_%d' % 2**scale] = resolution_scale_rates[idx]

        self.decoder_dict = nn.ModuleDict()
        self.pre_upsample_dict = nn.ModuleDict()

        for scale in range(self.log_size, -1, -1):
            in_channel = self.channel_dict['Level_%d' %
                                           2**scale] if scale == self.log_size else self.channel_dict['Level_%d' %
                                                                                                      2**(scale + 1)]
            stage_channel = self.channel_dict['Level_%d' % 2**scale]
            upsample_rate = self.resolution_scalerate_dict['Level_%d' % 2**scale]

            self.decoder_dict['Level_%d' % 2**scale] = nn.Sequential(
                *[ResnetBlock(stage_channel, stage_channel) for _ in range(decoder_num_blocks)])
            self.pre_upsample_dict['Level_%d' % 2**scale] = UpDownSample(
                in_channel, stage_channel, upsample_rate, direction='up')

        self.conv_out = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=self.channel_dict['Level_%d' % 2**0], eps=1e-6, affine=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.channel_dict['Level_%d' % 2**0], 3, kernel_size=3, stride=1, padding=1))

    def forward(self, quant_res_dict, return_feat=False):

        dec_res = {}

        x = quant_res_dict['Level_%d' % 2**self.log_size]  # set quant final as the begin of decoder

        for scale in range(self.log_size, -1, -1):
            x = self.pre_upsample_dict['Level_%d' % 2**scale](x)
            x = self.decoder_dict['Level_%d' % 2**scale](x)
            dec_res['Level_%d' % 2**scale] = x
        x = self.conv_out(x)
        if return_feat:
            return x, dec_res
        else:
            return x


class GeneralizedQuantizer(nn.Module):

    def __init__(self, quantizer_opt):
        super(GeneralizedQuantizer, self).__init__()
        self.quantize_dict = nn.ModuleDict()
        for level_name, level_opt in quantizer_opt.items():
            self.quantize_dict[level_name] = build_quantizer(level_opt)

    def forward(self, enc_dict, iters=-1):
        res_dict = {}
        extra_info_dict = {}

        emb_loss_total = 0.0

        for level_name in self.quantize_dict.keys():
            h_q, emb_loss, extra_info = self.quantize_dict[level_name](enc_dict[level_name], iters=iters)
            res_dict[level_name] = h_q
            emb_loss_total += emb_loss
            extra_info_dict[level_name] = extra_info
        return res_dict, emb_loss_total, extra_info_dict

    def reset_usage(self):
        for level_name, quantizer in self.quantize_dict.items():
            if hasattr(quantizer, 'reset_usage'):
                quantizer.reset_usage()

    def get_usage(self):
        res = {}
        for level_name, quantizer in self.quantize_dict.items():
            if hasattr(quantizer, 'get_usage'):
                usage = quantizer.get_usage()
                res[level_name] = '%.2f' % usage
        return res


@ARCH_REGISTRY.register()
class VQGANv1(nn.Module):

    def __init__(self,
                 base_channels,
                 proj_patch_size,
                 resolution_scale_rates,
                 channel_multipliers,
                 encoder_num_blocks,
                 decoder_num_blocks,
                 quant_level,
                 quantizer_opt,
                 fix_keys=[]):
        super().__init__()

        self.encoder = VQGANEncoder(
            base_channels=base_channels,
            proj_patch_size=proj_patch_size,
            resolution_scale_rates=resolution_scale_rates,
            channel_multipliers=channel_multipliers,
            encoder_num_blocks=encoder_num_blocks,
            quant_level=quant_level)

        self.decoder = VQGANDecoder(
            base_channels=base_channels,
            proj_patch_size=proj_patch_size,
            resolution_scale_rates=resolution_scale_rates,
            channel_multipliers=channel_multipliers,
            decoder_num_blocks=decoder_num_blocks)

        self.quantizer = GeneralizedQuantizer(quantizer_opt)

        self.apply(self._init_weights)

        for k, v in self.named_parameters():
            for fix_k in fix_keys:
                if fix_k in k:
                    v.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def quant(self, enc_dict, iters=-1):
        quant_dict, emb_loss, info_dict = self.quantizer(enc_dict, iters=iters)
        return quant_dict, emb_loss, info_dict

    def encode(self, x):
        enc_dict = self.encoder(x)
        return enc_dict

    def decode(self, quant_dict):
        dec = self.decoder(quant_dict)
        return dec

    def get_last_layer(self):
        return self.decoder.conv_out[-1].weight

    def forward(self, x, iters=-1, return_keys=('dec')):
        res = {}

        enc_dict = self.encode(x)
        quant_dict, quant_loss, feat_dict = self.quant(enc_dict, iters=iters)

        if 'feat_dict' in return_keys:
            res['feat_dict'] = feat_dict

        if 'dec' in return_keys:
            dec = self.decode(quant_dict)
            res['dec'] = dec

        return res, quant_loss
