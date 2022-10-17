import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from distutils.version import LooseVersion
from timm.models.layers import trunc_normal_

from vqfr.archs.vqganv1_arch import GeneralizedQuantizer, ResnetBlock, UpDownSample, VQGANDecoder, VQGANEncoder
from vqfr.ops.dcn import ModulatedDeformConvPack, modulated_deform_conv
from vqfr.utils import get_root_logger
from vqfr.utils.registry import ARCH_REGISTRY


class DCNv2Pack(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.
    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.
    Ref:
        Delving Deep into Deformable Alignment in Video Super-Resolution.
    """

    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 50:
            logger = get_root_logger()
            logger.warning(f'Offset abs mean is {offset_absmean}, larger than 50.')

        if LooseVersion(torchvision.__version__) >= LooseVersion('0.9.0'):
            return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                                 self.dilation, mask)
        else:
            return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding,
                                         self.dilation, self.groups, self.deformable_groups)


class TextureWarpingModule(nn.Module):

    def __init__(self, channel, cond_channels, cond_downscale_rate, deformable_groups, previous_offset_channel=0):
        super(TextureWarpingModule, self).__init__()
        self.downsample = UpDownSample(
            in_channels=cond_channels, out_channels=cond_channels, scale_factor=cond_downscale_rate, direction='down')
        self.offset_conv1 = nn.Sequential(
            nn.Conv2d(channel + cond_channels, channel, kernel_size=1),
            nn.GroupNorm(num_groups=32, num_channels=channel, eps=1e-6, affine=True), nn.SiLU(inplace=True),
            nn.Conv2d(channel, channel, groups=channel, kernel_size=7, padding=3),
            nn.GroupNorm(num_groups=32, num_channels=channel, eps=1e-6, affine=True), nn.SiLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=1))

        self.offset_conv2 = nn.Sequential(
            nn.Conv2d(channel + previous_offset_channel, channel, 3, 1, 1),
            nn.GroupNorm(num_groups=32, num_channels=channel, eps=1e-6, affine=True), nn.SiLU(inplace=True))
        self.dcn = DCNv2Pack(channel, channel, 3, padding=1, deformable_groups=deformable_groups)

    def forward(self, x_main, prior, previous_offset=None):
        prior = self.downsample(prior)
        offset = self.offset_conv1(torch.cat([prior, x_main], dim=1))
        if previous_offset is None:
            offset = self.offset_conv2(offset)
        else:
            offset = self.offset_conv2(torch.cat([offset, previous_offset], dim=1))

        warp_feat = self.dcn(x_main, offset)
        return warp_feat, offset


class MainDecoder(nn.Module):

    def __init__(self, base_channels, resolution_scale_rates, channel_multipliers, align_opt, align_from_patch):
        super(MainDecoder, self).__init__()
        self.log_size = int(math.log(align_from_patch, 2))

        self.channel_dict = {}
        self.resolution_scalerate_dict = {}

        resolution_scale_rates = resolution_scale_rates[::-1]

        for idx, scale in enumerate(range(self.log_size + 1)):
            self.channel_dict['Level_%d' % 2**scale] = channel_multipliers[idx] * base_channels
            self.resolution_scalerate_dict['Level_%d' % 2**scale] = resolution_scale_rates[idx]

        self.decoder_dict = nn.ModuleDict()
        self.pre_upsample_dict = nn.ModuleDict()

        for scale in range(self.log_size - 1, -1, -1):
            in_channel = self.channel_dict['Level_%d' %
                                           2**scale] if scale == self.log_size else self.channel_dict['Level_%d' %
                                                                                                      2**(scale + 1)]
            stage_channel = self.channel_dict['Level_%d' % 2**scale]
            upsample_rate = self.resolution_scalerate_dict['Level_%d' % 2**scale]

            self.decoder_dict['Level_%d' % 2**scale] = ResnetBlock(2 * stage_channel, stage_channel)
            self.pre_upsample_dict['Level_%d' % 2**scale] = UpDownSample(
                in_channel, stage_channel, upsample_rate, direction='up')

        self.align_func_dict = nn.ModuleDict()
        if align_opt is not None:
            for level_name, level_cfg in align_opt.items():
                level_idx = int(math.log(int(level_name.split('_')[1]), 2))
                previous_offset_channel = 0 if (
                    2**level_idx) == align_from_patch else self.channel_dict['Level_%d' % (2**(level_idx + 1))]
                channel = self.channel_dict[level_name]
                level_cfg['channel'] = channel
                level_cfg['previous_offset_channel'] = previous_offset_channel
                self.align_func_dict[level_name] = TextureWarpingModule(**level_cfg)

        self.conv_out_1 = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=self.channel_dict['Level_%d' % 2**0], eps=1e-6, affine=True),
            nn.SiLU(inplace=True))
        self.conv_out = nn.Sequential(
            nn.Conv2d(self.channel_dict['Level_%d' % 2**0] + 3, 3, kernel_size=3, stride=1, padding=1))

    def forward(self, dec_res_dict, prior, x_lq):

        x, offset = self.align_func_dict['Level_%d' % 2**self.log_size](dec_res_dict['Level_%d' % 2**self.log_size],
                                                                        prior)

        for scale in range(self.log_size - 1, -1, -1):
            x = self.pre_upsample_dict['Level_%d' % 2**scale](x)
            upsample_offset = F.interpolate(offset, scale_factor=2, align_corners=False, mode='bilinear') * 2
            warp_feat, offset = self.align_func_dict['Level_%d' % 2**scale](
                dec_res_dict['Level_%d' % 2**scale], prior, previous_offset=upsample_offset)
            x = self.decoder_dict['Level_%d' % 2**scale](torch.cat([x, warp_feat], dim=1))
        x = self.conv_out_1(x)
        x = self.conv_out(torch.cat([x, x_lq], dim=1))
        return x


class InpFeatConv(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(InpFeatConv, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, 3, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x


@ARCH_REGISTRY.register()
class VQFRv1(nn.Module):

    def __init__(self,
                 base_channels,
                 proj_patch_size,
                 resolution_scale_rates,
                 channel_multipliers,
                 encoder_num_blocks,
                 decoder_num_blocks,
                 quant_level,
                 inpfeat_extraction_opt,
                 align_opt,
                 align_from_patch,
                 quantizer_opt,
                 fix_keys=[]):
        super().__init__()

        self.inpfeat_extraction = InpFeatConv(**inpfeat_extraction_opt)

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

        self.main_decoder = MainDecoder(
            base_channels=base_channels,
            resolution_scale_rates=resolution_scale_rates,
            channel_multipliers=channel_multipliers,
            align_opt=align_opt,
            align_from_patch=align_from_patch)

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
        dec, feat_dict = self.decoder(quant_dict, return_feat=True)
        return dec, feat_dict

    def main_decode(self, dec_dict, prior, x_lq):
        dec = self.main_decoder(dec_dict, prior, x_lq)
        return dec

    def get_last_layer(self):
        return self.decoder.conv_out[-1].weight

    def get_main_last_layer(self):
        return self.main_decoder.conv_out[-1].weight

    def forward(self, x_lq, iters=-1, return_keys=('dec'), fidelity_ratio=None):

        inpfeat = self.inpfeat_extraction(x_lq)

        res = {}

        enc_dict = self.encode(x_lq)
        quant_dict, _, feat_dict = self.quant(enc_dict, iters=iters)

        if 'feat_dict' in return_keys:
            res['feat_dict'] = feat_dict

        if 'dec' in return_keys:
            dec, dec_feat_dict = self.decode(quant_dict)
            main_dec = self.main_decode(dec_feat_dict, inpfeat, x_lq)
            res['dec'] = dec
            res['main_dec'] = main_dec

        return res
