import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from vqfr.archs.quantizer_arch import build_quantizer
from vqfr.utils.registry import ARCH_REGISTRY


class Downsample(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode='constant', value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        x = self.conv(x)
        return x


class ResnetBlock(nn.Module):

    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=channels_in, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=(3, 3), stride=(1, 1), padding=1)

        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=channels_out, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(channels_out, channels_out, kernel_size=(3, 3), stride=(1, 1), padding=1)

        self.act = nn.SiLU(inplace=True)
        if channels_in != channels_out:
            self.residual_func = nn.Conv2d(channels_in, channels_out, kernel_size=1)
        else:
            self.residual_func = nn.Identity()

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)
        return x + self.residual_func(residual)


class AttnBlock(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class VQGANEncoder(nn.Module):

    def __init__(self, base_channels, channel_multipliers, num_blocks, use_enc_attention, code_dim):
        super(VQGANEncoder, self).__init__()

        self.num_levels = len(channel_multipliers)
        self.num_blocks = num_blocks

        self.conv_in = nn.Conv2d(
            3, base_channels * channel_multipliers[0], kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.blocks = nn.ModuleList()

        for i in range(self.num_levels):
            blocks = []
            if i == 0:
                channels_prev = base_channels * channel_multipliers[i]
            else:
                channels_prev = base_channels * channel_multipliers[i - 1]

            if i != 0:
                blocks.append(Downsample(channels_prev))

            channels = base_channels * channel_multipliers[i]
            blocks.append(ResnetBlock(channels_prev, channels))
            if i == self.num_levels - 1 and use_enc_attention:
                blocks.append(AttnBlock(channels))

            for j in range(self.num_blocks - 1):
                blocks.append(ResnetBlock(channels, channels))
                if i == self.num_levels - 1 and use_enc_attention:
                    blocks.append(AttnBlock(channels))

            self.blocks.append(nn.Sequential(*blocks))

        channels = base_channels * channel_multipliers[-1]
        if use_enc_attention:
            self.mid_blocks = nn.Sequential(
                ResnetBlock(channels, channels), AttnBlock(channels), ResnetBlock(channels, channels))
        else:
            self.mid_blocks = nn.Sequential(ResnetBlock(channels, channels), ResnetBlock(channels, channels))

        self.conv_out = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True), nn.SiLU(inplace=True),
            nn.Conv2d(channels, code_dim, kernel_size=3, padding=1))

    def forward(self, x):
        x = self.conv_in(x)
        for i in range(self.num_levels):
            x = self.blocks[i](x)
        x = self.mid_blocks(x)
        x = self.conv_out(x)
        return x


class VQGANDecoder(nn.Module):

    def __init__(self, base_channels, channel_multipliers, num_blocks, use_dec_attention, code_dim):
        super(VQGANDecoder, self).__init__()

        self.num_levels = len(channel_multipliers)
        self.num_blocks = num_blocks

        self.conv_in = nn.Conv2d(
            code_dim, base_channels * channel_multipliers[-1], kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.blocks = nn.ModuleList()

        channels = base_channels * channel_multipliers[-1]

        if use_dec_attention:
            self.mid_blocks = nn.Sequential(
                ResnetBlock(channels, channels), AttnBlock(channels), ResnetBlock(channels, channels))
        else:
            self.mid_blocks = nn.Sequential(ResnetBlock(channels, channels), ResnetBlock(channels, channels))

        for i in reversed(range(self.num_levels)):
            blocks = []

            if i == self.num_levels - 1:
                channels_prev = base_channels * channel_multipliers[i]
            else:
                channels_prev = base_channels * channel_multipliers[i + 1]

            if i != self.num_levels - 1:
                blocks.append(Upsample(channels_prev))

            channels = base_channels * channel_multipliers[i]
            blocks.append(ResnetBlock(channels_prev, channels))
            if i == self.num_levels - 1 and use_dec_attention:
                blocks.append(AttnBlock(channels))

            for j in range(self.num_blocks - 1):
                blocks.append(ResnetBlock(channels, channels))
                if i == self.num_levels - 1 and use_dec_attention:
                    blocks.append(AttnBlock(channels))
            self.blocks.append(nn.Sequential(*blocks))

        channels = base_channels * channel_multipliers[0]
        self.conv_out = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True), nn.SiLU(inplace=True),
            nn.Conv2d(channels, 3, kernel_size=3, padding=1))

    def forward(self, x, return_feat=False):
        dec_res = {}
        x = self.conv_in(x)
        x = self.mid_blocks(x)
        for i, level in enumerate(reversed(range(self.num_levels))):
            x = self.blocks[i](x)
            dec_res['Level_%d' % 2**level] = x
        x = self.conv_out(x)
        if return_feat:
            return x, dec_res
        else:
            return x


@ARCH_REGISTRY.register()
class VQGANv2(nn.Module):

    def __init__(self,
                 base_channels,
                 channel_multipliers,
                 num_enc_blocks,
                 use_enc_attention,
                 num_dec_blocks,
                 use_dec_attention,
                 code_dim,
                 quantizer_opt,
                 fix_keys=[]):
        super().__init__()

        self.encoder = VQGANEncoder(
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_blocks=num_enc_blocks,
            use_enc_attention=use_enc_attention,
            code_dim=code_dim)

        self.decoder = VQGANDecoder(
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_blocks=num_dec_blocks,
            use_dec_attention=use_dec_attention,
            code_dim=code_dim)

        self.quantizer = build_quantizer(quantizer_opt)

        self.apply(self._init_weights)

        for k, v in self.named_parameters():
            for fix_k in fix_keys:
                if fix_k in k:
                    v.requires_grad = False

    @torch.no_grad()
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

    def get_last_layer(self):
        return self.decoder.conv_out[-1].weight

    def forward(self, x, iters=-1, return_keys=('dec')):
        res = {}

        enc_feat = self.encoder(x)
        quant_feat, emb_loss, quant_index = self.quantizer(enc_feat)

        res['quant_feat'] = quant_feat
        res['quant_index'] = quant_index

        if 'dec' in return_keys:
            dec = self.decoder(quant_feat)
            res['dec'] = dec

        return res, emb_loss
