import math
import torch
import torch.nn.functional as F
from torch import nn

from vqfr.ops.upfirdn2d import upfirdn2d
from vqfr.utils.registry import ARCH_REGISTRY


class NormStyleCode(nn.Module):

    def forward(self, x):
        """Normalize the style codes.
        Args:
            x (Tensor): Style codes with shape (b, c).
        Returns:
            Tensor: Normalized tensor.
        """
        return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)


def make_resample_kernel(k):
    """Make resampling kernel for UpFirDn.
    Args:
        k (list[int]): A list indicating the 1D resample kernel magnitude.
    Returns:
        Tensor: 2D resampled kernel.
    """
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]  # to 2D kernel, outer product
    # normalize
    k /= k.sum()
    return k


class UpFirDnUpsample(nn.Module):
    """Upsample, FIR filter, and downsample (upsampole version).
    References:
    1. https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.upfirdn.html  # noqa: E501
    2. http://www.ece.northwestern.edu/local-apps/matlabhelp/toolbox/signal/upfirdn.html  # noqa: E501
    Args:
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude.
        factor (int): Upsampling scale factor. Default: 2.
    """

    def __init__(self, resample_kernel, factor=2):
        super(UpFirDnUpsample, self).__init__()
        self.kernel = make_resample_kernel(resample_kernel) * (factor**2)
        self.factor = factor

        pad = self.kernel.shape[0] - factor
        self.pad = ((pad + 1) // 2 + factor - 1, pad // 2)

    def forward(self, x):
        out = upfirdn2d(x, self.kernel.type_as(x), up=self.factor, down=1, pad=self.pad)
        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}(factor={self.factor})')


class UpFirDnDownsample(nn.Module):
    """Upsample, FIR filter, and downsample (downsampole version).
    Args:
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude.
        factor (int): Downsampling scale factor. Default: 2.
    """

    def __init__(self, resample_kernel, factor=2):
        super(UpFirDnDownsample, self).__init__()
        self.kernel = make_resample_kernel(resample_kernel)
        self.factor = factor

        pad = self.kernel.shape[0] - factor
        self.pad = ((pad + 1) // 2, pad // 2)

    def forward(self, x):
        out = upfirdn2d(x, self.kernel.type_as(x), up=1, down=self.factor, pad=self.pad)
        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}(factor={self.factor})')


class UpFirDnSmooth(nn.Module):
    """Upsample, FIR filter, and downsample (smooth version).
    Args:
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude.
        upsample_factor (int): Upsampling scale factor. Default: 1.
        downsample_factor (int): Downsampling scale factor. Default: 1.
        kernel_size (int): Kernel size: Default: 1.
    """

    def __init__(self, resample_kernel, upsample_factor=1, downsample_factor=1, kernel_size=1):
        super(UpFirDnSmooth, self).__init__()
        self.upsample_factor = upsample_factor
        self.downsample_factor = downsample_factor
        self.kernel = make_resample_kernel(resample_kernel)
        if upsample_factor > 1:
            self.kernel = self.kernel * (upsample_factor**2)

        if upsample_factor > 1:
            pad = (self.kernel.shape[0] - upsample_factor) - (kernel_size - 1)
            self.pad = ((pad + 1) // 2 + upsample_factor - 1, pad // 2 + 1)
        elif downsample_factor > 1:
            pad = (self.kernel.shape[0] - downsample_factor) + (kernel_size - 1)
            self.pad = ((pad + 1) // 2, pad // 2)
        else:
            raise NotImplementedError

    def forward(self, x):
        out = upfirdn2d(x, self.kernel.type_as(x), up=1, down=1, pad=self.pad)
        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}(upsample_factor={self.upsample_factor}'
                f', downsample_factor={self.downsample_factor})')


class ConvLayer(nn.Sequential):
    """Conv Layer used in StyleGAN2 Discriminator.
    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Kernel size.
        downsample (bool): Whether downsample by a factor of 2.
            Default: False.
        resample_kernel (list[int]): A list indicating the 1D resample
            kernel magnitude. A cross production will be applied to
            extent 1D resample kernel to 2D resample kernel.
            Default: (1, 3, 3, 1).
        bias (bool): Whether with bias. Default: True.
        activate (bool): Whether use activateion. Default: True.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 downsample=False,
                 resample_kernel=(1, 3, 3, 1),
                 bias=True,
                 activate=True):
        layers = []
        # downsample
        if downsample:
            layers.append(
                UpFirDnSmooth(resample_kernel, upsample_factor=1, downsample_factor=2, kernel_size=kernel_size))
            stride = 2
            self.padding = 0
        else:
            stride = 1
            self.padding = kernel_size // 2
        # conv
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=self.padding, bias=bias))
        # activation
        if activate:
            layers.append(nn.LeakyReLU(0.2))

        super(ConvLayer, self).__init__(*layers)


class ResBlock(nn.Module):
    """Residual block used in StyleGAN2 Discriminator.
    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        resample_kernel (list[int]): A list indicating the 1D resample
            kernel magnitude. A cross production will be applied to
            extent 1D resample kernel to 2D resample kernel.
            Default: (1, 3, 3, 1).
    """

    def __init__(self, in_channels, out_channels, resample_kernel=(1, 3, 3, 1)):
        super(ResBlock, self).__init__()

        self.conv1 = ConvLayer(in_channels, in_channels, 3, bias=True, activate=True)
        self.conv2 = ConvLayer(
            in_channels, out_channels, 3, downsample=True, resample_kernel=resample_kernel, bias=True, activate=True)
        self.skip = ConvLayer(
            in_channels, out_channels, 1, downsample=True, resample_kernel=resample_kernel, bias=False, activate=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        skip = self.skip(x)
        out = (out + skip) / math.sqrt(2)
        return out


class EqualConv2d(nn.Module):
    """Equalized Linear as StyleGAN2.
    Args:
        in_channels (int): Channel number of the input.
        out_channels (int): Channel number of the output.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution. Default: 1
        padding (int): Zero-padding added to both sides of the input.
            Default: 0.
        bias (bool): If ``True``, adds a learnable bias to the output.
            Default: ``True``.
        bias_init_val (float): Bias initialized value. Default: 0.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, bias_init_val=0):
        super(EqualConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.scale = 1 / math.sqrt(in_channels * kernel_size**2)

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels).fill_(bias_init_val))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        out = F.conv2d(
            x,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}(in_channels={self.in_channels}, '
                f'out_channels={self.out_channels}, '
                f'kernel_size={self.kernel_size},'
                f' stride={self.stride}, padding={self.padding}, '
                f'bias={self.bias is not None})')


class ScaledLeakyReLU(nn.Module):
    """Scaled LeakyReLU.
    Args:
        negative_slope (float): Negative slope. Default: 0.2.
    """

    def __init__(self, negative_slope=0.2):
        super(ScaledLeakyReLU, self).__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        out = F.leaky_relu(x, negative_slope=self.negative_slope)
        return out * math.sqrt(2)


@ARCH_REGISTRY.register()
class StyleGAN2Discriminator(nn.Module):
    """StyleGAN2 Discriminator.
    Args:
        out_size (int): The spatial size of outputs.
        channel_multiplier (int): Channel multiplier for large networks of
            StyleGAN2. Default: 2.
        resample_kernel (list[int]): A list indicating the 1D resample kernel
            magnitude. A cross production will be applied to extent 1D resample
            kernel to 2D resample kernel. Default: (1, 3, 3, 1).
        stddev_group (int): For group stddev statistics. Default: 4.
        narrow (float): Narrow ratio for channels. Default: 1.0.
    """

    def __init__(self, out_size, channel_multiplier=2, resample_kernel=(1, 3, 3, 1), stddev_group=4, narrow=1):
        super(StyleGAN2Discriminator, self).__init__()

        channels = {
            '4': int(512 * narrow),
            '8': int(512 * narrow),
            '16': int(512 * narrow),
            '32': int(512 * narrow),
            '64': int(256 * channel_multiplier * narrow),
            '128': int(128 * channel_multiplier * narrow),
            '256': int(64 * channel_multiplier * narrow),
            '512': int(32 * channel_multiplier * narrow),
            '1024': int(16 * channel_multiplier * narrow)
        }

        log_size = int(math.log(out_size, 2))

        conv_body = [ConvLayer(3, channels[f'{out_size}'], 1, bias=True, activate=True)]

        in_channels = channels[f'{out_size}']
        for i in range(log_size, 2, -1):
            out_channels = channels[f'{2**(i - 1)}']
            conv_body.append(ResBlock(in_channels, out_channels, resample_kernel))
            in_channels = out_channels
        self.conv_body = nn.Sequential(*conv_body)

        self.final_conv = ConvLayer(in_channels + 1, channels['4'], 3, bias=True, activate=True)
        self.final_linear = nn.Sequential(
            nn.Linear(channels['4'] * 4 * 4, channels['4'], bias=True), nn.LeakyReLU(0.2),
            nn.Linear(channels['4'], 1, bias=True))
        self.stddev_group = stddev_group
        self.stddev_feat = 1

    def forward(self, x):
        out = self.conv_body(x)

        b, c, h, w = out.shape
        # concatenate a group stddev statistics to out
        group = min(b, self.stddev_group)  # Minibatch must be divisible by (or smaller than) group_size
        stddev = out.view(group, -1, self.stddev_feat, c // self.stddev_feat, h, w)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, h, w)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)
        out = out.view(b, -1)
        out = self.final_linear(out)

        return out
