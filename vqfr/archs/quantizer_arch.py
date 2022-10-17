import torch
import torch.distributed as dist
import torch.nn as nn
from copy import deepcopy
from einops import rearrange
from sklearn import cluster

from vqfr.utils import all_gather, get_root_logger
from vqfr.utils.registry import QUANTIZER_REGISTRY


def build_quantizer(opt):
    """Build performance evaluator from options.
    Args:
        opt (dict): Configuration.
    """
    opt = deepcopy(opt)
    quantizer_type = opt.pop('type')
    quantizer = QUANTIZER_REGISTRY.get(quantizer_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Quantizer [{quantizer.__class__.__name__}] is created.')
    return quantizer


class NewReservoirSampler(nn.Module):

    def __init__(self, num_samples=1024):
        super(NewReservoirSampler, self).__init__()
        self.n = num_samples
        self.register_buffer('buffer', None, persistent=False)
        self.reset()

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        buffer_key = prefix + 'buffer'
        if buffer_key in state_dict:
            self.buffer = state_dict[buffer_key]
        return super(NewReservoirSampler, self)._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def reset(self):
        self.i = 0
        self.buffer = None

    def add(self, samples):
        cuda_device = samples.device
        samples = samples.detach()
        if self.buffer is None:
            self.buffer = torch.empty(self.n, samples.size(-1), device=samples.device)
        buffer = self.buffer
        if self.i < self.n:
            slots = self.n - self.i
            add_samples = samples[:slots]
            samples = samples[slots:]
            buffer[self.i:self.i + len(add_samples)] = add_samples
            self.i += len(add_samples)

        if len(samples) > 0:
            positions = torch.randint(0, self.i, size=(len(samples), ), device=cuda_device)
            sync_positions = all_gather(positions)[0].cpu()

            for s, pos_idx in zip(samples, sync_positions):
                # warning, includes right end too.
                self.i += 1
                if pos_idx < len(buffer):
                    buffer[pos_idx] = s

    def contents(self):
        return self.buffer[:self.i]


@QUANTIZER_REGISTRY.register()
class L2VectorQuantizerKmeans(nn.Module):

    def __init__(self, num_code, in_dim, code_dim, reservoir_size, reestimate_iters, reestimate_maxiters, warmup_iters):
        super().__init__()
        self.num_code = num_code
        self.code_dim = code_dim
        self.beta = 0.25

        self.reestimate_iters = reestimate_iters
        self.reestimate_maxiters = reestimate_maxiters
        self.warmup_iters = warmup_iters

        self.quant_conv = nn.Conv2d(in_dim, code_dim, 1)
        self.post_quant_conv = nn.Conv2d(code_dim, in_dim, 1)

        self.embedding = nn.Embedding(self.num_code, self.code_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_code, 1.0 / self.num_code)

        self.register_buffer('usage', torch.zeros(self.num_code, dtype=torch.int), persistent=False)

        self.reestimation_reservoir = NewReservoirSampler(reservoir_size)

    def get_distance(self, z, embedding):
        # (b h w) c
        z_flattened = z.view(-1, self.code_dim)
        distance = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding ** 2, dim=1) - 2 * \
            torch.einsum('b c, d c -> b d', z_flattened, embedding)
        return distance

    def compute_codebook_loss(self, z_quant, z):
        loss = torch.mean((z_quant.detach() - z)**2) + self.beta * torch.mean((z_quant - z.detach())**2)
        return loss

    def reset_usage(self):
        self.usage = self.usage * 0

    def get_usage(self):
        codebook_usage = 1.0 * (self.num_code - (self.usage == 0).sum()) / self.num_code
        return codebook_usage

    def reestimate(self):
        logger = get_root_logger()
        num_clusters = self.embedding.weight.shape[0]
        encodings = self.reestimation_reservoir.contents()
        if encodings.shape[0] < num_clusters:
            logger.info('Skipping reestimation, too few samples')
            return
        else:
            logger.info('begin reestimation')
            encodings = encodings.cpu().numpy()
            clustered, *_ = cluster.k_means(encodings, num_clusters, random_state=0)
            self.embedding.weight.data[...] = torch.tensor(clustered).to(self.embedding.weight.device)
            print(self.embedding.weight.sum())
            self.reestimation_reservoir.reset()
            logger.info('end reestimation')

    def forward(self, z, iters=-1):
        z_conv = self.quant_conv(z)
        z = rearrange(z_conv, 'b c h w -> b h w c').contiguous()

        if self.training and iters < self.reestimate_maxiters:
            if dist.is_initialized():
                z_gather = all_gather(z)
                z_gather = torch.cat(z_gather, dim=0)
            else:
                z_gather = z

            self.reestimation_reservoir.add(z_gather.view(-1, z_gather.size(-1)))

            if iters % self.reestimate_iters == 0:
                self.reestimate()

        if self.training and iters < self.warmup_iters:
            z_quant = z
            loss = torch.tensor(0.0).to(z_quant.device)
        else:
            embedding = self.embedding.weight
            distance = self.get_distance(z, embedding)

            min_encoding_indices = torch.argmin(distance, dim=1)

            if not self.training:
                for idx in range(self.num_code):
                    self.usage[idx] += (min_encoding_indices == idx).sum()

            z_quant = self.embedding(min_encoding_indices).view(z.shape)
            loss = self.compute_codebook_loss(z_quant, z)
            # preserve gradients
            z_quant = z + (z_quant - z).detach()
            # reshape back to match original input shape
        z_quant_before_conv = rearrange(z_quant, 'b h w c -> b c h w').contiguous()
        z_quant = self.post_quant_conv(z_quant_before_conv)
        return z_quant, loss, {'z_conv': z_conv, 'z_quant_before_conv': z_quant_before_conv}


@QUANTIZER_REGISTRY.register()
class L2VectorQuantizer(nn.Module):

    def __init__(self, num_code, code_dim, spatial_size):
        super().__init__()
        self.num_code = num_code
        self.code_dim = code_dim
        self.spatial_size = spatial_size

        self.beta = 0.25

        self.embedding = nn.Embedding(self.num_code, self.code_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_code, 1.0 / self.num_code)

        self.register_buffer('usage', torch.zeros(self.num_code, dtype=torch.int), persistent=False)

    def get_feature(self, indices):
        quant = self.embedding(indices)
        quant = rearrange(quant, 'b (h w) c -> b c h w', h=self.spatial_size[0])
        return quant

    def get_distance(self, z, embedding):
        # (b h w) c
        z_flattened = z.view(-1, self.code_dim)
        distance = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding ** 2, dim=1) - 2 * \
            torch.einsum('b c, d c -> b d', z_flattened, embedding)
        return distance

    def compute_codebook_loss(self, z_quant, z):
        loss = torch.mean((z_quant.detach() - z)**2) + self.beta * torch.mean((z_quant - z.detach())**2)
        return loss

    def reset_usage(self):
        self.usage = self.usage * 0

    def get_usage(self):
        codebook_usage = 1.0 * (self.num_code - (self.usage == 0).sum()) / self.num_code
        return codebook_usage

    def forward(self, z, iters=-1):

        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        embedding = self.embedding.weight

        distance = self.get_distance(z, embedding)

        min_encoding_indices = torch.argmin(distance, dim=1)

        if not self.training:
            for idx in range(self.num_code):
                self.usage[idx] += (min_encoding_indices == idx).sum()

        z_quant = self.embedding(min_encoding_indices).view(z.shape)

        loss = self.compute_codebook_loss(z_quant, z)
        # preserve gradients
        z_quant = z + (z_quant - z).detach()
        # reshape back to match original input shape
        z_quant = rearrange(z_quant, 'b h w c -> b c h w').contiguous()
        min_encoding_indices = rearrange(min_encoding_indices, '(b n)->b n', b=z_quant.shape[0])
        return z_quant, loss, (min_encoding_indices)
