import importlib
import os
import torch
from copy import deepcopy
from os import path as osp

from vqfr.utils import get_root_logger, scandir
from vqfr.utils.registry import ARCH_REGISTRY

__all__ = ['build_network']

# automatically scan and import arch modules for registry
# scan all the files under the 'archs' folder and collect files ending with
# '_arch.py'
arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder) if v.endswith('_arch.py')]
# import all the arch modules
_arch_modules = [importlib.import_module(f'vqfr.archs.{file_name}') for file_name in arch_filenames]


def build_network(opt):
    opt = deepcopy(opt)
    network_type = opt.pop('type')
    net = ARCH_REGISTRY.get(network_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Network [{net.__class__.__name__}] is created.')
    return net


def init_alignment_model(model_name='FAN', device='cuda'):
    if model_name == 'FAN':
        model = ARCH_REGISTRY.get(model_name)()
        model_url = 'alignment_WFLW_4HG.pth'
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')
    logger = get_root_logger()
    logger.info(f'Network [{model.__class__.__name__}] is created.')
    model_path = os.path.join('experiments/pretrained_models/metric_weights', model_url)
    model.load_state_dict(torch.load(model_path, map_location='cpu')['state_dict'], strict=True)
    model.eval()
    model = model.to(device)
    return model
