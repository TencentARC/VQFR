from .losses import GANLoss, L1Loss, MSELoss, MultiQuantMatchLoss, build_loss
from .lpips import LPIPS

__all__ = ['L1Loss', 'MSELoss', 'LPIPS', 'GANLoss', 'MultiQuantMatchLoss', 'build_loss']
