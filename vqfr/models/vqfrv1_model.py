import torch
import torch.distributed as dist
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from vqfr.archs import build_network
from vqfr.losses import build_loss
from vqfr.losses.losses import r1_penalty
from vqfr.metrics import calculate_metric
from vqfr.utils import get_root_logger, imwrite, tensor2img
from vqfr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


@MODEL_REGISTRY.register()
class VQFRv1Model(BaseModel):
    """VQGAN_BASE_Model"""

    def __init__(self, opt):
        super(VQFRv1Model, self).__init__(opt)

        # define network
        if 'network_g' in self.opt:
            self.net_g = build_network(opt['network_g'])
            self.net_g = self.model_to_device(self.net_g)
            self.print_network(self.net_g)

        self.net_sr = build_network(opt['network_sr'])
        self.net_sr = self.model_to_device(self.net_sr)
        self.print_network(self.net_sr)

        # define network net_d
        if 'network_d_global' in self.opt:
            self.net_d_global = build_network(self.opt['network_d_global'])
            if self.opt.get('syncbn') is True and self.opt['num_gpu'] > 1:
                self.net_d_global = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                    self.net_d_global)  # to avoid broadcast buffer error
            self.net_d_global = self.model_to_device(self.net_d_global)
            self.print_network(self.net_d_global)

        # define network net_d
        if 'network_d_local' in self.opt:
            self.net_d_local = build_network(self.opt['network_d_local'])
            if self.opt.get('syncbn') is True and self.opt['num_gpu'] > 1:
                self.net_d_local = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                    self.net_d_local)  # to avoid broadcast buffer error
            self.net_d_local = self.model_to_device(self.net_d_local)
            self.print_network(self.net_d_local)

        # define network net_d
        if 'network_d_main_global' in self.opt:
            self.net_d_main_global = build_network(self.opt['network_d_main_global'])
            if self.opt.get('syncbn') is True and self.opt['num_gpu'] > 1:
                self.net_d_main_global = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                    self.net_d_main_global)  # to avoid broadcast buffer error
            self.net_d_main_global = self.model_to_device(self.net_d_main_global)
            self.print_network(self.net_d_main_global)

        # define network net_d
        if 'network_d_main_local' in self.opt:
            self.net_d_main_local = build_network(self.opt['network_d_main_local'])
            if self.opt.get('syncbn') is True and self.opt['num_gpu'] > 1:
                self.net_d_main_local = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                    self.net_d_main_local)  # to avoid broadcast buffer error
            self.net_d_main_local = self.model_to_device(self.net_d_main_local)
            self.print_network(self.net_d_main_local)

        # load pretrained models

        logger = get_root_logger()

        load_path = self.opt['path'].get('pretrain_network_d_main_local', None)
        if load_path is not None:
            logger.info('Loading net_d_main_local:')
            param_key = 'params'
            self.load_network(self.net_d_main_local, load_path, True, param_key)

        load_path = self.opt['path'].get('pretrain_network_d_main_global', None)
        if load_path is not None:
            logger.info('Loading net_d_main_global:')
            param_key = 'params'
            self.load_network(self.net_d_main_global, load_path, True, param_key)

        load_path = self.opt['path'].get('pretrain_network_d_global', None)
        if load_path is not None:
            logger.info('Loading net_d_global:')
            param_key = 'params'
            self.load_network(self.net_d_global, load_path, True, param_key)

        load_path = self.opt['path'].get('pretrain_network_d_local', None)
        if load_path is not None:
            logger.info('Loading net_d_local:')
            param_key = 'params'
            self.load_network(self.net_d_local, load_path, True, param_key)

        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            logger.info('Loading net_g:')
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        load_path = self.opt['path'].get('pretrain_network_sr', None)
        if load_path is not None:
            logger.info('Loading net_sr:')
            param_key = self.opt['path'].get('param_key_sr', 'params')
            self.load_network(self.net_sr, load_path, self.opt['path'].get('strict_load_sr', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_sr.train()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        # define losses
        if train_opt.get('pixel_main_opt'):
            self.cri_pix_main = build_loss(train_opt['pixel_main_opt']).to(self.device)
        else:
            self.cri_pix_main = None

        # define losses
        if train_opt.get('latent_opt'):
            self.cri_latent = build_loss(train_opt['latent_opt']).to(self.device)
        else:
            self.cri_latent = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('global_gan_opt'):
            self.cri_globalgan = build_loss(train_opt['global_gan_opt']).to(self.device)
        else:
            self.cri_globalgan = None

        if train_opt.get('patch_gan_opt'):
            self.cri_patchgan = build_loss(train_opt['patch_gan_opt']).to(self.device)
        else:
            self.cri_patchgan = None

        self.r1_reg_weight = train_opt['r1_reg_weight']  # for discriminator
        self.net_d_reg_every = train_opt['net_d_reg_every']
        self.generator_d_global_weight = train_opt['generator_d_global_weight']
        self.generator_d_local_weight = train_opt['generator_d_local_weight']

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_sr.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_sr'].pop('type')
        self.optimizer_sr = self.get_optimizer(optim_type, optim_params, **train_opt['optim_sr'])
        self.optimizers.append(self.optimizer_sr)

        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')

        self.optimizer_d_local = self.get_optimizer(optim_type, self.net_d_local.parameters(), **train_opt['optim_d'])
        self.optimizer_d_global = self.get_optimizer(optim_type, self.net_d_global.parameters(), **train_opt['optim_d'])
        self.optimizer_d_main_local = self.get_optimizer(optim_type, self.net_d_main_local.parameters(),
                                                         **train_opt['optim_d'])
        self.optimizer_d_main_global = self.get_optimizer(optim_type, self.net_d_main_global.parameters(),
                                                          **train_opt['optim_d'])
        self.optimizers.extend([
            self.optimizer_d_local, self.optimizer_d_global, self.optimizer_d_main_local, self.optimizer_d_main_global
        ])

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True, allow_unused=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True, allow_unused=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True, allow_unused=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True, allow_unused=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight

    def optimize_parameters(self, current_iter):
        loss_dict = OrderedDict()

        self.optimizer_sr.zero_grad()

        # get hr quant_latent
        with torch.no_grad():
            hq_res, _ = self.net_g(self.gt, return_keys=('feat_dict'))

        # get_lq quant_latent
        lq_res = self.net_sr(self.lq, return_keys=('dec', 'feat_dict'))

        l_total_g = 0.0

        # pixel reconstruction loss
        if self.cri_pix:
            l_rec = self.cri_pix(lq_res['dec'], self.gt)
            loss_dict['l_rec'] = l_rec.detach().mean()
            l_total_g += l_rec

        # pixel reconstruction loss
        if self.cri_pix_main:
            l_main_rec = self.cri_pix_main(lq_res['main_dec'], self.gt)
            loss_dict['l_main_rec'] = l_main_rec.detach().mean()
            l_total_g += l_main_rec

        if self.cri_latent:
            l_latent = self.cri_latent(lq_res['feat_dict'], hq_res['feat_dict'])
            loss_dict['l_latent'] = l_latent.detach().mean()
            l_total_g += l_latent

        # perceptual loss
        if self.cri_perceptual:
            l_g_percep, l_g_style = self.cri_perceptual(lq_res['dec'], self.gt)

            if l_g_percep is not None:
                l_total_g += l_g_percep
                loss_dict['l_g_percep'] = l_g_percep.detach().mean()
            if l_g_style is not None:
                l_total_g += l_g_style
                loss_dict['l_g_style'] = l_g_style.detach().mean()

            l_g_main_percep, l_g_main_style = self.cri_perceptual(lq_res['main_dec'], self.gt)
            if l_g_main_percep is not None:
                l_total_g += l_g_main_percep
                loss_dict['l_g_main_percep'] = l_g_main_percep.detach().mean()
            if l_g_main_style is not None:
                l_total_g += l_g_main_style
                loss_dict['l_g_main_style'] = l_g_main_style.detach().mean()

        if current_iter > self.opt['train'].get('gan_start_iter', 0):
            # wgan loss with softplus (non-saturating loss) for generator

            for p in self.net_d_global.parameters():
                p.requires_grad = False
            fake_pred = self.net_d_global(lq_res['dec'])
            l_g_gan = self.cri_globalgan(fake_pred, True, is_disc=False)
            l_g_gan = self.generator_d_global_weight * l_g_gan
            loss_dict['l_g_gan_global'] = l_g_gan.detach().mean()
            l_total_g += l_g_gan

            for p in self.net_d_local.parameters():
                p.requires_grad = False
            fake_pred = self.net_d_local(lq_res['dec'])
            l_g_gan = self.cri_patchgan(fake_pred, True, is_disc=False)

            d_weight = self.generator_d_local_weight * self.calculate_adaptive_weight(
                l_rec + l_g_percep + l_g_style,
                l_g_gan,
                last_layer=self.net_sr.module.get_last_layer()
                if dist.is_initialized() else self.net_sr.get_last_layer())
            l_g_gan = d_weight * l_g_gan
            loss_dict['l_g_gan_local'] = l_g_gan.detach().mean()
            l_total_g += l_g_gan

        if current_iter > self.opt['train'].get('main_gan_start_iter', 0):

            for p in self.net_d_main_global.parameters():
                p.requires_grad = False
            fake_pred = self.net_d_main_global(lq_res['main_dec'])
            l_g_main_gan = self.cri_globalgan(fake_pred, True, is_disc=False)
            l_g_main_gan = self.generator_d_global_weight * l_g_main_gan
            loss_dict['l_g_main_gan_global'] = l_g_main_gan.detach().mean()
            l_total_g += l_g_main_gan

            for p in self.net_d_main_local.parameters():
                p.requires_grad = False
            fake_pred = self.net_d_main_local(lq_res['main_dec'])
            l_g_main_gan = self.cri_patchgan(fake_pred, True, is_disc=False)
            d_weight_main = self.generator_d_local_weight * self.calculate_adaptive_weight(
                l_main_rec + l_g_main_percep + l_g_main_style,
                l_g_main_gan,
                last_layer=self.net_sr.module.get_main_last_layer()
                if dist.is_initialized() else self.net_sr.get_main_last_layer())
            l_g_main_gan = d_weight_main * l_g_main_gan
            loss_dict['l_g_main_gan_local'] = l_g_main_gan.detach().mean()
            l_total_g += l_g_main_gan

        loss_dict['l_total_g'] = l_total_g.detach().mean()
        l_total_g.backward()
        self.optimizer_sr.step()

        if current_iter > self.opt['train'].get('gan_start_iter', 0):

            # global main
            # -------------------------------------------------------------------------
            for p in self.net_d_global.parameters():
                p.requires_grad = True
            self.optimizer_d_global.zero_grad()

            fake_pred = self.net_d_global(lq_res['dec'].detach())
            real_pred = self.net_d_global(self.gt)
            # wgan loss with softplus (logistic loss) for discriminator
            l_d = (
                self.cri_globalgan(real_pred, True, is_disc=True) + self.cri_globalgan(fake_pred, False, is_disc=True))
            loss_dict['l_d_global'] = l_d.detach().mean()
            # In wgan, real_score should be positive and fake_score should be
            # negative
            loss_dict['real_score_global'] = real_pred.detach().mean()
            loss_dict['fake_score_global'] = fake_pred.detach().mean()

            l_d.backward()
            if current_iter % self.net_d_reg_every == 0:
                self.gt.requires_grad = True
                real_pred = self.net_d_global(self.gt)
                l_d_r1 = r1_penalty(real_pred, self.gt)
                l_d_r1 = (self.r1_reg_weight / 2 * l_d_r1 * self.net_d_reg_every + 0 * real_pred[0])
                # TODO: why do we need to add 0 * real_pred, otherwise, a runtime
                # error will arise: RuntimeError: Expected to have finished
                # reduction in the prior iteration before starting a new one.
                # This error indicates that your module has parameters that were
                # not used in producing loss.
                loss_dict['l_d_r1'] = l_d_r1.detach().mean()
                l_d_r1.backward()
            self.optimizer_d_global.step()
            # ----------------------------------------------------------------------------------------------------------

            # ----------------------------------------------------------------------------------------------------------
            for p in self.net_d_local.parameters():
                p.requires_grad = True
            self.optimizer_d_local.zero_grad()
            fake_pred = self.net_d_local(lq_res['dec'].detach())
            real_pred = self.net_d_local(self.gt)
            l_d = 0.5 * (
                self.cri_patchgan(real_pred, True, is_disc=True) + self.cri_patchgan(fake_pred, False, is_disc=True))
            loss_dict['l_d_local'] = l_d.detach().mean()
            # In wgan, real_score should be positive and fake_score should be
            # negative
            loss_dict['real_score_local'] = real_pred.detach().mean()
            loss_dict['fake_score_local'] = fake_pred.detach().mean()
            l_d.backward()
            self.optimizer_d_local.step()
            # ----------------------------------------------------------------------------------------------------------
        if current_iter > self.opt['train'].get('main_gan_start_iter', 0):

            # ----------------------------------------------------------------------------------------------------------
            # global main
            for p in self.net_d_main_global.parameters():
                p.requires_grad = True

            self.optimizer_d_main_global.zero_grad()

            fake_pred = self.net_d_main_global(lq_res['main_dec'].detach())
            real_pred = self.net_d_main_global(self.gt)

            # wgan loss with softplus (logistic loss) for discriminator
            l_d = (
                self.cri_globalgan(real_pred, True, is_disc=True) + self.cri_globalgan(fake_pred, False, is_disc=True))

            loss_dict['l_d_global_main'] = l_d.detach().mean()
            # In wgan, real_score should be positive and fake_score should be
            # negative
            loss_dict['real_score_global_main'] = real_pred.detach().mean()
            loss_dict['fake_score_global_main'] = fake_pred.detach().mean()

            l_d.backward()
            if current_iter % self.net_d_reg_every == 0:
                self.gt.requires_grad = True
                real_pred = self.net_d_main_global(self.gt)
                l_d_r1 = r1_penalty(real_pred, self.gt)
                l_d_r1 = (self.r1_reg_weight / 2 * l_d_r1 * self.net_d_reg_every + 0 * real_pred[0])
                # TODO: why do we need to add 0 * real_pred, otherwise, a runtime
                # error will arise: RuntimeError: Expected to have finished
                # reduction in the prior iteration before starting a new one.
                # This error indicates that your module has parameters that were
                # not used in producing loss.
                loss_dict['l_d_r1_main'] = l_d_r1.detach().mean()
                l_d_r1.backward()
            self.optimizer_d_main_global.step()

            # --------------------------------------------------------------------------------------------------------
            for p in self.net_d_main_local.parameters():
                p.requires_grad = True
            self.optimizer_d_main_local.zero_grad()
            fake_pred = self.net_d_main_local(lq_res['main_dec'].detach())
            real_pred = self.net_d_main_local(self.gt)
            l_d = 0.5 * (
                self.cri_patchgan(real_pred, True, is_disc=True) + self.cri_patchgan(fake_pred, False, is_disc=True))
            loss_dict['l_d_local_main'] = l_d.detach().mean()
            # In wgan, real_score should be positive and fake_score should be
            # negative
            loss_dict['real_score_local_main'] = real_pred.detach().mean()
            loss_dict['fake_score_local_main'] = fake_pred.detach().mean()
            l_d.backward()
            self.optimizer_d_main_local.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            raise NotImplementedError
        else:
            self.net_sr.eval()
            with torch.no_grad():
                self.output = self.net_sr(self.lq, return_keys=('dec'))
                if self.opt['val'].get('test_which') == 'main_branch':
                    self.output = self.output['main_dec']
                elif self.opt['val'].get('test_which') == 'texture_branch':
                    self.output = self.output['dec']
                else:
                    raise NotImplementedError
            self.net_sr.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            range = self.opt['val'].get('range', [-1, 1])

            sr_img = tensor2img([visuals['sr']], min_max=range)
            gt_img = tensor2img([visuals['gt']], min_max=range)

            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], min_max=range)
                del self.gt
            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    metric_data = dict(img=sr_img, img2=gt_img)
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['gt'] = self.gt.detach().cpu()
        out_dict['sr'] = self.output.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_sr, 'net_sr', current_iter)
        self.save_network(self.net_d_local, 'net_d_local', current_iter)
        self.save_network(self.net_d_global, 'net_d_global', current_iter)
        self.save_network(self.net_d_main_local, 'net_d_main_local', current_iter)
        self.save_network(self.net_d_main_global, 'net_d_main_global', current_iter)
        self.save_training_state(epoch, current_iter)
