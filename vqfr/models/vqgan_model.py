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
class VQGANModel(BaseModel):

    def __init__(self, opt):
        super(VQGANModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        if self.opt.get('syncbn') is True and self.opt['num_gpu'] > 1:
            self.net_g = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net_g)  # to avoid broadcast buffer error
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        self.net_g.train()

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define network net_d
        if 'network_d' in self.opt:
            self.net_d = build_network(self.opt['network_d'])
            if self.opt.get('syncbn') is True and self.opt['num_gpu'] > 1:
                self.net_d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                    self.net_d)  # to avoid broadcast buffer error
            self.net_d = self.model_to_device(self.net_d)
            self.print_network(self.net_d)
            self.net_d.train()

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True), param_key)

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)
        else:
            self.cri_gan = None

        self.r1_reg_weight = train_opt['r1_reg_weight']  # for discriminator
        self.net_d_reg_every = train_opt['net_d_reg_every']

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']

        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, self.net_g.parameters(), **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

        # optimizer d
        if 'network_d' in self.opt:
            optim_type = train_opt['optim_d'].pop('type')
            self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
            self.optimizers.append(self.optimizer_d)

    def optimize_parameters(self, current_iter):
        # optimize net_g
        if self.cri_gan and current_iter > self.opt['train'].get('gan_start_iter', 0):
            for p in self.net_d.parameters():
                p.requires_grad = False

        loss_dict = OrderedDict()

        self.optimizer_g.zero_grad()
        self.output, codebook_loss = self.net_g(self.gt, current_iter, return_keys=('dec'))

        l_total_g = 0.0
        # pixel reconstruction loss
        if self.cri_pix:
            l_rec = self.cri_pix(self.output['dec'], self.gt)
            loss_dict['l_rec'] = l_rec
            l_total_g += l_rec

        # perceptual loss
        if self.cri_perceptual:
            l_g_percep, l_g_style = self.cri_perceptual(self.output['dec'], self.gt)
            if l_g_percep is not None:
                l_total_g += l_g_percep
                loss_dict['l_g_percep'] = l_g_percep
            if l_g_style is not None:
                l_total_g += l_g_style
                loss_dict['l_g_style'] = l_g_style

        # codebook loss
        l_codebook = codebook_loss * self.opt['train']['codebook_loss_weight']
        loss_dict['l_codebook'] = l_codebook
        l_total_g += l_codebook

        # gan loss
        # wgan loss with softplus (non-saturating loss) for generator
        if self.cri_gan and current_iter > self.opt['train'].get('gan_start_iter', 0):
            fake_pred = self.net_d(self.output['dec'])
            l_g_gan = self.cri_gan(fake_pred, True, is_disc=False)
            loss_dict['l_g_gan'] = l_g_gan
            l_total_g += l_g_gan

        loss_dict['l_total_g'] = l_total_g
        l_total_g.backward()
        self.optimizer_g.step()

        if self.cri_gan and current_iter > self.opt['train'].get('gan_start_iter', 0):
            # optimize net_d
            for p in self.net_d.parameters():
                p.requires_grad = True
            self.optimizer_d.zero_grad()

            fake_pred = self.net_d(self.output['dec'].detach())
            real_pred = self.net_d(self.gt)
            # wgan loss with softplus (logistic loss) for discriminator
            l_d = self.cri_gan(real_pred, True, is_disc=True) + self.cri_gan(fake_pred, False, is_disc=True)
            loss_dict['l_d'] = l_d
            # In wgan, real_score should be positive and fake_score should be
            # negative
            loss_dict['real_score'] = real_pred.detach().mean()
            loss_dict['fake_score'] = fake_pred.detach().mean()
            l_d.backward()
            if current_iter % self.net_d_reg_every == 0:
                self.gt.requires_grad = True
                real_pred = self.net_d(self.gt)
                l_d_r1 = r1_penalty(real_pred, self.gt)
                l_d_r1 = (self.r1_reg_weight / 2 * l_d_r1 * self.net_d_reg_every + 0 * real_pred[0])
                # TODO: why do we need to add 0 * real_pred, otherwise, a runtime
                # error will arise: RuntimeError: Expected to have finished
                # reduction in the prior iteration before starting a new one.
                # This error indicates that your module has parameters that were
                # not used in producing loss.
                loss_dict['l_d_r1'] = l_d_r1.detach().mean()
                l_d_r1.backward()
            self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)

    def feed_data(self, data):
        self.gt = data['gt'].to(self.device)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output, _ = self.net_g_ema(self.gt, return_keys=('dec'))
                self.output = self.output['dec']

        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output, _ = self.net_g(self.gt, return_keys=('dec'))
                self.output = self.output['dec']

            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        if dist.is_initialized():
            self.net_g.module.quantizer.reset_usage()
        else:
            self.net_g.quantizer.reset_usage()

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['gt_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            range = self.opt['val'].get('range', [-1, 1])

            sr_img = tensor2img([visuals['result']], min_max=tuple(range))
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], min_max=tuple(range))
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
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
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if dist.is_initialized():
            codebook_usage = self.net_g.module.quantizer.get_usage()
        else:
            codebook_usage = self.net_g.quantizer.get_usage()

        logger = get_root_logger()
        logger.info('codebook_usage:')
        logger.info(codebook_usage)

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict
