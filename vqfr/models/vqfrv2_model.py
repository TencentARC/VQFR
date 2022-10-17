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
class VQFRv2Model(BaseModel):
    """VQGAN_BASE_Model"""

    def __init__(self, opt):
        super(VQFRv2Model, self).__init__(opt)

        # define network
        if 'network_g' in self.opt:
            self.net_g = build_network(opt['network_g'])
            self.net_g = self.model_to_device(self.net_g)
            self.print_network(self.net_g)

        self.net_sr = build_network(opt['network_sr'])
        self.net_sr = self.model_to_device(self.net_sr)
        self.print_network(self.net_sr)

        # define network net_d
        if 'network_d' in self.opt:
            self.net_d = build_network(self.opt['network_d'])
            if self.opt.get('syncbn') is True and self.opt['num_gpu'] > 1:
                self.net_d = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                    self.net_d)  # to avoid broadcast buffer error
            self.net_d = self.model_to_device(self.net_d)
            self.print_network(self.net_d)

        # define network net_d
        if 'network_d_local' in self.opt:
            self.net_d_local = build_network(self.opt['network_d_local'])
            if self.opt.get('syncbn') is True and self.opt['num_gpu'] > 1:
                self.net_d_local = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                    self.net_d_local)  # to avoid broadcast buffer error
            self.net_d_local = self.model_to_device(self.net_d_local)
            self.print_network(self.net_d_local)

        # load pretrained models
        logger = get_root_logger()

        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            logger.info('Loading net_d:')
            param_key = 'params'
            self.load_network(self.net_d, load_path, True, param_key)

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
        if train_opt.get('quant_feature_opt'):
            self.cri_quant_feature = build_loss(train_opt['quant_feature_opt']).to(self.device)
        else:
            self.cri_quant_feature = None

        # define losses
        if train_opt.get('quant_index_opt'):
            self.cri_quant_index = build_loss(train_opt['quant_index_opt']).to(self.device)
        else:
            self.cri_quant_index = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)
        else:
            self.cri_gan = None

        if train_opt.get('patch_gan_opt'):
            self.cri_patchgan = build_loss(train_opt['patch_gan_opt']).to(self.device)
        else:
            self.cri_patchgan = None

        self.r1_reg_weight = train_opt.get('r1_reg_weight', None)  # for discriminator
        self.net_d_reg_every = train_opt.get('net_d_reg_every', None)

        self.generator_d_global_weight = train_opt['generator_d_global_weight']
        self.generator_d_local_weight = train_opt['generator_d_local_weight']

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        encoder_params = []
        main_decoder_params = []
        for k, v in self.net_sr.named_parameters():
            if 'encoder' in k or 'feat2index' in k:
                encoder_params.append(v)
            elif 'main_branch' in k or 'inpfeat_extraction' in k:
                main_decoder_params.append(v)
            else:
                # not optimize codebook and texture decoder
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_sr_enc'].pop('type')
        self.optimizer_sr_enc = self.get_optimizer(optim_type, encoder_params, **train_opt['optim_sr_enc'])
        self.optimizers.append(self.optimizer_sr_enc)

        optim_type = train_opt['optim_sr_maindec'].pop('type')
        self.optimizer_sr_maindec = self.get_optimizer(optim_type, main_decoder_params, **train_opt['optim_sr_maindec'])
        self.optimizers.append(self.optimizer_sr_maindec)

        # optimizer d
        if 'optim_d' in train_opt:
            optim_type = train_opt['optim_d'].pop('type')
            self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
            self.optimizers.append(self.optimizer_d)
            self.optimizer_d_local = self.get_optimizer(optim_type, self.net_d_local.parameters(),
                                                        **train_opt['optim_d'])
            self.optimizers.append(self.optimizer_d_local)

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

        self.optimizer_sr_enc.zero_grad()
        self.optimizer_sr_maindec.zero_grad()

        # get_lq result
        lq_res = self.net_sr(self.lq)

        l_total_g = 0.0

        # pixel reconstruction loss
        if self.cri_pix:
            l_rec = self.cri_pix(lq_res['main_dec'], self.gt)
            loss_dict['l_rec'] = l_rec.detach().mean()
            l_total_g += l_rec

        if self.cri_quant_feature or self.cri_quant_index:
            # get hr result
            with torch.no_grad():
                hq_res, _ = self.net_g(self.gt, return_keys=('feat_dict'))

        if self.cri_quant_feature:
            l_quant_feature = self.cri_quant_feature(lq_res['enc_feat'], hq_res['quant_feat'])
            loss_dict['l_quant_feature'] = l_quant_feature.detach().mean()
            l_total_g += l_quant_feature

        if self.cri_quant_index:
            l_quant_index = self.cri_quant_index(lq_res['quant_logit'], hq_res['quant_index'])
            loss_dict['l_quant_index'] = l_quant_index.detach().mean()
            l_total_g += l_quant_index

        # perceptual loss
        if self.cri_perceptual:
            l_g_percep, l_g_style = self.cri_perceptual(lq_res['main_dec'], self.gt)

            if l_g_percep is not None:
                l_total_g += l_g_percep
                loss_dict['l_g_percep'] = l_g_percep.detach().mean()
            if l_g_style is not None:
                l_total_g += l_g_style
                loss_dict['l_g_style'] = l_g_style.detach().mean()

        if self.cri_gan and current_iter > self.opt['train'].get('gan_start_iter', 0):
            # style gan
            for p in self.net_d.parameters():
                p.requires_grad = False
            fake_pred = self.net_d(lq_res['main_dec'])
            l_g_gan = self.cri_gan(fake_pred, True, is_disc=False)
            loss_dict['l_g_gan'] = l_g_gan.detach().mean()
            l_total_g += l_g_gan

            # patch gan
            for p in self.net_d_local.parameters():
                p.requires_grad = False
            fake_pred = self.net_d_local(lq_res['main_dec'])
            l_g_gan = self.cri_patchgan(fake_pred, True, is_disc=False)

            d_weight = self.generator_d_local_weight * self.calculate_adaptive_weight(
                l_g_percep,
                l_g_gan,
                last_layer=self.net_sr.module.get_last_layer()
                if dist.is_initialized() else self.net_sr.get_last_layer())
            l_g_gan = d_weight * l_g_gan
            loss_dict['l_g_gan_local'] = l_g_gan.detach().mean()
            l_total_g += l_g_gan

        l_total_g.backward()
        self.optimizer_sr_enc.step()
        self.optimizer_sr_maindec.step()

        if self.cri_gan and current_iter > self.opt['train'].get('gan_start_iter', 0):
            for p in self.net_d.parameters():
                p.requires_grad = True
            fake_pred = self.net_d(lq_res['main_dec'].detach())
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

            # ----------------------------------------------------------------------------------------------------------
            for p in self.net_d_local.parameters():
                p.requires_grad = True
            self.optimizer_d_local.zero_grad()
            fake_pred = self.net_d_local(lq_res['main_dec'].detach())
            real_pred = self.net_d_local(self.gt)
            l_d = self.cri_patchgan(real_pred, True, is_disc=True) + self.cri_patchgan(fake_pred, False, is_disc=True)
            loss_dict['l_d_local'] = l_d.detach().mean()
            # In wgan, real_score should be positive and fake_score should be
            # negative
            loss_dict['real_score_local'] = real_pred.detach().mean()
            loss_dict['fake_score_local'] = fake_pred.detach().mean()
            l_d.backward()
            self.optimizer_d_local.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_sr.eval()
        with torch.no_grad():
            self.output = self.net_sr(self.lq, fidelity_ratio=self.opt['val'].get('fidelity_ratio', 1.0))
            if self.opt['val'].get('test_which') == 'main_branch':
                self.output = self.output['main_dec']
            else:
                self.output = self.output['texture_dec']
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
        if hasattr(self, 'net_d'):
            self.save_network(self.net_d, 'net_d', current_iter)
        if hasattr(self, 'net_d_local'):
            self.save_network(self.net_d_local, 'net_d_local', current_iter)
        self.save_training_state(epoch, current_iter)
