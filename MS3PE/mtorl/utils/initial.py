import os
import pdb

import torch
from addict import Dict as adict
from mtorl.datasets.dataloader_occ import get_occ_dataloader
from mtorl.models import TPENet # , iTPENet
from mtorl.utils.serialization import load_model
from mtorl.models.init_ismodels import *

from mtorl.models.losses.occlusion_loss import OcclousionLoss, OcclousionLossMask, OcclousionFLLoss, FLLoss

from mtorl.utils.log import logger
from mtorl.utils.lr_scheduler import WarmUpLRScheduler

__all__ = ['ENV', 'init_resume', 'init_model']

ENV = adict()
ENV.device = torch.device('cpu')


def _load_checkpoint(src_path: str):
    """
    Load checkpoint from local of hdfs
    """

    if not isinstance(src_path, str):
        return None

    if os.path.exists(src_path):
        return torch.load(src_path)
    else:
        logger.warning(f'file {src_path} is not found.')


def init_resume(cfg):
    checkpoint = None
    if cfg.resume is not None:
        checkpoint = _load_checkpoint(cfg.resume)
    if checkpoint is not None:
        logger.info(f'=> Model resume: loaded from {cfg.resume}\n')
    return checkpoint


def init_model(cfg, resume_checkpoint=None):
    logger.info(f"=> Model: {cfg.model_name}")
    if cfg.model_name == 'tpenet':
        # IS mode:
        in_channels = 5 
                            
        if cfg.previous_mask:
            in_channels += 1

        model = TPENet(
                backbone_name=cfg.backbone_name,
                use_pixel_shuffle=True,
                in_channels=in_channels,
                deep_fusion=cfg.deep_fusion,
                backbone_path=cfg.bankbone_pretrain
                )   

        if cfg.backbone_name.startswith('resnet') and cfg.bankbone_pretrain is not None:
            model.load_backbone_pretrained(cfg.bankbone_pretrain)
                    
    # Resume model if necessary
    if resume_checkpoint is not None:
        state_dict = resume_checkpoint.get('state_dict', resume_checkpoint)
        model.load_state_dict(state_dict)
        logger.info(f'  Model resume')

    if cfg.cuda:
        model = model.cuda()
    return model


def init_is_model(cfg):

    if cfg.model_name.startswith("ritm_"):
        model, criterion = init_ritm(cfg)

    elif cfg.model_name == "cdnet":
        # without previous
        model, criterion = init_cdnet(cfg)

    elif cfg.model_name.startswith("focal_"):
        print("FocalClick segformer without refinet")
        model, criterion = init_focalclick(cfg)  

    elif cfg.model_name.startswith("plainvit_"):
        print("Simple click")
        model, criterion = init_simpleclick(cfg)         

    elif cfg.model_name.startswith("adaptiveclick_"):
        print("Adaptive click")
        model, criterion = init_adaptiveclick(cfg)   

    elif cfg.model_name == "acc99net":
        print("Modified Acc99 Net")
        model, criterion = init_acc99net(cfg)

    elif cfg.model_name == "fcanet":
        print("Modified FCANet")
        model, criterion = init_fcanet(cfg)

    elif cfg.model_name == "deepthin":
        print("Modified Thin Object Selection Network (TOS-Net) ")
        model, criterion = init_deepthin(cfg)

    else:
        raise ValueError 

    return model, criterion


def init_dataloader(cfg):
    dataloader = get_occ_dataloader(cfg)
    return dataloader


def init_criterion(cfg):
    logger.info('=> Criterion')
    boundary_weights = cfg.boundary_weights.split(',')
    boundary_weights = list(map(float, boundary_weights))
    logger.info(f'   boundary_weights: {boundary_weights}')
    logger.info(f'   boundary_lambda: {cfg.boundary_lambda}')
    logger.info(f'   orientation_weight: {cfg.orientation_weight}')
   
    if cfg.occ_loss_mask:   # unused
        logger.info('   Add additional loss for fnfp map area! }')
        print("Add additional loss for fnfp map area!")
        print()
        criterion = OcclousionLossMask(
        boundary_weights=boundary_weights,
        boundary_lambda=cfg.boundary_lambda,
        orientation_weight=cfg.orientation_weight)
        
    else:
        # OcclousionFLLoss
        # OcclousionLoss
        # FLLoss
        criterion = OcclousionLoss(
        boundary_weights=boundary_weights,
        boundary_lambda=cfg.boundary_lambda,
        orientation_weight=cfg.orientation_weight)
        
    if cfg.cuda:
        criterion = criterion.cuda()
    return criterion


def init_optimizer(cfg):
    logger.info('=> optimizer')
    logger.info(f'  optim_name: {cfg.optim}')
    logger.info(f'  lr: {cfg.base_lr}')
    logger.info(f'  weight_decay: {cfg.weight_decay}')
    module_name_scale = eval(cfg.module_name_scale)
    logger.info(f'  module_name_scale: {cfg.module_name_scale}')

    params_group = []
    for name, m in ENV.model.named_children():
        scale = module_name_scale.get(name, None)
        if scale is not None:
            params_group.append({'params': m.parameters(), 'lr': cfg.base_lr * scale})
            # print(name, cfg.base_lr*scale)
        else:
            params_group.insert(0, {'params': m.parameters(), 'lr': cfg.base_lr})

    if cfg.optim == 'adamw':
        optim = torch.optim.AdamW(params_group, lr=cfg.base_lr, weight_decay=cfg.weight_decay)
    elif cfg.optim == 'sgd':
        optim = torch.optim.SGD(params_group, momentum=cfg.momentum, lr=cfg.base_lr,
                                weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f'cfg.optim={cfg.optim} is invalid')
    return optim


def init_scheduler(cfg):
    logger.info(' => scheduler')
    logger.info(f'  name: {cfg.scheduler_name}')
    logger.info(f'  scheduler_mode: {cfg.scheduler_mode}')
    logger.info(f'  warmup_epochs: {cfg.warmup_epochs}')

    scheduler_param = eval(cfg.scheduler_param)
    logger.info(f'  scheduler_param: {scheduler_param}')

    T_scale = 1
    if cfg.scheduler_mode == 'epoch':
        T_scale = 1
    elif cfg.scheduler_mode == 'iter':
        T_scale = len(ENV.data_loaders['train'])
    else:
        raise ValueError(f'cfg.scheduler_mode={cfg.scheduler_mode} is not supported')

    T_warmup = cfg.warmup_epochs * T_scale
    if cfg.scheduler_name == 'CosineAnnealingLR':
        if 'T_max' not in scheduler_param.keys():
            scheduler_param['T_max'] = cfg.epoch * T_scale

    scheduler = WarmUpLRScheduler(ENV.optimizer, T_warmup,
                                  after_scheduler_name=cfg.scheduler_name, **scheduler_param)
    return scheduler


def load_is_model(checkpoint, device, **kwargs):
    if isinstance(checkpoint, (str, Path)):
        state_dict = torch.load(checkpoint, map_location='cpu')
    else:
        state_dict = checkpoint

    if isinstance(state_dict, list):
        model = load_single_is_model(state_dict[0], device, **kwargs)
        models = [load_single_is_model(x, device, **kwargs) for x in state_dict]

        return model, models
    else:
        return load_single_is_model(state_dict, device, **kwargs)


def load_single_is_model(state_dict, device, **kwargs):
    model = load_model(state_dict['config'], **kwargs)
    model.load_state_dict(state_dict['state_dict'], strict=False)

    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    model.eval()

    return model
