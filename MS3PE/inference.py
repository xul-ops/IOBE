import os
import random
import numpy as np

import torch
import torchvision
from addict import Dict as adict

from mtorl.utils.initial import (ENV, init_criterion, init_dataloader,
                                 init_model, init_optimizer, init_resume,
                                 init_scheduler)
from mtorl.utils.log import logger, set_logger
from mtorl.utils.runner_infer import Runner  # for mulit-round exp in supp, etc.

from parse_args import *
from isutils.interactions import *

_CURR_DIR = os.path.dirname(os.path.realpath(__file__))

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    

def update_config(cfg, info):

    pretrained_name = cfg.backbone_name 
    cfg.bankbone_pretrain = info["PRETRAINED_MODELS"][pretrained_name]
    
    cfg.dataset_dir = info[cfg.dataset_name]["dataset_dir"]
    cfg.train_img_txt = info[cfg.dataset_name]["train_img_txt"]
    cfg.val_dataset_dir = info[cfg.val_dataset_name]["val_dataset_dir"]
    cfg.val_img_txt =  info[cfg.val_dataset_name]["val_img_txt"]
    
    if cfg.dataset_name == "bsds":
        cfg.boundary_lambda = 1.1

    if not os.path.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir)

    cfg.mtorl_root_dir = './'
    cfg.random_range = set_random_range(cfg.is_radius, allowed_changes=2, fixed=False, fixed_range=(-1, 1))

    if cfg.is_continue_training and not cfg.continue_cp_path.endswith("_st.pth"):
        co_train_checkpoint = torch.load(cfg.continue_cp_path)        
        cfg.staEpochs = co_train_checkpoint['meta']['epoch'] + 1    
    
    return cfg


def main():
    # print(name)
    cfg = parse_args()
    cfg = adict(vars(cfg))
    info = load_config(cfg.config_path)
    cfg = update_config(cfg, info)

    log_file = os.path.join(cfg.save_dir, 'log_eval.txt')
    set_logger(log_file=log_file)
    cfg.mtorl_root_dir = './'
    cfg.random_range = set_random_range(cfg.is_radius, allowed_changes=2, fixed=False, fixed_range=(-1, 1)) 

    logger.info(f"pytorch vision: {torch.__version__}")
    logger.info(f"torchvision vision: {torchvision.__version__}")
    logger.info(f"log_file: {log_file}")
    logger.info('*' * 80)
    logger.info('the args are the below')
    logger.info('*' * 80)
    for key, value in cfg.items():
        logger.info('{:<20}:{}'.format(key, str(value)))
    logger.info('*' * 80 + '\n')

    setup_seed(cfg.seed)
        
    ENV.model = init_model(cfg)
    ENV.criterion = init_criterion(cfg)
    ENV.data_loaders = init_dataloader(cfg)
    ENV.optimizer = init_optimizer(cfg)

    if not cfg.inference:
        ENV.scheduler = init_scheduler(cfg)

    runner = Runner(cfg)

    
    if os.path.isfile(cfg.resume_checkpoint):
        runner.evaluate(runner.data_loaders["test"], cfg.resume_checkpoint, concat_prev_fnfp=cfg.concat_prevfnfp, round_num=cfg.multi_round_num) # exsiting_map_path="./newnew/e_pre/")
        
    elif os.path.isdir(cfg.resume_checkpoint):
        elements = os.listdir(cfg.resume_checkpoint)

        for element in elements:
            element_str = element.split("_")

            # if not element.endswith("_cp.pth"):
            if element_str[-1].endswith("st.pth"): # and int(element_str[0]) >= cfg.startIterEpoch:
                c_path = cfg.resume_checkpoint + element
                print("Current checkpoint is located in ", c_path) 
                runner.evaluate(runner.data_loaders["test"], c_path, concat_prev_fnfp=cfg.concat_prevfnfp, round_num=cfg.multi_round_num)


if __name__ == '__main__':
    main()
