import os
import pdb
import torch
import torchvision
from addict import Dict as adict

from mtorl.utils.initial import (ENV, init_criterion, init_dataloader,
                                 init_model, init_optimizer, init_resume,
                                 init_scheduler)
from mtorl.utils.log import logger, set_logger
from mtorl.utils.runner_iter import Runner

from parse_args import *
from isutils.interactions import *

# compare models
from mtorl.utils.initial import init_is_model
from mtorl.utils.runner_ritm import Runner as Runner_RITM

_CURR_DIR = os.path.dirname(os.path.realpath(__file__))


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
    cfg.random_range = set_random_range(cfg.is_radius, allowed_changes=2, fixed=False, fixed_range=(-9, 9))

    if cfg.is_continue_training and not cfg.continue_cp_path.endswith("_st.pth"):
        co_train_checkpoint = torch.load(cfg.continue_cp_path)        
        cfg.staEpochs = co_train_checkpoint['meta']['epoch'] + 1    
    
    return cfg
    

def main():
    
    cfg = parse_args()
    cfg = adict(vars(cfg))
    info = load_config(cfg.config_path)
    cfg = update_config(cfg, info)
    
    log_file = os.path.join(cfg.save_dir, 'log.txt')
    set_logger(log_file=log_file)
    logger.info(f"pytorch vision: {torch.__version__}")
    logger.info(f"torchvision vision: {torchvision.__version__}")
    logger.info(f"log_file: {log_file}")
    logger.info('*' * 80)
    logger.info('the args are the below')
    logger.info('*' * 80)
    for key, value in cfg.items():
        logger.info('{:<20}:{}'.format(key, str(value)))
    logger.info('*' * 80 + '\n')

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    if cfg.model_name == "tpenet":
        ENV.checkpoint = init_resume(cfg)
        ENV.model = init_model(cfg, ENV.checkpoint)
        ENV.criterion = init_criterion(cfg)

    else:
        ENV.model, ENV.criterion = init_is_model(cfg)

    ENV.data_loaders = init_dataloader(cfg)
    ENV.optimizer = init_optimizer(cfg)

    if not cfg.inference:
        ENV.scheduler = init_scheduler(cfg)


    if cfg.is_continue_training:
        if not cfg.continue_cp_path.endswith("_st.pth"):
        
            co_train_checkpoint = torch.load(cfg.continue_cp_path)        
            ENV.model.load_state_dict(co_train_checkpoint['state_dict'], strict=False)
            ENV.optimizer.load_state_dict(co_train_checkpoint['optimizer_state_dict'])
            ENV.scheduler.load_state_dict(co_train_checkpoint['lr_scheduler_state_dict'])
            
            # cfg.staEpochs = co_train_checkpoint['epoch']
            # ENV.model = torch.nn.DataParallel(ENV.model, device_ids=list(range(2))).cuda()
            print("Load the checkpoint with lr and retrain the model.")
            
        else:
            ENV.model.load_state_dict(torch.load(cfg.continue_cp_path))
            print("Load the state dict and retrain the model.")


    if cfg.model_name == "tpenet":
        runner = Runner(cfg)
    else:
        runner = Runner_RITM(cfg)
    runner.run()


if __name__ == '__main__':
    main()
