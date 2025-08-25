import yaml
import argparse
from easydict import EasyDict as edict


def load_config(config_path, return_edict=False):

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    return edict(cfg) if return_edict else cfg


def parse_args():
    parser = argparse.ArgumentParser(description='Train for different args')
    
    parser.add_argument('--config_path', type=str, default="./config.yml", help='datasets info and backbone path')

    parser.add_argument('--epoch', type=int, default=15, help='max epoch to train network, default is 15')
    parser.add_argument('--inference', action='store_true', help='amp')
    parser.add_argument('--seed', type=float, default=42, help='seed for torch.cuda.manual_seed()')
    parser.add_argument('--amp', action='store_true', help='amp')
    parser.add_argument('--cuda', action='store_true', help='whether use gpu to train network')
    parser.add_argument('--resume', type=str, default=None, help='whether resume from some, default is None')
    parser.add_argument('--save_dir', type=str, default='result', help='')
    parser.add_argument('--staEpochs', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--is_continue_training', action='store_true', help='stopped model continue training')
    parser.add_argument('--continue_cp_path', default="./", type=str, help='continue training checkpoin path')
    parser.add_argument('--resume_checkpoint', default="./", type=str, help='')
    
    # params interactive
    parser.add_argument('--is_interactive', action='store_false', help='intearctive mode training')
    parser.add_argument('--deep_fusion', action='store_false', help="InP and FEM")
    parser.add_argument('--startIterEpoch', type=int, default=4, help="second stage")
    parser.add_argument('--matlab_nms', action='store_true', help='default false')
    parser.add_argument('--num_iter', type=int, default=3, help="max initeraction numbers")
    parser.add_argument('--previous_mask', action='store_true', help="if use pervious output")
    # parser.add_argument('--only_positive', action='store_true')
    parser.add_argument('--with_edge_pro', action='store_true') # unused
    parser.add_argument('--is_encoding', type=str, default="scribbles") # support many encoding ways
    parser.add_argument('--is_interaction', type=str, default="scribbles") # support many interaction modes
    parser.add_argument('--p_threshold', type=float, default=0.7, help="threshold to process pervious output")
    parser.add_argument('--match_radius', type=int, default=4)
    parser.add_argument('--candidate_len', type=int, default=30, help="candidate length for performing interactions")
    parser.add_argument('--is_radius', type=int, default=12, help="interaction radii")
    parser.add_argument('--occ_loss_mask', action='store_true') # unused

    # data
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--dataset_name', default="synocc", type=str, choices=['synocc', 'synocc_rgba',  'piod', 'bsds', 'nyuocpp', 'diode', 'entityseg', "cmu"], help='dataset name')
    parser.add_argument('--dataset_dir', default="", type=str, help='path to dataset')
    parser.add_argument('--val_dataset_name', default="diode", type=str)
    parser.add_argument('--val_dataset_dir', default="", type=str, help='')
    parser.add_argument('--random_crop_size', type=int, default=320)  
    parser.add_argument('--train_transform', action='store_true', help='just use crop')  # unused
    parser.add_argument('--train_transform_A', type=bool, default=True, help='use albumentations transform')
    parser.add_argument('--additional_train_trans', action='store_true', help='use additional albumentations transform for train data')
    parser.add_argument('--train_img_txt', type=str, default='')
    parser.add_argument('--val_img_txt', type=str, default='')

    # model
    parser.add_argument('--model_name', type=str, default='tpenet')
    parser.add_argument('--bankbone_pretrain', type=str, default='./data/resnet50s-a75c83cf.pth',
                        help='init net from pretrained model default is None')
    parser.add_argument('--task_type', type=str, default='occ_edge', help='edge or ori')
    parser.add_argument('--backbone_name', type=str, default='swinformer_large')  

    # train
    # loss
    parser.add_argument('--boundary_weights', type=str, default='0.5,0.5,0.5,0.5,0.5,1.1', help='')
    parser.add_argument('--orientation_weight', type=float, default=0.4, help='')
    parser.add_argument('--boundary_lambda', type=float, default=1.7, help='')

    # optim
    parser.add_argument('--optim', type=str, default='adamw', choices=['adamw', 'radam', 'sgd'], help='self.optimizer')
    parser.add_argument('--base_lr', type=float, default=0.00003, help='the base learning rate of model')
    # {'backbone':0.9}
    parser.add_argument('--module_name_scale', type=str, default="{'backbone': 0.9}",
                        help='module_name= backbone, ori_convolution, ori_decoder, \
                            boundary_convolution, boundary_decoder, osm, encoder_sides, fuse')
    parser.add_argument('--weight_decay', type=float, default=0.002, help='the weight_decay of net')
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum')

    parser.add_argument('--scheduler_name', type=str, default='MultiStepLR',
                        help='learning rate scheduler (default: MultiStepLR)')

    # parser.add_argument('--scheduler_param', type=str, default="{'milestones':[6, 9, 11]}", help='')
    parser.add_argument('--scheduler_param', type=str, default="{'milestones':[10,14,16]}", help='')
    parser.add_argument('--scheduler_mode', type=str, default='epoch', help='epoch or iter')
    parser.add_argument('--warmup_epochs', type=int, default=0, help='warmup_epochs')

    # interactive abalations
    parser.add_argument('--aba_loss_type', type=str, default='original', help="CCE, original, FL, AL")  
    
    # for multi round test, in supp
    parser.add_argument('--concat_prevfnfp', action='store_true', help="")  # unused
    parser.add_argument('--multi_round_num', type=int, default=3)  # unused
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print(dir(args))
