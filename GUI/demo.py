import argparse
import tkinter as tk

import torch

# from isegm.utils import exp
# from isegm.inference import utils
from interactive_demo.app import InteractiveDemoApp


def main():
    args, cfg = parse_args()

    torch.backends.cudnn.deterministic = True
    # checkpoint_path = utils.find_checkpoint(cfg.INTERACTIVE_MODELS_PATH, args.checkpoint)
    # model = utils.load_is_model(checkpoint_path, args.device, cpu_dist_maps=True)

    root = tk.Tk()
    root.minsize(960, 550)  # 480
    app = InteractiveDemoApp(root, args, args.checkpoint)
    root.deiconify()  # 使窗口重新显示在电脑屏幕上
    app.mainloop()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default="./savedModels/swins_entity_e12.pth", 
                        help='The path to the checkpoint.')
    parser.add_argument('--dataset_name', type=str, default="diode")
    parser.add_argument('--gpu', type=int, default=0,
                        help='Id of GPU to use.')
    parser.add_argument('--cpu', action='store_true', default=True,
                        help='Use only CPU for inference.')
    parser.add_argument('--limit-longest-size', type=int, default=800,
                        help='If the largest side of an image exceeds this value, '
                             'it is resized so that its largest side is equal to this value.')
    parser.add_argument('--cfg', type=str, default="config.yml",
                        help='The path to the config file.')
    parser.add_argument('--output_dir', type=str, default="./demo_outputs/",
                        help='output_dir.')
    parser.add_argument('--existing_pre_dir', type=str, default="./pre_res/psb_l10e11_diode/epoch_11_test_result/",
                        help='output_dir.')                    
    args = parser.parse_args()
    if args.cpu:
        args.device = torch.device('cpu')
    else:
        args.device = torch.device(f'cuda:{args.gpu}')
    # cfg = exp.load_config_file(args.cfg, return_edict=True)

    return args, 0 # cfg


if __name__ == '__main__':
    main()
