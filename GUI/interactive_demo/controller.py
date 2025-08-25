import os
import cv2
import time
import torch
import numpy as np
from PIL import Image
import imageio
from collections import Counter
from tkinter import messagebox

# from isegm.inference import clicker
# from isegm.inference.predictors import get_predictor
# from isegm.utils.vis import draw_with_blend_and_clicks
from interactive_demo.isegmtools import Click, Clicker
from interactive_demo.isegmtools import draw_with_blend_and_clicks

from torch.utils.data import DataLoader
from torchvision import transforms
# from isutils.geprmap import get_matlab_eng, mb_nmst
from mtorl.models import TPENet # OPNet, OPNetv1, OPNetv2
from isutils.edgeEvalPy.nms_process import nms_process_one_image


def reverse_single_pred(pre_ep):
    img = np.array(pre_ep)
    img = img / 255.0 - 1    
    return img


class InteractiveController:
    def __init__(self, net_path, device, args, predictor_params, update_image_callback, load_mask_k, prob_thresh=0.7):
        self.net_path = net_path
        self.args = args
        self.prob_thresh = prob_thresh
        self.clicker = Clicker()
        self.states = []
        self.probs_history = []
        self.object_count = 0
        self._result_mask = None
        self._init_mask = None

        self.image = None
        self.image_model = None
        self.image_name = None
        self.predictor = None
        self.device = device
        self.update_image_callback = update_image_callback
        self.predictor_params = predictor_params
        # self.reset_predictor()
        self.final_pred = None
        self.load_mask_k = load_mask_k

        self.dataset_name = args.dataset_name
        if self.dataset_name == "occ":
            normalize = transforms.Normalize(mean=[0.780, 0.771, 0.771], std=[0.203, 0.206, 0.217])
        else:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.trans = transforms.Compose([transforms.ToTensor(), normalize])

        self.existing_pre_dir = args.existing_pre_dir
        self.output_dir = args.output_dir 
        self.pos_save_path = args.output_dir + "pos/"
        self.neg_save_path = args.output_dir + "neg/"
        self.pre_save_path = args.output_dir + "pre/"
        self.interactive_res_path = args.output_dir + "f_res/"

        print("-"*80 + "\n\t\t Interactive Edge&Boundary Annotation start!\n" + "-"*80)
        print()


    def set_image(self, image, image_name, vis=False):
        self.image = image
        if not vis:
            print("Training image reset!")
            self.image_name = image_name
            self.image_model = image

        # cv2.imwrite("./test.png", self.image_model) # BGR
        # print(self.image_name)
        self._result_mask = np.zeros(image.shape[:2], dtype=np.uint16)
        self.object_count = 0
        self.reset_last_object(update_image=False)
        self.update_image_callback(reset_canvas=True)


    def set_mask(self, mask):
        if self.image.shape[:2] != mask.shape[:2]:
            messagebox.showwarning("Warning", "A segmentation mask must have the same sizes as the current image!")
            return

        if len(self.probs_history) > 0:
            self.reset_last_object()

        self._init_mask = mask.astype(np.float32)
        # self.probs_history.append((np.zeros_like(self._init_mask), self._init_mask))
        # self._init_mask = torch.tensor(self._init_mask, device=self.device).unsqueeze(0).unsqueeze(0)
        # self.clicker.click_indx_offset = 1
        print("Recivied mask form app user side!")
    
    def input_img_process(self):
        start = time.time()
        img = np.array(self.image_model).astype(np.float32) / 255.0
    
        zero_channels = False
        count_3 = 0

        if self._init_mask is not None:
            # print(self._init_mask.shape)
            # print(Counter(self._init_mask.reshape(-1)))
            pep_pred = self._init_mask
        else:
            # check the pos/neg/pre dir, if already existing the images
            print("Didn't find interaction mask from the app user side, try to check the local folder!")
            img_name = self.image_name.split('/')[-1][:-4]
            pos_list = os.listdir(self.pos_save_path)
            neg_list = os.listdir(self.neg_save_path)
            pre_list = os.listdir(self.pre_save_path)
            if img_name + "_pynms.png" in pre_list:  # "_pre.png"
                pre = np.array(cv2.imread(self.pre_save_path+img_name + "_pynms.png", cv2.IMREAD_UNCHANGED)).reshape(img.shape[0], img.shape[1], 1)
                print("Found local previous edge probility map!")
            else:
                pre = np.zeros((img.shape[0], img.shape[1], 1))
                count_3 += 1

            if img_name + "_pos.png" in pos_list:
                pos = np.array(cv2.imread(self.pos_save_path+img_name + "_pos.png", cv2.IMREAD_UNCHANGED)).reshape(img.shape[0], img.shape[1], 1)
                print("Found local pos mask map!")
            else:
                pos = np.zeros((img.shape[0], img.shape[1], 1))
                count_3 += 1
            if img_name + "_neg.png" in neg_list:
                neg = np.array(cv2.imread(self.neg_save_path+img_name + "_neg.png", cv2.IMREAD_UNCHANGED)).reshape(img.shape[0], img.shape[1], 1)
                print("Found local neg mask map!")
            else:
                neg = np.zeros((img.shape[0], img.shape[1], 1))
                count_3 += 1

            pep_pred = np.concatenate((pos, neg, pre), axis=-1).astype(np.float32)
            # print(pos.shape, neg.shape, pre.shape, pep_pred.shape)
            # print(Counter(pos.reshape(-1)))
            # print(Counter(neg.reshape(-1)))
            # print(Counter(pre.reshape(-1)))
            if count_3 == 3:
                zero_channels = True
                print("Didn't find any local interaction mask! Using three zero channels.")
        
        # else:
        #     pep_pred = np.zeros((np.array(img).shape[0], np.array(img).shape[1], 3)).astype(np.float32)
        #     zero_channels = True

        add_channels = np.transpose(pep_pred / 255, axes=(2, 0, 1))
        add_channels = torch.from_numpy(add_channels).float()                        
        img_trans = self.trans(img)
        img_input = torch.cat((img_trans, add_channels), axis=0)
        print("Input image has been processed, time cost {0:.2f}s".format(time.time()-start))
        return img_input, zero_channels
    
    def output_img_process(self, out_b, zero_channels=True):
        # nms process now is in app.py

        out_b             = out_b[0, 0, :, :]
        model_pred = (out_b * 255).cpu().type(torch.uint8).numpy()
        self.final_pred = model_pred

        # occ_edge_prob     = (1 - out_b) * 255
        occ_edge_prob     = out_b * 255
        occ_edge_prob_vis = occ_edge_prob.cpu().type(torch.uint8)        

        if zero_channels:
            # TO DO: delete zero_channels param and use the original probs_history
            save_path = self.pre_save_path + self.image_name.split('/')[-1][:-4] + "_pre.png"
            # cv2.imwrite(save_path, occ_edge_prob_vis)
            imageio.imwrite(save_path, occ_edge_prob_vis)
 
    def add_click(self, x, y, is_positive):  
        # self.states.append({
        #     'clicker': self.clicker.get_state(),
        #     'predictor': self.predictor.get_states()
        # })

        click = Click(is_positive=is_positive, coords=(y, x))  
        self.clicker.add_click(click)
        # print("【add_click】self.clicker: ", self.clicker)

        # pred = self.predictor.get_prediction(self.clicker, prev_mask=self._init_mask)
        # if self._init_mask is not None and len(self.clicker) == 1:
        #     pred = self.predictor.get_prediction(self.clicker, prev_mask=self._init_mask)
        #
        # torch.cuda.empty_cache()
        #
        # if self.probs_history:
        #     self.probs_history.append((self.probs_history[-1][0], pred))
        # else:
        #     self.probs_history.append((np.zeros_like(pred), pred))

        self.update_image_callback()

    def undo_click(self):
        if not self.states:
            return

        prev_state = self.states.pop()
        self.clicker.set_state(prev_state['clicker'])
        self.predictor.set_states(prev_state['predictor'])
        self.probs_history.pop()
        if not self.probs_history:
            self.reset_init_mask()
        self.update_image_callback()

    # def partially_finish_object(self):
    #     object_prob = self.current_object_prob
    #     if object_prob is None:
    #         return

    #     self.probs_history.append((object_prob, np.zeros_like(object_prob)))
    #     self.states.append(self.states[-1])

    #     self.clicker.reset_clicks()
    #     self.reset_predictor()
    #     self.reset_init_mask()
    #     self.update_image_callback()

    # do the prdictions here
    def finish_object(self):
        img, zero_channels = self.input_img_process()
        if self.predictor is None:
            self.reset_predictor()
        start = time.time()
        with torch.no_grad():
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            output_b, _ = self.predictor(img.unsqueeze(axis=0))                
            out_b = self.predictor.getBoundary(output_b)
            print("Predicting single image, time cost {0:.2f}s".format(time.time()-start))
            self.output_img_process(out_b, zero_channels)
            torch.cuda.empty_cache()
        
        
    # def point_disk(self, canvas_):
    #     print("point disk", canvas_.user_click)

    def reset_last_object(self, update_image=True):
        self.states = []
        self.probs_history = []
        self.clicker.reset_clicks()
        self.reset_predictor()
        self.reset_init_mask()
        if update_image:
            self.update_image_callback()

    def reset_predictor(self, predictor_params=None):
        if self.predictor is None:
            start = time.time()
            model = TPENet( backbone_name="swinformer_small",
                            use_pixel_shuffle=True,
                            in_channels=6,
                            deep_fusion=True,
                            backbone_path=None).to(self.device)
 

            if os.path.isfile(self.net_path):
                print(f"Restoring weights from: {self.net_path}")
                model.load_state_dict(torch.load(self.net_path, map_location=self.device))
            else:
                print("Didn't find existing checkpoints")

            model.eval()
            self.predictor = model
            print("Loading model, time cost {0:.2f}s".format(time.time()-start))
            # if predictor_params is not None:
            #     self.predictor_params = predictor_params
            # self.predictor = get_predictor(self.net_path, device=self.device, **self.predictor_params)
            # if self.image is not None:
            #     self.predictor.set_input_image(self.image)

    def reset_init_mask(self):
        self._init_mask = None
        self.clicker.click_indx_offset = 0

    @property
    def current_object_prob(self):
        if self.probs_history:
            current_prob_total, current_prob_additive = self.probs_history[-1]
            return np.maximum(current_prob_total, current_prob_additive)
        else:
            return None

    @property
    def is_incomplete_mask(self):
        return len(self.probs_history) > 0

    @property
    def result_mask(self):
        result_mask = self._result_mask.copy()
        
        if self.probs_history:
            result_mask[self.current_object_prob > self.prob_thresh] = self.object_count + 1
        return result_mask

    def get_visualization(self, alpha_blend, click_radius):
        if self.image is None:
            return None

        results_mask_for_vis = self.result_mask
        vis = draw_with_blend_and_clicks(self.image, mask=results_mask_for_vis, alpha=alpha_blend,
                                         clicks_list=self.clicker.clicks_list, radius=click_radius)
        if self.probs_history:
            total_mask = self.probs_history[-1][0] > self.prob_thresh
            results_mask_for_vis[np.logical_not(total_mask)] = 0
            vis = draw_with_blend_and_clicks(vis, mask=results_mask_for_vis, alpha=alpha_blend)

        # if self.final_pred:
        #     vis = draw_with_blend_and_clicks(vis, mask=self.final_pred*255, alpha=alpha_blend)    

        return vis
