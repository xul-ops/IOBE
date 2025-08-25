import time
from contextlib import nullcontext

import os
import cv2
import pdb
import torch
import imageio
import numpy as np
from collections import Counter
from mtorl.utils.initial import ENV
from mtorl.utils.log import logger
from torch.cuda.amp import GradScaler, autocast
from mtorl.utils.checkpoint import Checkpoint
from mtorl.models.backbones.pos_embed import interpolate_pos_embed_inference

from isutils.interactions import *
from isutils.geprmap import ob_nmst, get_matlab_eng


class Runner(object):
    def __init__(self, cfg):
        self.cfg = cfg

        self.model = ENV.model
        self.data_loaders = ENV.data_loaders        
        self.model_name = cfg.model_name

        self.optimizer = ENV.optimizer
        self.criterion = ENV.criterion
        self.scheduler = ENV.scheduler
        self.inference = cfg.inference
        self._max_epoch = cfg.epoch
        self.start_epoch = cfg.staEpochs
        self.save_dir = cfg.save_dir
        self.scheduler_mode = cfg.scheduler_mode
        assert self.scheduler_mode == 'epoch' or self.scheduler_mode == 'iter'
        
        # interaction params
        self.is_interactive = cfg.is_interactive
        self.startIterEpoch = cfg.startIterEpoch
        self.mb_nms = cfg.matlab_nms
        if self.is_interactive and self.mb_nms:
            self.mb_engine = get_matlab_eng()
        else:
            self.mb_engine = None

        self.training_as_is = False
        self.vis_last_validation = True
        self.max_num_iter = cfg.num_iter
        self.pre_mask = cfg.previous_mask
        self.is_encoding = cfg.is_encoding
        self.is_interaction = cfg.is_interaction
        self.is_radius = cfg.is_radius
        self.p_threshold = cfg.p_threshold
        self.match_radius = cfg.match_radius
        self.candidate_len = cfg.candidate_len
        self.random_range = cfg.random_range
        self.use_edge_path = cfg.with_edge_pro
        self.loss_mask = cfg.occ_loss_mask
        self.val_dataset_name = cfg.val_dataset_name
        self.binary_prev_mask = False

        # amp
        self.amp_context = nullcontext
        if self.cfg.amp:
            self.amp_context = autocast
            self.scaler = GradScaler()

        self.iter = 0
        self.epoch = 0
        self.data_loader = None
        self.data_batch = None
        self.output = None
        self.task_type = cfg.task_type
        self.vars_record = {'b_losses': [], 'o_loss': 0, 'count': 1}
        self.checkpoint = Checkpoint(self)

    def run(self):
        workflow = ['train']
        if self.inference:
            workflow = ['test']
            self._max_epoch = 1
        data_loaders = ENV.data_loaders

        for self.epoch in range(self.start_epoch, self._max_epoch):
            logger.info('*' * 80)
            logger.info(f'epoch_{self.epoch}')
            logger.info('*' * 80)
            for mode in workflow:
                logger.info(f'=> {mode}')
                self.epoch_start_time = time.time()
                epoch_runner = getattr(self, str(mode))
                data_loader = data_loaders[mode]
                if self.epoch + 1 > self.startIterEpoch:
                    if not self.training_as_is:
                        print("Current epoch traning mode: give all interactive fn fp pairs at one time!")
                    else:
                        print("Current epoch traning mode: iteractively give one biggest fn fp pair and train as RITM")
                self._max_iter = len(data_loader)

                epoch_runner(data_loader)

    def train(self, data_loader):
        self.data_loader = data_loader
        self.model.train()
        for self.iter, self.data_batch in enumerate(self.data_loader):

            self.optimizer.zero_grad()

            with self.amp_context():
                self.output = self.batch_processor()
                if self.task_type == "occ_edge":
                    loss = torch.sum(self.output['b_losses'])
                else:
                    loss = sum(self.output['b_losses']) + self.output['o_loss']

                if self.cfg.amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                else:
                    loss.backward()
                    self.optimizer.step()
                   
            self._iter_record()
            if self.scheduler_mode == 'iter':
                self.scheduler.step()

        if self.scheduler_mode == 'epoch':
            self.scheduler.step()

        self._iter_record(force=True)

        if self.epoch >= self.startIterEpoch:
            self.checkpoint.save_state_dict(self) 
            # self.checkpoint.save_checkpoint(self)

        if self.epoch >= 0:
            # self.checkpoint.save_state_dict(self)
            self.checkpoint.save_checkpoint(self)


    def test(self, data_loader):
        self.data_loader = data_loader
        self.model.eval()
        pos_nums = list()
        neg_nums = list()
        neg_nums2 = list()
        names_list = list()
        if self.epoch >= 1:
            for self.iter, self.data_batch in enumerate(self.data_loader):
                with torch.no_grad(), self.amp_context():
                    self.output = self.batch_processor(validation=True)
   
                c_pos_nums, c_neg_nums, c_names, c_neg_nums2 =  self.output['pos_num'], self.output['neg_num'], self.output['image_name'], self.output['neg_num2']
                pos_nums.extend(c_pos_nums)
                neg_nums.extend(c_neg_nums)
                neg_nums2.extend(c_neg_nums2)

                names_list.extend(c_names)
                self.checkpoint.save_test_results(self)
                self._iter_record(interval_iters=1000)
 
            self.save_iternum(pos_nums, neg_nums, neg_nums2, names_list)
            self._iter_record(force=True)
            self.checkpoint.save_test_results(self, epoch_end=True)

    def evaluate(self, data_loader, state_dict_path, is_maevit=False):

        resume_checkpoint = torch.load(state_dict_path)
        if not state_dict_path.endswith("_st.pth"):
            # resume_checkpoint = torch.load(state_dict_path)
            state_dict = resume_checkpoint.get('state_dict', resume_checkpoint)
            self.model.load_state_dict(state_dict)
            self.epoch = resume_checkpoint['meta']['epoch']
        else:
            self.model.load_state_dict(resume_checkpoint)             
            self.epoch = int(state_dict_path.split("/")[-1].replace("_st.pth", ""))

        if is_maevit:
            if self.val_dataset_name in ["synocc", "synocc_rgba"]:
                # not checked yet
                inference_shape = (1080, 1080)
                print("mae vit interpolate pos embed to h, w (1080, 1080)!")
            
            elif self.val_dataset_name == "diode":
                # resize to (896, 896) for adaptive click
                inference_shape = (768, 1024)
                print("mae vit interpolate pos embed to h, w (768, 1024)!")

            elif self.val_dataset_name == "piod":
                # for simpleclick piod, use the same resize (448, 448) as Pascal VOC in their codes, 375, 500 not work
                inference_shape =  (448, 448)
                print("mae vit interpolate pos embed to h, w (448, 448)!")
    
            elif self.val_dataset_name == "entityseg":
                inference_shape =  (896, 896)
                print("mae vit interpolate pos embed to h, w (896, 896)!")
                
            elif self.val_dataset_name == "nyuocpp":
                # 592, 440
                inference_shape =  (672, 672)
                print("mae vit interpolate pos embed to h, w (448, 448)!")
                
            elif self.val_dataset_name == "bsds":
                inference_shape =  (448, 448)
                print("mae vit interpolate pos embed to h, w (448, 448)!")
                
            interpolate_pos_embed_inference(self.model.backbone, inference_shape, "cuda")    

        self.data_loader = data_loader
        self._max_iter = len(data_loader)
        self.epoch_start_time = time.time()
        pos_nums = list()
        neg_nums = list()
        neg_nums2 = list()
        names_list = list()

        self.model.eval()
        for self.iter, self.data_batch in enumerate(self.data_loader):

            with torch.no_grad(), self.amp_context():
                self.output = self.batch_processor(validation=True, val_unlimited=True)

            c_pos_nums, c_neg_nums, c_names, c_neg_nums2  =  self.output['pos_num'], self.output['neg_num'], self.output['image_name'], self.output['neg_num2']
            pos_nums.extend(c_pos_nums)
            neg_nums.extend(c_neg_nums)
            neg_nums2.extend(c_neg_nums2)
            names_list.extend(c_names)

            self.checkpoint.save_test_results(self)
            self._iter_record(interval_iters=1000)
 
        self.save_iternum(pos_nums, neg_nums, neg_nums2, names_list)
        self._iter_record(force=True)
        self.checkpoint.save_test_results(self, epoch_end=True)

    def batch_processor(self, validation=False, val_unlimited=True):

        images, pnped, labels, img_names = self.data_batch["image"], self.data_batch["is_channel"], self.data_batch["labels"].cuda(), self.data_batch['image_name']
        pos_nums, neg_nums, neg_nums2 = [], [], []

        if self.model_name == "cdnet":
            fdm_pnped = self.data_batch['fdm_is_channels']        

        if self.epoch+1 > self.startIterEpoch: 

            self.model.eval()
            fp_masks = np.zeros((images.shape[0], 1, images.shape[2], images.shape[3]))
            fn_masks = np.zeros((images.shape[0], 1, images.shape[2], images.shape[3]))
            # ldcedges = self.data_batch["ldcedge"][:, 0, :, :].numpy()

            if self.model_name == "cdnet":
                fdmfp_masks = np.zeros((images.shape[0], 1, images.shape[2], images.shape[3]))
                fdmfn_masks = np.zeros((images.shape[0], 1, images.shape[2], images.shape[3]))
                # fdm_pnped = torch.zeros_like(fdm_pnped).cuda()
            
            pnped = torch.zeros_like(pnped).cuda()
          
            with torch.no_grad(), self.amp_context():
                if validation:
                    num_iter = -1  # if val_unlimited else 12 # self.max_num_iter*2

                else:
                    # num_iter = self.max_num_iter+2
                    num_iter = random.randint(0, self.max_num_iter+2)

                loop_iter_num = num_iter if self.training_as_is else 1
                loop_cand_num = 1 if self.training_as_is else num_iter

                for i_iter in range(loop_iter_num):
                    
                    # print(self.use_edge_path, loop_iter_num)
                    # use ldc edge map as previous mask (for first iter)
                    # pnped = torch.from_numpy(np.concatenate((fp_masks/255, fn_masks/255, ldcedges), axis=1)) 
                    # img_input = torch.cat((images.cuda(), pnped.float().cuda()), axis=1)
                    # b_x, o_x = self.model(img_input)
                    # boundary_result_p = self.model.getBoundary(b_x)[:, 0]
                    # boundary_result = boundary_result_p.cpu()
                    
                    if self.model_name == "cdnet":
                        fdm_pnped = torch.from_numpy(np.concatenate((fdmfp_masks/255, fdmfn_masks/255), axis=1))
                        b_x = self.model(images.float().cuda(), pnped.float().cuda(), fdm_pnped.float().cuda()) 
                    else:
                        b_x = self.model(images.float().cuda(), pnped.float().cuda())   
            
                    boundary_result_p = torch.sigmoid(b_x['instances']).detach()[:, 0]
                    boundary_result = boundary_result_p.cpu()                    
                    
                    if self.use_edge_path:
                        # use ldc edge map to search the possible connections between two points
                        # edge_nms = self.data_batch["ldcedge"][:, 0, :, :].numpy()
                        # use previous mask to search the possible connections between two points (better than ldc edge )
                        edge_nms = boundary_result.numpy()
                    else:
                        edge_nms = None
                                                                    
                    # for RITM way, should select an interaction(lens/error bigger one) between fn fp instead of one pair
                    if self.model_name == "cdnet":
                        fp_masks, fn_masks, obnmsts, pos_nums, neg_nums, fdmfp_masks, fdmfn_masks, neg_nums2 = get_batch_masks_cdnet(boundary_result, labels[:,0].cpu().numpy(), p_threshold=self.p_threshold, 
                                                                                                                                    match_radius=self.match_radius, random_range=self.random_range, 
                                                                                                                                    candidate_len=self.candidate_len, need_sort=True, cand_num=loop_cand_num, 
                                                                                                                                    encoding=self.is_encoding, interaction=self.is_interaction, radius=self.is_radius,  
                                                                                                                                    recursion_outlen=200, final_pmask=fp_masks[:, 0], final_nmask=fn_masks[:, 0], 
                                                                                                                                    use_mbnms=self.mb_nms, matlab_engine=self.mb_engine, epmap=None, epgraph_t=0.1, 
                                                                                                                                    cdnet_fdm_mask=True, fdm_final_pmask=fdmfp_masks[:, 0], fdm_final_nmask=fdmfn_masks[:, 0])
                    else:
                        fp_masks, fn_masks, obnmsts, pos_nums, neg_nums, neg_nums2 = get_batch_masks(boundary_result, labels[:,0].cpu().numpy(), p_threshold=self.p_threshold, 
                                                                                                        match_radius=self.match_radius, random_range=self.random_range,
                                                                                                        candidate_len=self.candidate_len, need_sort=True, cand_num=loop_cand_num, 
                                                                                                        encoding=self.is_encoding, interaction=self.is_interaction, radius=self.is_radius, 
                                                                                                        recursion_outlen=200, final_pmask=fp_masks[:, 0], final_nmask=fn_masks[:, 0], 
                                                                                                        use_mbnms=self.mb_nms, matlab_engine=self.mb_engine, epmap=None, epgraph_t=0.1)

                    if self.model_name == "cdnet":
                        fdm_pnped = torch.from_numpy(np.concatenate((fdmfp_masks/255, fdmfn_masks/255), axis=1))

                    if not self.pre_mask:
                        pnped = torch.from_numpy(np.concatenate((fp_masks/255, fn_masks/255), axis=1))

                    else:
                        # use ldc edge map as previous mask
                        # pnped = torch.from_numpy(np.concatenate((fp_masks/255, fn_masks/255, ldcedges), axis=1)) 

                        # use occ boundary nms previous edge
                        if self.binary_prev_mask:
                            obnmsts = (obnmsts > self.p_threshold) * 1.0
                        pnped = torch.from_numpy(np.concatenate((fp_masks/255, fn_masks/255, obnmsts), axis=1)) 
                        # pnped = torch.from_numpy(np.concatenate((fp_masks/255, fn_masks/255, boundary_result[:, np.newaxis, :, :]), axis=1))   

                if validation and self.vis_last_validation:
                    self.check_vis(b_x, labels, fn_masks, fp_masks, img_names)  
 

            labels = labels.cuda()
            if not validation:
                self.model.train()  
                # if validation, should be eval mode, turn off BN/Dropout 

        if self.model_name == "cdnet":
            # fdm_pnped = torch.from_numpy(np.concatenate((fdmfp_masks/255, fdmfn_masks/255), axis=1))
            b_x = self.model(images.float().cuda(), pnped.float().cuda(), fdm_pnped.float().cuda()) 
        else:
            b_x = self.model(images.float().cuda(), pnped.float().cuda())    
        
        fnfp = pnped[:, 0:2].cuda()    
        boundary_losses = None
        if labels is not None and self.criterion is not None:
            boundary_losses = self.criterion(b_x, labels[:, 0].unsqueeze(1), fnfp)

        return {
            'image_name': img_names,
            'b_losses': boundary_losses,
            'o_loss': torch.Tensor([0.0]),
            'b_x': b_x,
            'o_x': torch.Tensor([0.0]),
            'pos_num': pos_nums,
            'neg_num': neg_nums,
            'neg_num2': neg_nums2,
        }

    def _iter_record(self, force=False, interval_iters=100):

        if self.output is not None and self.output['o_loss'] is not None:
            self.vars_record['o_loss'] = self.output['o_loss'].item()
            self.vars_record['count'] += 1

        self.vars_record['b_losses'] = [torch.sum(self.output['b_losses']).item()]     

        if self.iter % interval_iters == 0 or force:
            lr = self.optimizer.param_groups[0]['lr']
            count = self.vars_record['count']
            b_losses_str = [f'{b_loss / count:.3e}' for b_loss in self.vars_record['b_losses']]
            log_str = f' Epoch [{self.epoch}/{self._max_epoch}][{self.iter}/{self._max_iter}] | '
            log_str += f'lr: {lr:.3e} | '
            log_str += f"b_losses: {b_losses_str} | "
            log_str += f"o_loss: {self.vars_record['o_loss'] / count:.3e} | "
            log_str += f'time: {(time.time() - self.epoch_start_time):.1f}s'
            logger.info(log_str)
            self.vars_record = {'b_losses': [], 'o_loss': 0, 'count': 1}
            
    def save_iternum(self, pos_nums, neg_nums, neg_nums2, img_names):
        iternum_path = self.save_dir + "/iter_num/"

        if not os.path.exists(iternum_path):
            os.makedirs(iternum_path)
        
        if self.epoch+1 > self.startIterEpoch:

            pos_nums = np.array(pos_nums) 
            neg_nums = np.array(neg_nums)
            avg_pos = np.average(pos_nums)
            avg_neg = np.average(neg_nums)
            neg_nums2 = np.array(neg_nums2)
            avg_neg2 = np.average(neg_nums2)
            
            avg_pos = np.round(avg_pos*100, 2)  
            avg_neg = np.round(avg_neg*100000, 2) 
            avg_neg2 = np.round(avg_neg2*10000, 2)   
                                
            save_list = [np.array([img_names[i], pos_nums[i], neg_nums[i]]) for i in range(len(pos_nums))]
            file_name = "epoch_" + str(self.epoch) + "_avgFN_" + str(avg_pos)  + "_avgFP_" + str(avg_neg) + "_avgFP2_" + str(avg_neg2) + ".npy"
            np.save(iternum_path+file_name, np.array(save_list))             
            
    def check_vis(self, b_x, labels, fn_masks, fp_masks, img_names):
        root_fnfp_path = self.save_dir + "/fnfp/"
        root_pre_path = self.save_dir + "/pre/"
        fnfp_path = root_fnfp_path + "epoch_{}/".format(self.epoch) 
        pre_path = root_pre_path + "epoch_{}/".format(self.epoch) 
                
        if not os.path.exists(root_fnfp_path):
            os.makedirs(root_fnfp_path)
            os.makedirs(root_pre_path)
        if not os.path.exists(fnfp_path):
            os.makedirs(fnfp_path)
            os.makedirs(pre_path)            
            
        # pre = self.model.getBoundary(b_x)[0, 0]
        pre = torch.sigmoid(b_x['instances']).detach()[0, 0]

        obnmst = ob_nmst(pre.detach().cpu().numpy(), self.p_threshold)
        nmst_save = pre_path +img_names[0]+"_pynmst{}.png".format(int(self.p_threshold*10))
        # print(obnmst.shape)
        # from collections import Counter
        # print(Counter(obnmst.reshape(-1)))
        obnmst = obnmst.reshape(pre.shape[0], pre.shape[1], 1)
        cv2.imwrite(nmst_save, (1-obnmst)*255)

        pre = (1 - pre) * 255
        pre = pre.cpu().type(torch.uint8)
        pre_save = pre_path +img_names[0]+"_ob.png"
        imageio.imwrite(pre_save, pre)

        gt_crop = labels[0,0].cpu().numpy().reshape(pre.shape[0], pre.shape[1], 1) * [255, 255, 255]
        fnfp_save = fnfp_path+"gt_"+img_names[0]
        vis_np_mask(gt_crop, fp_masks[0, 0].reshape(pre.shape[0], pre.shape[1], 1), fn_masks[0,0].reshape(pre.shape[0], pre.shape[1], 1), fnfp_save)    
        # rgb_crop = images[0, 0].cpu().numpy().reshape(obnmst.shape[0], obnmst.shape[1], 1)
        # vis_inter(rgb_crop,  fp_masks[0,0].reshape(obnmst.shape[0], obnmst.shape[1], 1), fn_masks[0,0].reshape(obnmst.shape[0], obnmst.shape[1], 1), save_vis_path+"rgb_"+img_names[0])                     
                                
