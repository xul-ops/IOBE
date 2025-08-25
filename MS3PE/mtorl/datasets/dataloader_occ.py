import os
import cv2
import math
import h5py
import torch
import random
import numpy as np
from collections import Counter

from PIL import Image, ImageStat
from torch.utils import data
from torchvision.io import read_image
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.io.image import ImageReadMode
from isutils.interactions import *

from mtorl.datasets.prefetcher import PreFetcher, PreFetcher2
from mtorl.datasets.transform_occ import crop_with_label, get_A_transform, transform_with_A
PI = np.pi # 3.1416  # math.pi
negtive_use_canny = True


def add_rgba_background(rgba_path, bg_path, save_path="./", save_name="default.png"):
    render_img = Image.open(rgba_path)
    bg_img = Image.open(bg_path)
    # bg_img = bg_img.transpose(Image.FLIP_TOP_BOTTOM)
    bg_img = bg_img.resize((render_img.width, render_img.height))

    assert render_img.mode == 'RGBA'
    # assert bg_img.width == render_img.width
    # assert bg_img.height == render_img.height

    bg_img.paste(render_img, (0, 0), mask=render_img)
      
    return bg_img
    

def get_edge_channel(img_name, img_rgb=None, add_edge_method = "LDC", need_nms=False, add_edge_weight = 0.5, random_add_edge=True, canny_t1=100):
    # we can do this after data augmentation directly
    if add_edge_method == "canny":
        # Image.open is RGB, cv2 BGR
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        canny_edge = cv2.Canny(np.array(img_gray).astype(np.uint8), canny_t1, canny_t1*3)
        # print(Counter(canny_edge.reshape(-1)))
        edge = (canny_edge.reshape(img_rgb.shape[0], img_rgb.shape[1], 1)).astype(np.float32) 
        
    elif add_edge_method == "LDC":
        ldc_ep_path = "./occdata/ldcpiod/"
        path = ldc_ep_path + img_name + ".jpg"
        ep_ldc = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        edge = np.array(ep_ldc).reshape(ep_ldc.shape[0], ep_ldc.shape[1], 1)
        edge = (255.0-edge).astype(np.float32) / 255
        if need_nms:
            # python nms or matlab nms
            edge = ob_nmst(edge.reshape(edge.shape[0], edge.shape[1]), 0.51)            
            edge = np.array(edge).reshape(edge.shape[0], edge.shape[1], 1).astype(np.float32)
    

    return edge



def get_interaction_channels(pred_crop, gt_crop, img_name, candidate_len=30, radius=2, interaction="scribbles", iencoding="scribbles", random_range=(-1, 1), num_points=3, op_way="middle", is_inter=False, match_radius=4, recursion_outlen=200, epmap=None, fdm_cdnet=False):

    if is_inter:
        return torch.from_numpy(np.zeros((3, gt_crop.shape[0], gt_crop.shape[1])))
    
    if iencoding == "edge_path": # unused
        edge_graph = build_edge_graph(epmap.reshape(epmap.shape[0], epmap.shape[1]), 0.51) 
        # cv2.imwrite("./edgevis/ep_"+img_name+"_.png", epmap*255)
        
    else:
        edge_graph = 0
    
    if num_points == -1 or num_points==0:
        num_points = 1   

    # between 0 and max iter num     
    num_points = random.randint(0, num_points)
    # num_points = num_points

    # print(random_range)

    gt_crop_r = gt_crop.reshape(gt_crop.shape[0], gt_crop.shape[1])
    zero_pred = np.zeros((gt_crop.shape[0], gt_crop.shape[1])) 
    pos_choices, _ = get_fnfp_candidates(gt_crop_r, zero_pred, match_radius=4, candidate_len=30,  recursion_outlen=recursion_outlen)    
    
    if len(pos_choices) != 0: # and num_points != 0:
        if len(pos_choices) >= num_points:
            pos_segment = random.sample(pos_choices, num_points)
            pnum_points = num_points
        else:
            # pos_segment = random.sample(pos_choices, len(pos_choices))
            pos_segment = pos_choices
            pnum_points = len(pos_segment)
        
        p_mask = generate_interaction_maps(pos_segment, gt_crop.shape, num_points=pnum_points, encoding=iencoding, interaction=interaction, random_range=random_range, radius=radius, op_way=op_way, epmap=edge_graph)
        if fdm_cdnet:
            fdm_pmask = generate_interaction_maps(pos_segment, gt_crop.shape, num_points=pnum_points, encoding=iencoding, interaction=interaction, random_range=random_range, radius=24, op_way=op_way, epmap=edge_graph)
                    
    else:
        p_mask = np.zeros((gt_crop.shape[0], gt_crop.shape[1]))
        ppoints = list()
        if fdm_cdnet:
            fdm_pmask = np.zeros((gt_crop.shape[0], gt_crop.shape[1]))
            
    if negtive_use_canny:
        img_gray = cv2.cvtColor(pred_crop, cv2.COLOR_RGB2GRAY)
        canny_edge = cv2.Canny(np.array(img_gray).astype(np.uint8), 100, 200)
        _, neg_choices = get_fnfp_candidates(gt_crop_r, canny_edge/255, match_radius=4, candidate_len=30, recursion_outlen=recursion_outlen)
    
    if negtive_use_canny and len(neg_choices) != 0:
        if len(neg_choices) >= num_points:
            # neg_segment = random.choice(neg_choices)
            neg_segment = random.sample(neg_choices, num_points)
            nnum_points = num_points
        else:
            neg_segment = neg_choices
            nnum_points = len(neg_choices)
        n_mask = generate_interaction_maps(neg_segment, gt_crop.shape, num_points=nnum_points, encoding=iencoding, interaction=interaction, random_range=random_range, radius=radius, op_way=op_way, epmap=edge_graph)
        if fdm_cdnet:
            fdm_nmask = generate_interaction_maps(neg_segment, gt_crop.shape, num_points=nnum_points, encoding=iencoding, interaction=interaction, random_range=random_range, radius=24, op_way=op_way, epmap=edge_graph)          
    else:
        n_mask = np.zeros((gt_crop.shape[0], gt_crop.shape[1]))
        npoints = list()
        if fdm_cdnet:
            fdm_nmask = np.zeros((gt_crop.shape[0], gt_crop.shape[1]))         

    p_mask = p_mask.reshape(gt_crop.shape[0], gt_crop.shape[1], 1) 
    n_mask = n_mask.reshape(gt_crop.shape[0], gt_crop.shape[1], 1) 

    # for IS method cdnet
    if fdm_cdnet:
        fdmp_mask = fdm_pmask.reshape(gt_crop.shape[0], gt_crop.shape[1], 1) 
        fdmn_mask = fdm_nmask.reshape(gt_crop.shape[0], gt_crop.shape[1], 1)     

    # save_dir = "./geo/" 
    # vis_np_mask(pred_crop, p_mask, n_mask, save_dir+"rgb_"+img_name)
    # vis_np_mask(gt_crop * [255, 255, 255], p_mask, n_mask, save_dir+"gt_"+img_name,  p_points=ppoints, n_points=npoints)

    pep_pred = np.zeros((gt_crop.shape[0], gt_crop.shape[1], 1))
    add_channels = np.transpose(np.concatenate((p_mask/255.0, n_mask/255.0, pep_pred), axis=-1), axes=(2, 0, 1))
    if fdm_cdnet:
        # print(fdm_pmask.shape, fdm_nmask.shape)
        fdm_add_channels = np.transpose(np.concatenate((fdmp_mask/255.0, fdmn_mask/255.0), axis=-1), axes=(2, 0, 1)) 
        return torch.from_numpy(add_channels), torch.from_numpy(fdm_add_channels)   

    return torch.from_numpy(add_channels)
 

def get_interaction_channels_f(pred_crop, gt_crop, img_name, candidate_len=30, radius=2, interaction="scribbles", iencoding="scribbles", random_range=(-1, 1), num_points=3, op_way="middle", is_inter=False, match_radius=4, recursion_outlen=200, epmap=None, fdm_cdnet=False):
    # ablation test

    # print(gt_crop.shape)
    full_pos = list()
    for i in range(gt_crop.shape[0]):
        for j in range(gt_crop.shape[1]):
            if gt_crop[i, j] != 0:
                full_pos.append((i, j))
    p_mask = generate_interaction_maps([full_pos], gt_crop.shape, num_points=1, encoding=iencoding, interaction=interaction, random_range=random_range, radius=radius, op_way=op_way, epmap=None)
    p_mask = p_mask.reshape(gt_crop.shape[0], gt_crop.shape[1], 1)
    # print(p_mask.shape)
    # print(Counter(p_mask.reshape(-1)))

    n_mask = np.zeros((gt_crop.shape[0], gt_crop.shape[1], 1))
    # save_dir = "./geo/"
    # vis_np_mask(pred_crop, p_mask, n_mask, save_dir+"rgb_"+img_name)
    # vis_np_mask(gt_crop * [255, 255, 255], p_mask, n_mask, save_dir+"gt_"+img_name,  p_points=ppoints, n_points=npoints)


    pep_pred = np.zeros((gt_crop.shape[0], gt_crop.shape[1], 1))
    # add_channels = np.transpose(p_mask/255.0, axes=(2, 0, 1))
    add_channels = np.transpose(np.concatenate((p_mask/255.0, n_mask/255.0, pep_pred), axis=-1), axes=(2, 0, 1))

    return torch.from_numpy(add_channels)


def transfer_label(img_paths, is_ob3=False, left_rule=True, keep_n180=True, angle_degree=False):

    # counter_list = list()
    # BGR is ok
    img_ob_path, img_oo_path = img_paths
    img_oo = cv2.imread(img_oo_path) # , cv2.IMREAD_UNCHANGED)
    if is_ob3:
        img_ob_path = img_ob_path.replace("OB", "OB3")
        img_ob = cv2.imread(img_ob_path)
    else:
        img_ob = cv2.imread(img_ob_path)

    img_label = np.zeros((img_ob.shape[0], img_ob.shape[1], 2))

    for i in range(img_ob.shape[0]):
        for j in range(img_ob.shape[1]):
        
            # ob, ob3 to be finished
            if img_ob[i, j, :].tolist() != [0, 0, 0]: # == [255, 255, 255]:
                img_label[i, j, 0] = 1

            # oo
            # if img_oo[i, j, :].tolist() != [112, 112, 112]:
                angle = img_oo[i, j, 0]
                # image/screen coordinates angle
                angle = int(angle * 360 / 255)
                # one easy way is not using opencv img store oo info, use numpy npy.
                # angle_lists = np.array([0, 45, 90, 135, 180, 225, 270, 315, 360])
                if angle == 269 or angle == 224 or angle == 314:
                    angle += 1
                # if we subtract 90 degree, it follows the left rule
                if left_rule:
                    angle = angle - 90
                if not left_rule and keep_n180 and angle == 360:
                    # special angle, cf. code geocc
                    angle = -180
                if angle > 180:
                    # [0, 360] -> [-180, 180]
                    angle = angle - 360
                # counter_list.append(angle)
                if angle_degree:
                    img_label[i, j, 1] = angle / 180
                else:
                    img_label[i, j, 1] = angle * PI / 180                


    return img_label


def get_gt_image(img_paths, is_ob3=False):
    """
    BGR is ok
    """
    img_ob_path, img_oo_path = img_paths
    img_oo = cv2.imread(img_oo_path)
    if is_ob3:
        img_ob_path = img_ob_path.replace("OB", "OB3")
        img_ob = cv2.imread(img_ob_path)
    else:
        img_ob = cv2.imread(img_ob_path)
    return img_ob, img_oo


def transfer_transform_gt(img_ob, img_oo, angle_degree=False):
    """
    transfer img transformed/augmented by other library, not A
    """
    white_point = [255, 255, 255]
    img_label = np.zeros((img_ob.shape[0], img_ob.shape[1], 2))

    for i in range(img_oo.shape[0]):
        for j in range(img_oo.shape[1]):
            # oo
            if img_oo[i, j, :].tolist() != [112, 112, 112]:
                angle = img_oo[i, j, 0]
                angle = angle * 360 / 255
                # if we subtract 90 degree, it follows the left rule
                angle = int(angle - 90)
                if angle > 180:
                    # [0, 360] -> [-180, 180]
                    angle = angle - 360
                if angle_degree:
                    img_label[i, j, 1] = angle
                else:
                    img_label[i, j, 1] = angle * PI / 180
            if img_ob[i, j, :].tolist() == white_point:
                img_label[i, j, 0] = 1

    return img_label


def read_labels_from_png(png_files):
    edge_label_file, ori_label_file = png_files
    edge_label = read_image(edge_label_file, ImageReadMode.GRAY).float().div(255)
    ori_label = read_image(ori_label_file, ImageReadMode.GRAY).float().div(255)
    ori_label = ori_label * (2 * PI) - PI
    labels = torch.cat([edge_label, ori_label], dim=0)
    return labels.numpy()


def read_labels_from_h5(h5_file):
    with h5py.File(h5_file, 'r') as f:
        labels = f['label'][0]
    return np.array(labels)  # torch.from_numpy(labels).float()


def init_gt_files(root_path, img_list_path, dataset_name, train_copy=False):
    """
    this function can be a class static method
    """
    # synocc is OB-FUTURE, diode is OB-DIODE, entityseg is OB-Entityseg
    assert dataset_name in ["synocc", "synocc_rgba", "cmu", "piod", "bsds", "nyuocpp", "diode", 'entityseg']
    
    if dataset_name == "synocc":
        with open(img_list_path, 'r') as f:
            names = f.readlines()
        
        # cnfl = ['18316', '00427', "11414"]
        names = [x.replace('\n', '') for x in names]
        img_list = [os.path.join(root_path, name, 'rgb_1.png') for name in names]
        # dis_fOB.png
        gt_list = [[os.path.join(root_path, name, 'dis_fOB.png'),
                    os.path.join(root_path, name, 'dis_fOO.png')]
                   for name in names]
        img_names = names
        return img_list, gt_list, img_names
        
    elif dataset_name == "synocc_rgba":
        # one ablation
        with open(img_list_path, 'r') as f:
            names = f.readlines()

        names = [x.replace('\n', '') for x in names]   
        img_list = [os.path.join(root_path, name, 'brgba.png') for name in names]
        # dis_fOB.png
        gt_list = [[os.path.join(root_path, name, 'dis_fOB.png'), 
                    os.path.join(root_path, name, 'dis_fOO.png')]
                   for name in names]
        img_names = names
        return img_list, gt_list, img_names
        
    elif dataset_name == "piod":
        # label_path = os.path.join(root_path, 'Aug_HDF5EdgeOriLabel')
        with open(img_list_path, 'r') as f:
            names = f.readlines()
        names = [x.replace('\n', '') for x in names]
        img_list = [os.path.join(root_path, 'Aug_JPEGImages', f'{name}.jpg') for name in names]
        gt_list = [[os.path.join(root_path, 'Aug_PngEdgeLabel', f'{name}.png'),
                    os.path.join(root_path, 'Aug_PngOriLabel', f'{name}.png')]
                   for name in names]
        img_names = names
        return img_list, gt_list, img_names

    elif dataset_name == "bsds":
        with open(img_list_path, 'r') as f:
            names = f.readlines()
        names = [x.replace('\n', '') for x in names]

        img_list = list()
        gt_list = list()
        img_names = list()
        for name in names:
            img_file, labels_file = name.split(' ')
            img_list.append(f'{root_path}/{img_file}')
            gt_list.append(f'{root_path}/{labels_file}')
            img_names.append(img_file.split("/")[-1].replace(".jpg", ""))
        if train_copy:
            img_list_1, gt_list_1, img_names_1 = img_list.copy(), gt_list.copy(), img_names.copy()
            copy_time = 4
            for i in range(copy_time):
                img_list.extend(img_list_1)
                gt_list.extend(gt_list_1)
                img_names.extend(img_names_1)
        return img_list, gt_list, img_names

    elif dataset_name == "nyuocpp":
        with open(img_list_path, 'r') as f:
            names = f.readlines()
        names = [x.replace('\n', '') for x in names]
        img_list = [os.path.join(root_path, 'imgs/'+name) for name in names]
        gt_list = [[os.path.join(root_path, 'ob/'+name),
                    os.path.join(root_path, 'oo/'+name)]
                   for name in names]
        img_names = [x.replace('.png', '') for x in names]
        if train_copy:
            img_list_1, gt_list_1, img_names_1 = img_list.copy(), gt_list.copy(), img_names.copy()
            copy_time = 16
            for i in range(copy_time):
                img_list.extend(img_list_1)
                gt_list.extend(gt_list_1)
                img_names.extend(img_names_1)        
        return img_list, gt_list, img_names
    
    elif dataset_name == "diode":
        with open(img_list_path, 'r') as f:
            names = f.readlines()
        names = [x.replace('\n', '') for x in names]
        img_names = [x.split('/')[-1] for x in names]
        img_list = [os.path.join(root_path, name) for name in names]
        gt_list = [[os.path.join(root_path, 'gt/'+name),
                    os.path.join(root_path, 'gt/'+name)]
                   for name in img_names]
        img_names_ = [x.replace('.png', '') for x in img_names]
        return img_list, gt_list, img_names_
        
    elif dataset_name == "entityseg":
        with open(img_list_path, 'r') as f:
            names = f.readlines()
        names = [x.replace('\n', '') for x in names]
        img_names = [x.split('/')[-1] for x in names]
        
        img_list = [os.path.join(root_path, name+".jpg") for name in names]
        gt_list = [[os.path.join(root_path, 'gt/'+name+".png"),
                    os.path.join(root_path, 'gt/'+name+".png")]
                   for name in img_names]
        img_names = [x.replace('.jpg', '') for x in img_names]

        if train_copy:
            # for train
            # img_list = [os.path.join(root_path, name) for name in names]
            # gt_list = [[os.path.join(root_path, 'gt/'+name.split(".")[0]+".png"),
            #         os.path.join(root_path, 'gt/'+name.split(".")[0]+".png")]
            #        for name in img_names]
            # img_names = [x.split(".")[0] for x in img_names]
            img_list_1, gt_list_1, img_names_1 = img_list.copy(), gt_list.copy(), img_names.copy()
            copy_time = 3
            for i in range(copy_time):
                img_list.extend(img_list_1)
                gt_list.extend(gt_list_1)
                img_names.extend(img_names_1)
        
        return img_list, gt_list, img_names

    elif dataset_name == "cmu":
        # label_path = os.path.join(root_path, 'Aug_HDF5EdgeOriLabel')
        with open(img_list_path, 'r') as f:
            names = f.readlines()
        names = [x.replace('\n', '').replace(".png", "") for x in names]
        img_list = [os.path.join(root_path, 'imgs_r/' + name+".png") for name in names]
        gt_list = [[os.path.join(root_path, 'gt/'+name+"_objectgroundtruth.png"),
                    os.path.join(root_path, 'gt/'+name+"_objectgroundtruth.png")]
                   for name in names]
        img_names = names
        return img_list, gt_list, img_names
               
    else:
        print("Add new occ dataset.")


class OccDataset(data.Dataset):
    def __init__(self, root_path, img_list_path, dataset_name, train_transform=None, A_transform=None, additional_train_trans=True, cfg=None):

        if train_transform is not None:
            train_copy = True
            print("Copy datasets for nyuocpp, bsds, entityseg")
        else:
            train_copy = False
        
        img_list, gt_list, img_names = init_gt_files(root_path, img_list_path, dataset_name, train_copy=train_copy)
        # img_list, gt_list, img_names = init_gt_files(root_path, img_list_path, dataset_name)

        self.img_list = img_list
        self.gt_list = gt_list
        self.img_name_list = img_names
        self.dataset_name = dataset_name
        self.train_transform = train_transform
        self.A_transform = A_transform
        self.additional_train_trans = additional_train_trans
        self.model_name = cfg.model_name
        self.backbone_name = cfg.backbone_name
        
        # adaptive click also will resize didoe  
        # synocc ? 
        if self.backbone_name.startswith("plainvit_") and self.dataset_name in ["piod", "entityseg", "nyuocpp", "bsds"] and train_transform is None:
            self.vit_resized = True
        else:
            self.vit_resized = False
        
        # iter params
        self.is_inter = cfg.is_interactive
        self.pre_mask = cfg.previous_mask
        self.crop_size = cfg.random_crop_size
        self.candidate_len = cfg.candidate_len
        self.is_radius = cfg.is_radius
        self.match_radius = cfg.match_radius
        self.is_interaction = cfg.is_interaction
        self.is_encoding = cfg.is_encoding
        self.random_range = cfg.random_range
        self.max_iter = cfg.num_iter
        self.add_edge = cfg.with_edge_pro
        self.add_edge_nms = True
        
        diode_wall_path = root_path + "/walls/"      
        if train_transform is None:
            self.diode_walls_list = [diode_wall_path + "00000_00000_indoors_310_020.png"]
        else:
            if dataset_name == "synocc_rgba":
                diode_walls = os.listdir(diode_wall_path)
                self.diode_walls_list = [diode_wall_path + item for item in diode_walls]
                self.diode_walls_list.remove((diode_wall_path+"00000_00000_indoors_310_020.png"))
            else:
                self.diode_walls_list = list()
        if self.is_inter:
            self.add_is_channels = True

        if dataset_name == "synocc":
            normalize = transforms.Normalize(mean=[0.780, 0.771, 0.771], std=[0.203, 0.206, 0.217])
        else:
            # ImageNet mean std, pre-process of pre-trained model of pytorch resnet-50
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # coco  mean = [0.471, 0.448, 0.408], std = [0.234, 0.239, 0.242]

        self.trans = transforms.Compose([
            transforms.ToTensor(),  # (H,W,3)
            normalize])

    @staticmethod
    def _get_sample(img_idx, gt_idx, dataset_name, walls=[], vit_resized=False):
        # img data
        img = Image.open(img_idx)

        if dataset_name == "synocc":
            # label data(H,W,2),  0-edgemap, 1-orientmap
            label_gt = transfer_label(gt_idx)
            
        elif dataset_name == "synocc_rgba":
            bg_img = random.choice(walls)
            img = add_rgba_background(img_idx, bg_img, "./")            
            # label data(H,W,2),  0-edgemap, 1-orientmap
            label_gt = transfer_label(gt_idx)

        elif dataset_name == "cmu":
            label_gt = transfer_label(gt_idx)
            
        elif dataset_name == "piod":
            img = img.convert('RGB')

            # label = read_labels_from_h5(gt_idx)
            if vit_resized:
                # for simpleclick piod, use the same resize (448, 448) as Pascal VOC in their codes
                target_shape = (448, 448)
                if img.size[0] != target_shape[0] or img.size[1] != target_shape[1]:
                    img = img.resize(target_shape)
                    gt_img = Image.open(gt_idx[0]).resize(target_shape)
                    label_gt = np.array(gt_img).reshape(target_shape[0], target_shape[1], 1)/255
                    label_gt = np.concatenate((label_gt, label_gt), axis=-1)
                else:            
                    label_gt = read_labels_from_png(gt_idx)
                    label_gt = np.transpose(label_gt, axes=(1, 2, 0))
            else:            
                label_gt = read_labels_from_png(gt_idx)
                label_gt = np.transpose(label_gt, axes=(1, 2, 0))
                
        elif dataset_name == "bsds":
            img = img.convert('RGB')
            # label = read_labels_from_png(gt_idx)
            label_gt = read_labels_from_h5(gt_idx)
            label_gt = np.transpose(label_gt, axes=(1, 2, 0))

            if vit_resized:
                target_shape = (448, 448)
                if img.size[0] != target_shape[0] or img.size[1] != target_shape[1]:
                    img = img.resize(target_shape)
                    label_gt = read_labels_from_h5(gt_idx)
                    label_gt = np.transpose(label_gt, axes=(1, 2, 0))
                    label_gt = label_gt[:, :, 0]
                    gt_img = Image.fromarray(label_gt).resize(target_shape)
                    # print(np.array(gt_img))
                    # gt_img = Image.open(gt_idx[0]).resize(target_shape)
                    label_gt = np.array(gt_img).reshape(target_shape[0], target_shape[1], 1) # /255
                    label_gt = np.concatenate((label_gt, label_gt), axis=-1)

        elif dataset_name == "nyuocpp":
            label_gt = read_labels_from_png(gt_idx)
            label_gt = np.transpose(label_gt, axes=(1, 2, 0))

            if vit_resized:
                target_shape = (672, 672)
                if img.size[0] != target_shape[0] or img.size[1] != target_shape[1]:
                    img = img.resize(target_shape)
                    gt_img = Image.open(gt_idx[0]).resize(target_shape)
                    label_gt = np.array(gt_img).reshape(target_shape[0], target_shape[1], 1)/255
                    label_gt = np.concatenate((label_gt, label_gt), axis=-1)
            else:
                label_gt = read_labels_from_png(gt_idx)
                label_gt = np.transpose(label_gt, axes=(1, 2, 0))
            
        elif dataset_name == "diode":
            # should add crop in the future, since some edge loop exsit
            label_gt = transfer_label(gt_idx) 

        elif dataset_name == "entityseg":
            # should add crop in the future, since some edge loop exsit
            label_gt = transfer_label(gt_idx)  

            if vit_resized:
                target_shape = (896, 896)
                if img.size[0] != target_shape[0] or img.size[1] != target_shape[1]:
                    img = img.resize(target_shape)
                    gt_img = Image.open(gt_idx[0]).resize(target_shape)
                    # print(np.array(gt_img).shape)
                    label_gt = np.array(gt_img)[:, :, 0].reshape(target_shape[0], target_shape[1], 1)/255
                    label_gt = np.concatenate((label_gt, label_gt), axis=-1)
                              
        else:
            raise ValueError("Wrong occ dataset.")

        img = np.array(img).astype(np.float32)  # np.float64

        return img, label_gt

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        img, label_gt = self._get_sample(self.img_list[idx], self.gt_list[idx], self.dataset_name, self.diode_walls_list, vit_resized=self.vit_resized)
        
        # not used
        if self.add_edge:
            # strong aug
            edge = get_edge_channel(self.img_name_list[idx], add_edge_method = "LDC", need_nms=self.add_edge_nms)
            # if want to use additional transform, can concat dt with edge
            img = np.concatenate((img, edge), axis=-1)

        # image transform/augmenter
        if (self.train_transform is not None) and (self.A_transform is not None):
            img_transformed, gt_transformed = self.train_transform(self.A_transform, img, np.array(label_gt), self.additional_train_trans)
        elif self.train_transform is not None:
            img_transformed, gt_transformed = self.train_transform(img, label_gt, self.crop_size)
        else:
            img_transformed, gt_transformed = img, label_gt
            # if self.dataset_name in ["synocc_rgba", "synocc"]:
            #     # crop for fast evaluations
            #     if self.img_name_list[idx].startswith("01306"):
            #         img_transformed, gt_transformed = img[:768, :1024], label_gt[:768, :1024]
            #     else:
            #         img_transformed, gt_transformed = img[1080-768:, :1024], label_gt[1080-768:, :1024]            
            # else:
            #     img_transformed, gt_transformed = img, label_gt
            
        # normalize and to tensor
        if self.add_edge:
            edge = img_transformed[:,:, -1].reshape(img_transformed.shape[0], img_transformed.shape[1], 1)
            img_transformed = img_transformed[:,:, 0:3]
            epmap = edge
        else:
            epmap = None

        if self.add_is_channels:
            gt_trans_edge = gt_transformed[:,:, 0].reshape(gt_transformed.shape[0], gt_transformed.shape[1], 1)

            if self.model_name == "cdnet":  
                is_channels, fdm_is_channels = get_interaction_channels(img_transformed, gt_trans_edge, self.img_name_list[idx], candidate_len=self.candidate_len, radius=self.is_radius, interaction=self.is_interaction, iencoding=self.is_encoding, num_points=self.max_iter, random_range=self.random_range, is_inter=False, epmap=epmap, fdm_cdnet=True)     
            else:                
                is_channels = get_interaction_channels(img_transformed, gt_trans_edge, self.img_name_list[idx], candidate_len=self.candidate_len, radius=self.is_radius, interaction=self.is_interaction, iencoding=self.is_encoding, num_points=self.max_iter, random_range=self.random_range, is_inter=False, epmap=epmap)
            # img_tensor = torch.cat((img_tensor, is_channels.float()), axis=0)

        img_transformed = img_transformed / 255.0         
        # normalize and to tensor                    
        img_tensor = self.trans(img_transformed)
        # img_transformed = img_transformed[:,:,[2,1,0]] - np.array([104.0,116.6,122.6])
        # img_transformed = np.transpose(img_transformed,axes=(2,0,1))
        # img_tensor = torch.from_numpy(img_transformed).float()
                    
        # (2, H, W)
        gt_transformed = np.transpose(gt_transformed, axes=(2, 0, 1))                    
        gt_tensor = torch.from_numpy(gt_transformed).float()

        sample = {'image': img_tensor, 'labels': gt_tensor}
        sample['image_name'] = self.img_name_list[idx]

        if self.add_is_channels and (not self.pre_mask):
            sample['is_channel'] = is_channels[0:2, :, :]
        else:
            sample['is_channel'] = is_channels

        if self.model_name == "cdnet":  
            sample["fdm_is_channels"] = fdm_is_channels  
                    
        if self.add_edge:
            # cv2.imwrite("./ldc_crop/"+sample['image_name']+".png", edge*255)
            edge_tensor =  torch.from_numpy(np.transpose(edge, axes=(2,0,1))).float()
            sample["ldcedge"] = edge_tensor

        return sample


def get_occ_dataloader(cfg):
    # torch.manual_seed(cfg.seed)
    # train_size = int(len(occ_dataset) * 0.7)
    # test_size = len(occ_dataset) - train_size
    # train_dataset, val_dataset = torch.utils.data.random_split(occ_dataset, [train_size, test_size])

    if not cfg.inference:
        if cfg.train_transform_A:
            # use A transform
            A_transform = get_A_transform(cfg.random_crop_size)
            # function
            transform_pipeline = transform_with_A
            train_dataset = OccDataset(cfg.dataset_dir, cfg.train_img_txt, cfg.dataset_name,
 train_transform=transform_pipeline, A_transform=A_transform, additional_train_trans=cfg.additional_train_trans, cfg=cfg)

        elif cfg.train_transform:
            # random crop with label image
            # function
            transform_pipeline = crop_with_label
            train_dataset = OccDataset(cfg.dataset_dir, cfg.train_img_txt, cfg.dataset_name,
 train_transform=transform_pipeline,cfg=cfg)

        train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                                      shuffle=True, num_workers=cfg.num_workers, drop_last=True)
        val_dataset = OccDataset(cfg.val_dataset_dir, cfg.val_img_txt, cfg.val_dataset_name,
                                 train_transform=None, cfg=cfg)
    else:
        train_dataloader = None
        val_dataset = OccDataset(cfg.val_dataset_dir, cfg.val_img_txt, cfg.val_dataset_name,
                                  train_transform=None, cfg=cfg)

    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

    if cfg.cuda is not None:
        return {'train': train_dataloader, 'test': val_dataloader}
        # return {'train': PreFetcher(train_dataloader), 'test': PreFetcher(val_dataloader)}
    else:
        return {'train': train_dataloader, 'test': val_dataloader}




    

if __name__ == '__main__':
    # print(os.getcwd())

    # use A transform
    A_transform = get_A_transform(1080)
    # function
    transform_pipeline = transform_with_A
    occ_dataset = OccDataset("../occdata/synocc/", "./datasets/occ_split/50_occ_train.txt", "occ",
                             crop_size=1080, train_transform=transform_pipeline,
                             A_transform=A_transform, additional_train_trans=True)
                             
    # occ_dataset = OccDataset("../occdata/PIOD/", "../occdata/PIOD/train_ids.lst", "piod",
    #                          crop_size=320, train_transform=transform_pipeline,
    #                          A_transform=A_transform, additional_train_trans=True)
    
    # occ_dataset = OccDataset("../occdata/BSDSownership/", "../occdata/BSDSownership/train.lst", "bsds",
    #                          crop_size=320, train_transform=transform_pipeline,
    #                          A_transform=A_transform, additional_train_trans=True)
    
    # occ_dataset = OccDataset("../occdata/NYUv2-OCpp/", "../occdata/NYUv2-OCpp/train.txt", "nyuocpp",
    #                          crop_size=320, train_transform=transform_pipeline,
    #                          A_transform=A_transform, additional_train_trans=True)

    img_tensor, gt_tensor, img_name = occ_dataset[1]
    sample = occ_dataset[1]
    print(img_tensor.shape, gt_tensor.shape)
    print(img_name)

    # a = Image.fromarray(img_crop)
    # a.save('3.jpg')
    # b = np.stack([label_crop[:, :, 0], label_crop[:, :, 0], label_crop[:, :, 0]], axis=2)
    # b = Image.fromarray((b * 255).astype(np.uint8))
    # b.save('4.jpg')
