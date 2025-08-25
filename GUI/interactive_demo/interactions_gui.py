import cv2
import math
import random
import numpy as np
# TypeError: deprecated() got an unexpected keyword argument 'name'
# from isutils.cython import get_dist_maps
# from isutils.geprmap import ob_nmst, mb_nmst
from collections import Counter
from scipy.ndimage import distance_transform_edt

import torch
from torch import nn as nn


class DistMaps(nn.Module):
    def __init__(self, norm_radius, spatial_scale=1.0, cpu_mode=False, use_disks=False):
        super(DistMaps, self).__init__()
        self.spatial_scale = spatial_scale
        self.norm_radius = norm_radius
        self.cpu_mode = cpu_mode
        self.use_disks = use_disks
        if self.cpu_mode:
            # from isegm.utils.cython import get_dist_maps
            self._get_dist_maps = get_dist_maps

    def get_coord_features(self, points, batchsize, rows, cols):
        if self.cpu_mode:
            coords = []
            for i in range(batchsize):
                norm_delimeter = 1.0 if self.use_disks else self.spatial_scale * self.norm_radius
                coords.append(self._get_dist_maps(points[i].cpu().float().numpy(), rows, cols,
                                                  norm_delimeter))
            coords = torch.from_numpy(np.stack(coords, axis=0)).to(points.device).float()
        else:
            num_points = points.shape[1] // 2
            points = points.view(-1, points.size(2))
            points, points_order = torch.split(points, [2, 1], dim=1)

            invalid_points = torch.max(points, dim=1, keepdim=False)[0] < 0
            row_array = torch.arange(start=0, end=rows, step=1, dtype=torch.float32, device=points.device)
            col_array = torch.arange(start=0, end=cols, step=1, dtype=torch.float32, device=points.device)

            coord_rows, coord_cols = torch.meshgrid(row_array, col_array)
            coords = torch.stack((coord_rows, coord_cols), dim=0).unsqueeze(0).repeat(points.size(0), 1, 1, 1)

            add_xy = (points * self.spatial_scale).view(points.size(0), points.size(1), 1, 1)
            coords.add_(-add_xy)
            if not self.use_disks:
                coords.div_(self.norm_radius * self.spatial_scale)
            coords.mul_(coords)

            coords[:, 0] += coords[:, 1]
            coords = coords[:, :1]

            coords[invalid_points, :, :, :] = 1e6

            coords = coords.view(-1, num_points, 1, rows, cols)
            coords = coords.min(dim=1)[0]  # -> (bs * num_masks * 2) x 1 x h x w
            coords = coords.view(-1, 2, rows, cols)

        if self.use_disks:
            coords = (coords <= (self.norm_radius * self.spatial_scale) ** 2).float()
        else:
            coords.sqrt_().mul_(2).tanh_()

        return coords

    def forward(self, x, coords):
        return self.get_coord_features(coords, x.shape[0], x.shape[2], x.shape[3])


def get_coord_features(points, shape, use_disks=True, spatial_scale=1.0, norm_radius=3):
    rows = shape[1]
    cols = shape[0]

    norm_delimeter = 1.0 if use_disks else spatial_scale * norm_radius
    x = get_dist_maps(points, rows, cols, norm_delimeter)
    print(len(x), x[0].shape)
    print(Counter(x[0].reshape(-1).tolist()))
    coords = [x]
    coords = np.stack(coords, axis=0)
    if use_disks:
        coords = (coords <= (norm_radius * spatial_scale) ** 2)
    else:
        coords.sqrt_().mul_(2).tanh_()
    
    return coords


def get_dist_maps(points, height, width, norm_delimeter):
    dist_maps = np.full((2, height, width), 1e6, dtype=np.float32)

    dxy = [-1, 0, 0, -1, 0, 1, 1, 0]
    q = []
    qhead = 0
    qtail = -1
    # print(points)
    for i in range(len(points)):
        x, y = round(points[i][0]), round(points[i][1])
        if x >= 0:
            qtail += 1
            q.append({
                'row': x,
                'col': y,
                'orig_row': x,
                'orig_col': y,
                'layer': 1 if i >= len(points) / 2 else 0
            })
            dist_maps[q[qtail]['layer'], x, y] = 0

    while qtail - qhead + 1 > 0:
        v = q[qhead]
        qhead += 1

        for k in range(4):
            x = v['row'] + dxy[2 * k]
            y = v['col'] + dxy[2 * k + 1]

            ndist = ((x - v['orig_row']) / norm_delimeter) ** 2 + ((y - v['orig_col']) / norm_delimeter) ** 2
            if (x >= 0 and y >= 0 and x < height and y < width and
                    dist_maps[v['layer'], x, y] > ndist):
                qtail += 1
                q.append({
                    'orig_row': v['orig_row'],
                    'orig_col': v['orig_col'],
                    'layer': v['layer'],
                    'row': x,
                    'col': y
                })
                dist_maps[v['layer'], x, y] = ndist

    return dist_maps

    

def check_direction_change(segment, new_point, angle_threshold=45):
    # 2, 3, 4
    if len(segment) <= 3:
        return 1
    
    # should check twice for the jugged line???
    # segment_p1 = segment[0]
    # segment_p2 = segment[1]
    initial_angle = get_line_angle(segment[0], segment[1])
    second_angle = get_line_angle(segment[0], segment[2])
    new_angle_p = get_line_angle(segment[-2], segment[-1])
    new_angle = get_line_angle(segment[0], new_point)
    if (np.abs(new_angle - initial_angle) >= angle_threshold) \
        and (np.abs(new_angle_p-new_angle) >= angle_threshold):
        return 0
    else:
        return 1


def get_line_angle(point1, point2):
    if point2[1]-point1[1] == 0:
        return -90
    if point2[0]-point1[0] == 0:
        return 0
        
    x = math.atan2(point2[1]-point1[1], point2[0]-point1[0])
    x= x*180/np.pi
    # print(x)
    return x


def get_point_distance(point1, point2):
    distance_points = np.sqrt(np.power((point1[0]-point2[0]), 2) + np.power((point1[1]-point2[1]), 2))
    return distance_points


def get_point_neighbors(pixel_point, img_shape, num_neighbor=4):
    results = list()
    if num_neighbor == 4:
        if (pixel_point[0] - 1 > 0) and (pixel_point[1] - 1 > 0) and \
                (pixel_point[0] + 1 <= img_shape[0] - 1) and (pixel_point[1] + 1 <= img_shape[1] - 1):
            # keep this order
            results.extend([(pixel_point[0], pixel_point[1] + 1), (pixel_point[0], pixel_point[1] - 1),
                            (pixel_point[0] + 1, pixel_point[1]), (pixel_point[0] - 1, pixel_point[1])])
            return results
        # If the condition is not met, the current pixel is the image boundary point
        else:
            return results

    elif num_neighbor == 8:

        if (pixel_point[0] - 1 > 0) and (pixel_point[1] - 1 > 0) and \
                (pixel_point[0] + 1 <= img_shape[0] - 1) and (pixel_point[1] + 1 <= img_shape[1] - 1):

            results.extend([(pixel_point[0], pixel_point[1] + 1), (pixel_point[0], pixel_point[1] - 1),
                            (pixel_point[0] + 1, pixel_point[1]), (pixel_point[0] - 1, pixel_point[1])])
            results.extend([(pixel_point[0] + 1, pixel_point[1] + 1), (pixel_point[0] + 1, pixel_point[1] - 1),
                            (pixel_point[0] - 1, pixel_point[1] + 1), (pixel_point[0] - 1, pixel_point[1] - 1)])
            return results
        # If the condition is not met, the current pixel is the image boundary point
        else:
            return results
    else:
        raise ValueError("num_neighbor should be 4 or 8!")


def get_corner_neighbors(pixel_point, img_shape):
    results = list()
    if (pixel_point[0] - 1 > 0) and (pixel_point[1] - 1 > 0) and \
            (pixel_point[0] + 1 <= img_shape[0] - 1) and (pixel_point[1] + 1 <= img_shape[1] - 1):

        results.extend([(pixel_point[0] + 1, pixel_point[1] + 1), (pixel_point[0] + 1, pixel_point[1] - 1),
                        (pixel_point[0] - 1, pixel_point[1] + 1), (pixel_point[0] - 1, pixel_point[1] - 1)])
        return results
    # If the condition is not met, the current pixel is the image boundary point
    else:
        return results


def naive_check(ppoint, check_point_rgb, c_segment, checked,  pr_map):
    # c_segment = [ppoint]
    # RecursionError: maximum recursion depth exceeded in comparison
    # or we can set to another number just reduce edge frament pixel
    if len(c_segment) >= 100:
        return c_segment
    # first check 4 direct neighbors:
    cp_n4 = get_point_neighbors(ppoint, pr_map.shape, 4)
    direct_n_count = 0
    for cp_n4_item in cp_n4:
        # normally just one new in 4 direct neighbors
        if pr_map[cp_n4_item[0], cp_n4_item[1], :].tolist() == check_point_rgb and (len(checked[(cp_n4_item[0], cp_n4_item[1])]) == 0):
            # if check_direction_change(c_segment, cp_n4_item, angle_threshold=60) == 1:
            # checked.append(cp_n4_item)
            checked[(cp_n4_item[0], cp_n4_item[1])] = [1]
            # print(cp_n4_item)
            current_check = cp_n4_item
            direct_n_count += 1
    # print(direct_n_count)
    if direct_n_count >= 4:
        # not clear area
        return c_segment
    elif 3 >= direct_n_count >= 1:
        # checked.append(current_check)
        c_segment.append(current_check)
        # recursion
        c_segment = naive_check(current_check, check_point_rgb, c_segment, checked,  pr_map)
        return c_segment
    else:
        # then check 4 corners
        cp_c4 = get_corner_neighbors(ppoint, pr_map.shape)
        corner_n_count = 0
        # print("here")
        # print(cp_c4)
        for cp_c4_item in cp_c4:
            # normally just one new in 4 corner neighbors
            if pr_map[cp_c4_item[0], cp_c4_item[1], :].tolist() == check_point_rgb and (len(checked[(cp_c4_item[0], cp_c4_item[1])]) == 0):
                # if check_direction_change(c_segment, cp_c4_item, angle_threshold=60) == 1:
                # print(cp_c4_item)
                # checked.append(cp_c4_item)
                checked[(cp_c4_item[0], cp_c4_item[1])] = [1]
                current_check = cp_c4_item
                corner_n_count += 1
        # print(corner_n_count)
        if corner_n_count > 1:
            # not clear area
            return c_segment
        elif corner_n_count == 1:
            # checked.append(current_check)
            c_segment.append(current_check)
            # recursion
            c_segment = naive_check(current_check, check_point_rgb, c_segment, checked,  pr_map)
            return c_segment
        else:
            # print(c_segment)
            return c_segment


def split_edge_frament(pr_map, candidate_len=15, check_point_rgb=[[255, 0, 0], [0, 0, 255]]):
    h_size, w_size, channel = pr_map.shape
    dict_keys = [(i, j) for i in range(h_size) for j in range(w_size)]
    prmap_checked_dict = dict.fromkeys(dict_keys, [])    
    checked = list()
    ef_list = list()
    p_candidate_ef = list()
    p_candidate_ef_points = list()    
    n_candidate_ef = list()
    n_candidate_ef_points = list()    
    for i in range(pr_map.shape[0]):
        for j in range(pr_map.shape[1]):
            c_pixel = (i, j)
            if pr_map[i, j, :].tolist() == check_point_rgb[0] and (len(prmap_checked_dict[(i, j)]) == 0):
                # checked.append(c_pixel)
                prmap_checked_dict[(i, j)] = [1]
                # start to check
                # c_segment = [c_pixel]
                c_segment = naive_check(c_pixel, check_point_rgb[0], [c_pixel], prmap_checked_dict, pr_map)
                ef_list.append(c_segment)
                if len(c_segment) >= candidate_len:
                    # print(len(c_segment))
                    n_candidate_ef.append(c_segment)
                    n_candidate_ef_points.extend(c_segment)
                    # checked.extend(c_ef_item)
            if pr_map[i, j, :].tolist() == check_point_rgb[1] and (len(prmap_checked_dict[(i, j)]) == 0):
                # checked.append(c_pixel)
                prmap_checked_dict[(i, j)] = [1]
                # start to check
                # c_segment = [c_pixel]
                c_segment = naive_check(c_pixel, check_point_rgb[1], [c_pixel], prmap_checked_dict, pr_map)
                ef_list.append(c_segment)
                if len(c_segment) >= candidate_len:
                    # print(len(c_segment))
                    p_candidate_ef.append(c_segment)
                    p_candidate_ef_points.extend(c_segment)
                    # checked.extend(c_ef_item)                    

    return p_candidate_ef, n_candidate_ef 


def get_soreted_edgeframent(ef_list):
    ef_sorted = sorted(ef_list, key=lambda x:len(x))
    return ef_sorted


def naive_checkv2(ppoint, check_value, c_segment, checked, fnfp_map, recursion_out=100):
    if len(c_segment) >= recursion_out:
        # Recursion out
        return c_segment

    # first check 4 direct neighbors:
    cp_n4 = get_point_neighbors(ppoint, fnfp_map.shape, 4)
    direct_n_count = 0
    for cp_n4_item in cp_n4:
        if fnfp_map[cp_n4_item[0], cp_n4_item[1]] == check_value and (checked[(cp_n4_item[0], cp_n4_item[1])] == 0):
            # if check_direction_change(c_segment, cp_n4_item, angle_threshold=60) == 1:
            # checked.append(cp_n4_item)
            checked[(cp_n4_item[0], cp_n4_item[1])] = [1]
            current_check = cp_n4_item
            direct_n_count += 1
    if direct_n_count >= 4:
        # not clear area
        return c_segment
    elif 3 >= direct_n_count >= 1:
        # checked.append(current_check)
        c_segment.append(current_check)
        # recursion
        c_segment = naive_checkv2(current_check, check_value, c_segment, checked, fnfp_map)
        return c_segment
    else:
        # then check 4 corners
        cp_c4 = get_corner_neighbors(ppoint, fnfp_map.shape)
        corner_n_count = 0
        for cp_c4_item in cp_c4:
            if fnfp_map[cp_c4_item[0], cp_c4_item[1]] == check_value and (checked[(cp_c4_item[0], cp_c4_item[1])] == 0):
                # if check_direction_change(c_segment, cp_c4_item, angle_threshold=60) == 1:
                # checked.append(cp_c4_item)
                checked[(cp_c4_item[0], cp_c4_item[1])] = [1]
                current_check = cp_c4_item
                corner_n_count += 1
        # print(corner_n_count)
        if corner_n_count > 1:
            # not clear area
            return c_segment
        elif corner_n_count == 1:
            # checked.append(current_check)
            c_segment.append(current_check)
            # recursion
            c_segment = naive_checkv2(current_check, check_value, c_segment, checked, fnfp_map)
            return c_segment
        else:
            return c_segment
    

def get_fnfp_candidates(gt, edge_prob, match_radius=4, candidate_len=30, need_sort=True, recursion_outlen=130, vis_img=False, vis_path=None, vis_point_rgb=[[255, 0, 0], [0, 0, 255]]):

    """
    # gt: shape 2, 0, 1
    # edge_prob : edge prob predicted after NMS and filtered by threhold or canny res
    # match_radius: in intial pretrain could be higher??? should be tested
    
    """

    h_size, w_size = gt.shape
    dict_keys = [(i, j) for i in range(h_size) for j in range(w_size)]
    map_checked_dict = dict.fromkeys(dict_keys, 0)    
    checked = list()
    # ef_list = list()
    p_candidate_ef = list()
    # p_candidate_ef_points = list()    
    n_candidate_ef = list()
    # n_candidate_ef_points = list()   

    img1_bmap = ((np.array(gt[:, :]) > 0)*1).reshape(gt.shape[0], gt.shape[1], 1)
    img2_bmap = ((np.array(edge_prob[:, :]) > 0)*1).reshape(gt.shape[0], gt.shape[1], 1)
    # print(Counter(img1_bmap.reshape(-1)))
    # print(Counter(img2_bmap.reshape(-1)))
    img_output = np.zeros((gt.shape[0], gt.shape[1], 3))
    fnfp_map = np.zeros((gt.shape[0], gt.shape[1]))

    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            if gt[i, j] != 0 or edge_prob[i, j] != 0:
                img1_check = np.sum(img1_bmap[i-match_radius:i+match_radius+1, j-match_radius:j+match_radius+1 ], axis=(0, 1))[0] > 0
                img2_check = np.sum(img2_bmap[i-match_radius:i+match_radius+1, j-match_radius:j+match_radius+1 ], axis=(0, 1))[0] > 0
                if img1_check and img2_check:
                    # img_output[i, j, :] = [0, 255, 0]
                    continue
                elif img1_check and not img2_check:
                    img_output[i, j, :] = vis_point_rgb[1]
                    fnfp_map[i, j] = 1
                elif not img1_check and img2_check:
                    img_output[i, j, :] = vis_point_rgb[0]
                    fnfp_map[i, j] = -1
                else:
                    continue
    if vis_img:
        cv2.imwrite(vis_path, img_output)
    
    for i in range(fnfp_map.shape[0]):
        for j in range(fnfp_map.shape[1]):
            c_pixel = (i, j)
            if fnfp_map[i, j] == -1 and (map_checked_dict[(i, j)] == 0):
                map_checked_dict[(i, j)] = 1
                # start to check
                c_segment = naive_checkv2(c_pixel, -1, [c_pixel], map_checked_dict, fnfp_map, recursion_out=recursion_outlen)
                # ef_list.append(c_segment)
                if len(c_segment) >= candidate_len:
                    n_candidate_ef.append(c_segment)
                    # n_candidate_ef_points.extend(c_segment)
            if fnfp_map[i, j] == 1 and (map_checked_dict[(i, j)] == 0):
                map_checked_dict[(i, j)] = 1
                c_segment = naive_checkv2(c_pixel, 1, [c_pixel], map_checked_dict, fnfp_map, recursion_out=recursion_outlen)
                # ef_list.append(c_segment)
                if len(c_segment) >= candidate_len:
                    p_candidate_ef.append(c_segment)
                    # p_candidate_ef_points.extend(c_segment)
     
    # return img_output  
    if need_sort:
        return get_soreted_edgeframent(p_candidate_ef), get_soreted_edgeframent(n_candidate_ef)
    else:
        return p_candidate_ef, n_candidate_ef  


def get_batch_masks(pred_list, gt_list, p_threshold=0.7, match_radius=4, candidate_len=30, need_sort=True, cand_num=1, encoding="disk", interaction="scribbles", recursion_outlen=130,  random_range=(-1, 1), radius=2, op_way="middle", final_pmask=None, final_nmask=None, use_mbnms=False, matlab_engine=None, return_nmst=False,  **kwargs):
    # map_list = list()
    fp_masks = list()
    fn_masks = list()
    # no_changes = False
    nmst_list = list()
    for i in range(pred_list.shape[0]):
        if use_mbnms:
           # matlab nms threshold could be higher, and candidate_len should be less
            obnmst = mb_nmst(pred_list[i], p_threshold, matlab_engine)
            obnmst = np.array(obnmst)
        else:
            obnmst = ob_nmst(pred_list[i], p_threshold)
        nmst_list.append(obnmst)
        p_candidates, n_candidates = get_fnfp_candidates(gt_list[i], obnmst, match_radius=match_radius, candidate_len=candidate_len, need_sort=need_sort, recursion_outlen=recursion_outlen, **kwargs)
        # map_list.append(c_pr_map)
           
        if len(p_candidates) != 0 and len(p_candidates) >= cand_num:
            p_len_s = len(p_candidates) - cand_num
            cp_mask = generate_interaction_maps(p_candidates[p_len_s:], obnmst.shape, num_points=cand_num, encoding=encoding, interaction=interaction, random_range=random_range, radius=radius, final_mask=final_pmask[i])
            fp_masks.append(cp_mask)
        elif len(p_candidates) != 0 and len(p_candidates) < cand_num:
            pcand_num = len(p_candidates)
            cp_mask = generate_interaction_maps(p_candidates, obnmst.shape, num_points=pcand_num, encoding=encoding, interaction=interaction, random_range=random_range, radius=radius, final_mask=final_pmask[i])
            fp_masks.append(cp_mask)
        else:
            fp_masks.append(final_pmask[i])

        if len(n_candidates) != 0 and len(n_candidates) >= cand_num:
            n_len_s = len(n_candidates) - cand_num
            cn_mask = generate_interaction_maps(n_candidates[n_len_s:], obnmst.shape, num_points=cand_num, encoding=encoding, interaction=interaction, random_range=random_range, radius=radius, final_mask=final_nmask[i])
            fn_masks.append(cn_mask)
        elif len(n_candidates) != 0 and len(n_candidates) < cand_num:   
            ncand_num = len(n_candidates)
            cn_mask = generate_interaction_maps(n_candidates, obnmst.shape, num_points=ncand_num, encoding=encoding, interaction=interaction, random_range=random_range, radius=radius, final_mask=final_nmask[i])   
            fn_masks.append(cn_mask)
        else:
            fn_masks.append(final_nmask[i])
    if return_nmst:
        return np.array(fp_masks)[:, np.newaxis, :, :], np.array(fn_masks)[:, np.newaxis, :, :], np.array(nmst_list)[:, np.newaxis, :, :]
    else:
        return np.array(fp_masks)[:, np.newaxis, :, :], np.array(fn_masks)[:, np.newaxis, :, :], 0


def euclidean_distance_encoding(points, shape):
    #print points.shape[1];
    tmpDist= 255 * np.ones((shape[0],shape[1]))
    [mx,my]= np.meshgrid(np.arange(shape[1]),np.arange(shape[0]))      
    for i in range(len(points)):
        tmpX = mx-points[i][1]
        tmpY = my-points[i][0]
        tmpDist = np.minimum(tmpDist,np.sqrt(np.square(tmpX)+np.square(tmpY)))
    tmpRst = np.array(tmpDist)
    tmpRst[np.where(tmpRst > 255)] = 255
    return tmpRst


# def gaussian_distance_encoding(points, shape):
#     # not working
#     mask=np.zeros((shape[0], shape[1])).astype(np.uint8)
#     mask[points[0][0], points[0][1]] = 1
#
#     max_dist=255
#     points_mask_dist = distance_transform_edt(1-mask)
#     points_mask_dist = np.minimum(points_mask_dist, max_dist)
#     points_mask_dist = points_mask_dist*255
#     return points_mask_dist

# def get_points_mask(size, points):
#     mask=np.zeros(size[::-1]).astype(np.uint8)
#     if len(points)!=0:
#         points=np.array(points)
#         mask[points[:,1], points[:,0]]=1
#     return mask


def slopee(x1,y1,x2,y2):
    x = (y2 - y1) / (x2 - x1)
    return x


def find_points_on_both_sides(x1, y1, slope, distance):
    y = y1
    x = x1 + distance

    x_left = x1 - distance
    y_left = y1 - slope * distance
    x_right = x1 + distance
    y_right = y1 + slope * distance

    return (int(x_left), int(y_left)), (int(x_right), int(y_right))


def scribbles_encoding_synv2(weighted_list, radius, shape, final_mask=None):
    # from skimage import draw
    # assert len(points) % 2 == 0
    # 给定两点，连接并画粗线

    if final_mask is not None:
        s_mask = final_mask
    else:
        s_mask = np.zeros((shape[0], shape[1])).astype(np.uint8)

    for i in range(0, len(weighted_list) - 1, 2):
        p1, p2 = weighted_list[i], weighted_list[i + 1]
        p1c = (p1[1], p1[0])
        p2c = (p2[1], p2[0])
        # s_mask = cv2.line(s_mask, p1c, p2c, [127, 127, 127], radius)
        s_mask = cv2.line(s_mask, p1c, p2c, [255, 255, 255], radius*2)

    for point in weighted_list:
        s_mask = cv2.circle(s_mask, (point[1], point[0]), radius, [255, 255, 255], -1)
    return s_mask


def bresenham_line(x1, y1, x2, y2):
    # Calculate differences
    dx = x2 - x1
    dy = y2 - y1
    
    # Determine signs
    sx = 1 if dx >= 0 else -1
    sy = 1 if dy >= 0 else -1
    
    # Take absolute values
    dx = abs(dx)
    dy = abs(dy)
    
    # Determine traversal direction
    if dx > dy:
        increment_x = True
        p = 2 * dy - dx
    else:
        increment_x = False
        p = 2 * dx - dy
    
    # Initialize starting point
    x = x1
    y = y1
    
    # Store the pixel points
    pixel_points = [(x, y)]
    
    # Iterate over the range from 0 to dx
    for _ in range(dx):
        if increment_x:
            x += sx
        else:
            y += sy
        
        # Update the error term
        if increment_x:
            p += 2 * dy
        else:
            p += 2 * dx
        
        # Check if the error term is greater than 0
        if p >= 0:
            if increment_x:
                y += sy
            else:
                x += sx
            p -= 2 * (dx if increment_x else dy)
        
        # Add the current pixel point to the list
        pixel_points.append((x, y))
    
    return pixel_points


# 通用的Bresenham算法
def GenericBresenhamLine(x1, y1, x2, y2):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    # 根据直线的走势方向，设置变化的单位是正是负
    s1 = 1 if ((x2 - x1) > 0) else -1
    s2 = 1 if ((y2 - y1) > 0) else -1
    # 根据斜率的大小，交换dx和dy，可以理解为变化x轴和y轴使得斜率的绝对值为（0,1）
    boolInterChange = False
    if dy > dx:
        np.swapaxes(dx, dy)
        boolInterChange = True
    # 初始误差
    e = 2 * dy - dx
    x = x1
    y = y1
    pixel_points = [(x, y)]
    for i in range(0, int(dx + 1)):
        if e >= 0:
            # 此时要选择横纵坐标都不同的点，根据斜率的不同，让变化小的一边变化一个单位
            if boolInterChange:
                x += s1
            else:
                y += s2
            e -= 2 * dx
        # 根据斜率的不同，让变化大的方向改变一单位，保证两边的变化小于等于1单位，让直线更加均匀
        if boolInterChange:
            y += s2
        else:
            x += s1
        e += 2 * dy
        pixel_points.append((x, y))
    return pixel_points


def bres(x1, y1, x2, y2):
    x, y = x1, y1
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    gradient = dy/float(dx)

    if gradient > 1:
        dx, dy = dy, dx
        x, y = y, x
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    p = 2*dy - dx
    # print(f"x = {x}, y = {y}")
    # # Initialize the plotting points
    # xcoordinates = [x]
    # ycoordinates = [y]
    pixel_points = [(x, y)]

    for k in range(2, dx + 2):
        if p > 0:
            y = y + 1 if y < y2 else y - 1
            p = p + 2 * (dy - dx)
        else:
            p = p + 2 * dy

        x = x + 1 if x < x2 else x - 1
        pixel_points.append((x, y))

    return pixel_points

def eudis_encoding(points, shape, isdis_radius=20, mode="distance_limit"):
    tmpDist = 255 * np.ones((shape[0], shape[1]))
    [mx, my] = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
   
    if mode == "distance_full":
        for i in range(len(points)):
            tmpX = mx-points[i][1]
            tmpY = my-points[i][0]
            tmpDist = np.minimum(tmpDist, np.sqrt(np.square(tmpX)+np.square(tmpY)))
        tmpDist = 255-tmpDist

    elif mode == "distance_limit":
        b_mask = np.zeros((shape[0], shape[1], 3)) 
        for i in range(len(points)):
            tmpX = mx-points[i][1]
            tmpY = my-points[i][0]
            tmpDist = np.minimum(tmpDist, np.sqrt(np.square(tmpX)+np.square(tmpY)))
            b_mask = cv2.circle(b_mask, (points[i][1], points[i][0]), isdis_radius, [255, 255, 255], -1)

        b_mask_m = b_mask[:, :, 0] / 255
        # # print(Counter(tmpDist.reshape(-1)))
        # print(tmpDist.shape, b_mask_m.shape)
        # cv2.imwrite("./bmask.png", b_mask)
        tmpDist = (255-tmpDist) * b_mask_m

    tmpRst = np.array(tmpDist)
    tmpRst[np.where(tmpRst > 255)] = 255
    return tmpRst


def gaudis_encoding(points, shape, sigma=0.04, isdis_radius=20, mode="distance_limit"):
    """
    Paper: Interactive Boundary Prediction for Object Selection
    :param sigma: 0.02, 0.04, 0.08
    a small σ value provides exact information about the location of selection            小数值边缘数字越小
    a larger σ value tends to encourage the network to learn features at larger scopes    大数值边缘数字越大
    :return: 255 interaction map
    """
    L = min(shape[0], shape[1])
    tmpDist = np.zeros((shape[0], shape[1]))  # * 255
    [mx, my] = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

    if mode == "distance_full":
        for i in range(len(points)):
            tmpX = mx - points[i][1]
            tmpY = my - points[i][0]
            eudis2 = np.square(tmpX) + np.square(tmpY)
            gaudis = np.exp(-(eudis2) / (2 * np.square(sigma*L)))
            tmpDist = np.maximum(tmpDist, gaudis)

    elif mode == "distance_limit":
        b_mask = np.zeros((shape[0], shape[1], 3))
        for i in range(len(points)):
            tmpX = mx - points[i][1]
            tmpY = my - points[i][0]
            eudis2 = np.square(tmpX) + np.square(tmpY)
            gaudis = np.exp(-(eudis2) / (2 * np.square(sigma * L)))  # 越靠近选中点，数值越大(1)
            tmpDist = np.maximum(tmpDist, gaudis)  # 在所有距离中选择最大的
            b_mask = cv2.circle(b_mask, (points[i][1], points[i][0]), isdis_radius, [255, 255, 255], -1)

        b_mask_m = b_mask[:, :, 0] / 255
        tmpDist = (tmpDist) * b_mask_m

    tmpRst = np.array(tmpDist)
    tmpRst[np.where(tmpRst > 255)] = 255
    return tmpRst * 255



def syn_scribbles_bboxdistance_encoding(synscri_segment, weighted_list):
    start_point = scri_segment[0]
    end_point = scri_segment[-1]
    # one bbox, in the syn line area, nearest distance
    # at the side of bbox, calculate the nearest distance to the start/end point
    # need to be done?????
    return



def bbox_encoding(points, radius, shape, fill_area=False, final_mask=None):
    # from skimage import draw
    assert len(points) % 2 == 0
        
    if final_mask is not None:
        s_mask = final_mask
    else:
        s_mask = np.zeros((shape[0], shape[1])).astype(np.uint8)
    
    areas = list()
    for i in range(0, len(points), 2):
        point1 = points[i]
        point2 = points[i+1]
        if point1[0] == point2[0]:
            area = np.array([[point1[1], point1[0]+radius], [point1[1], point1[0]-radius],
                             [point2[1], point2[0]+radius], [point2[1], point2[0]-radius] ])
            if fill_area:
                s_mask = cv2.rectangle(s_mask, [point1[1], point1[0]+radius], [point2[1], point2[0]-radius], [255, 255, 255], -1)
            else:
                s_mask = cv2.rectangle(s_mask, [point1[1], point1[0]+radius], [point2[1], point2[0]-radius], [255, 255, 255], 1)
            # s_mask = cv2.fillConvexPoly(s_mask, area, [255, 255, 255])

            areas.append(area)
        elif point1[1] == point2[1]:
            area = np.array([[point1[1]+radius, point1[0]], [point1[1]-radius, point1[0]],
                             [point2[1]+radius, point2[0]], [point2[1]-radius, point2[0]]])
            areas.append(area)

            if fill_area:
                s_mask = cv2.rectangle(s_mask, [point1[1]+radius, point1[0]], [point2[1]-radius, point2[0]], [255, 255, 255], -1)
            else:
                s_mask = cv2.rectangle(s_mask, [point1[1]+radius, point1[0]], [point2[1]-radius, point2[0]], [255, 255, 255], 1)
            
        else:
            #
            c_slopee = slopee(point1[1], point1[0], point2[1], point2[0])
            p1, p2 = find_points_on_both_sides(point1[1], point1[0], -1/c_slopee, radius)
            p3, p4 = find_points_on_both_sides(point2[1], point2[0], -1/c_slopee, radius)
            areas.append(np.array([p1, p2, p3, p4]))
            s_mask = cv2.rectangle(s_mask, p1, p4, [255, 255, 255], 1)
            if fill_area:
                s_mask = cv2.rectangle(s_mask, p1, p4, [255, 255, 255], -1)
            else:
                s_mask = cv2.rectangle(s_mask, p1, p4, [255, 255, 255], 1)

    # s_mask = cv2.fillPoly(s_mask, areas, (255, 255, 255))
    return s_mask
    

def scribbles_encoding_v2(points, radius, shape, final_mask=None):
    # from skimage import draw
    # assert len(points) % 2 == 0
        
    if final_mask is not None:
        s_mask = final_mask
    else:
        s_mask = np.zeros((shape[0], shape[1])).astype(np.uint8)
    
    for point in points:
        s_mask = cv2.circle(s_mask, (point[1], point[0]), radius, [255, 255, 255], -1)
        
    return s_mask
        
        
def scribbles_encoding_syn(points, radius, shape, weighted_list, final_mask=None):
    # from skimage import draw
    # assert len(points) % 2 == 0
        
    if final_mask is not None:
        s_mask = final_mask
    else:
        s_mask = np.zeros((shape[0], shape[1])).astype(np.uint8)
    
    for point in points:
        s_mask = cv2.circle(s_mask, (point[1], point[0]), radius, [122.5, 122.5, 122.5], -1)
    for point in  weighted_list:
        s_mask = cv2.circle(s_mask, (point[1], point[0]), radius, [255, 255, 255], -1)
    return s_mask
    
    
def disk_encoding(points, radius, shape, final_mask=None):
    # norm_delimeter = 1.0 # nrom_radius * spatial_scale
    # disk_map = get_dist_maps(points, shape[0], shape[1], 1)
    # disk_map_test = DistMaps(norm_radius=5, spatial_scale=1.0, cpu_mode=True, use_disks=True)
    # disk_map = get_coord_features(points, shape)
    
    if final_mask is not None:
        disk_map = final_mask
    else:
        disk_map = np.zeros((shape[0], shape[1])).astype(np.uint8)
    # print(disk_map.shape, type(disk_map))
    print("points:", points)
    for point in points:  #(x, y)
        print('point', point)
        disk_map = cv2.circle(disk_map, (point[1], point[0]), radius, [255, 255, 255], -1)
    return disk_map


def generate_middle_point(inter_segment, random_range=(-1, 1)):
    mid_index = int(len(inter_segment)/2)
    random_index = random.randint(random_range[0], random_range[1])

    initial_center_circle = mid_index + random_index
    icc_x = inter_segment[initial_center_circle][0]
    icc_y = inter_segment[initial_center_circle][1]

    random_change_x = random.randint(-1, 1)
    random_change_y = random.randint(-1, 1)
    final_center_circle = (icc_x+random_change_x, icc_y+random_change_y)
    # encoding_map = disk_encoding([final_center_circle], radius, pr_map_shape)
    # encoding_map = euclidean_distance_encoding([final_center_circle], pr_map_shape)
    # print(Counter(encoding_map.reshape(-1)))
    # print(icc_x, icc_y, final_center_circle)
    return final_center_circle


def generate_random_point(inter_segment):
    random_index = random.randint(0, len(inter_segment)-1)
    icc_x = inter_segment[random_index][0]
    icc_y = inter_segment[random_index][1]
    random_change_x = random.randint(-1, 1)
    random_change_y = random.randint(-1, 1)
    final_center_circle = (icc_x+random_change_x, icc_y+random_change_y)
    # encoding_map = disk_encoding([final_center_circle], radius, pr_map_shape)
    return final_center_circle


# or check other connected candidate, or record direction change positions
def generate_multiple_points(inter_segment, random_range=(-2, 2), points_num=3):
    points_list = generate_two_points(inter_segment, random_range=random_range)
    middle_point = generate_middle_point(inter_segment, random_range=random_range)
    points_list.append(middle_point)
    # too many points are useless unless direction changes
    return points_list
    

def generate_two_points(inter_segment, random_range=(-2, 2)):
    start_index = 0
    # for now don't let it exceed the segment range
    random_index = random.randint(0, random_range[1])
    
    initial_center_circle = start_index+ random_index
    icc_x = inter_segment[initial_center_circle][0]
    icc_y = inter_segment[initial_center_circle][1]

    random_change_x = random.randint(-1, 1)
    random_change_y = random.randint(-1, 1)
    final_center_circle = (icc_x+random_change_x, icc_y+random_change_y)

    end_index = len(inter_segment) - 1
    # for now don't let it exceed the segment range
    random_index = random.randint(random_range[0], 0)
    
    initial_center_circle_e = end_index + random_index
    icc_x_e = inter_segment[initial_center_circle_e][0]
    icc_y_e = inter_segment[initial_center_circle_e][1]
    final_center_circle_e = (icc_x_e+random_change_x, icc_y_e+random_change_y)

    return [final_center_circle, final_center_circle_e]

    
def generate_scribbles(inter_segment, random_range=(-2, 2), keep_random_change=True):
    points_return = list()
    
    start_index = 0
    # for now don't let it exceed the segment range
    random_index = random.randint(0, random_range[1])
    
    initial_center_circle = start_index+ random_index
    icc_x = inter_segment[initial_center_circle][0]
    icc_y = inter_segment[initial_center_circle][1]

    random_change_x = random.randint(-1, 1)
    random_change_y = random.randint(-1, 1)
    # final_center_circle = (icc_x+random_change_x, icc_y+random_change_y)

    end_index = len(inter_segment) - 1
    # for now don't let it exceed the segment range
    random_index = random.randint(random_range[0], 0)
    initial_center_circle_e = end_index + random_index
    
    for i in range(initial_center_circle, initial_center_circle_e+1):
        if keep_random_change:
            c_point = (inter_segment[i][0]+random_change_x, inter_segment[i][1]+random_change_y)
        else:
            random_change_x = random.randint(-1, 1)
            random_change_y = random.randint(-1, 1)
            c_point = (inter_segment[i][0]+random_change_x, inter_segment[i][1]+random_change_y)
        points_return.append(c_point)
    
    # icc_x_e = inter_segment[initial_center_circle_e][0]
    # icc_y_e = inter_segment[initial_center_circle_e][1]
    # final_center_circle_e = (icc_x_e+random_change_x, icc_y_e+random_change_y)

    return points_return    
    
    
def generate_syn_scribbles(inter_segment, random_range=(-2, 2)):
    points_return = list()
    
    p1, p2 = generate_two_points(inter_segment, random_range)
    x1, y1 = p1
    x2, y2 = p2
    points_return = bresenham_line(x1, y1, x2, y2)  # (x1, y1, x2, y2)  bresenham_line  GenericBresenhamLine

    return points_return, [p1, p2]


# sometimes dont't have little.....
# check my IS results if have little or not....
def generate_nmst_scribbles(inter_segment, epmap, random_range=(-2, 2), threshold=0.5):
    dict_keys = [(i, j) for i in range(epmap.shape[0]) for j in range(epmap.shape[1])]
    map_checked_dict = dict.fromkeys(dict_keys, 0)  
    segments_list = list()

    p1, p2 = generate_two_points(inter_segment, random_range)
    nmst_map = ob_nmst(epmap, threshold)
    nmst_bmap = ((np.array(nmst_map[:, :]) > 0)*1).reshape(nmst_map.shape[0], nmst_map.shape[1], 1)
    c_segment = [p1, p2]


    while len(c_segment) != 1:
        # start from p1 and p2, find main possible frament.
        c_segment = check_2points_path(p1, p2, [p1], map_checked_dict, nmst_bmap, recursion_out=100)
        segments_list.append(c_segment)
    return get_soreted_edgeframent(segments_list)


def check_2points_path(ppoint, ppoint2, c_segment, checked, fnfp_map, recursion_out=250):
    # if len(c_segment) >= recursion_out or (c_segment[-1] == ppoint2) :
    #     # Recursion out
    #     return c_segment

    if c_segment[-1] == ppoint2:
        # Recursion out
        return c_segment
    
    check_value = 1
    # first check 4 direct neighbors:
    cp_n4 = get_point_neighbors(ppoint, fnfp_map.shape, 4)
    direct_n_count = 0
    for cp_n4_item in cp_n4:
        if fnfp_map[cp_n4_item[0], cp_n4_item[1]] == check_value and (checked[(cp_n4_item[0], cp_n4_item[1])] == 0):
            if cp_n4_item[0] == ppoint2[0] and (cp_n4_item[1] == ppoint2[1]):
                c_segment.append(cp_n4_item)
                return c_segment
            checked[(cp_n4_item[0], cp_n4_item[1])] = [1]
            current_check = cp_n4_item
            direct_n_count += 1
    if direct_n_count >= 4:
        # not clear area
        return c_segment
    elif 3 >= direct_n_count >= 1:
        # checked.append(current_check)
        c_segment.append(current_check)
        # recursion
        c_segment = check_2points_path(current_check, ppoint2, c_segment, checked, fnfp_map)
        return c_segment
    else:
        # then check 4 corners
        cp_c4 = get_corner_neighbors(ppoint, fnfp_map.shape)
        corner_n_count = 0
        for cp_c4_item in cp_c4:
            if fnfp_map[cp_c4_item[0], cp_c4_item[1]] == check_value and (checked[(cp_c4_item[0], cp_c4_item[1])] == 0):
                if cp_c4_item[0] == ppoint2[0] and (cp_c4_item[1] == ppoint2[1]):
                    c_segment.append(cp_c4_item)
                    return c_segment
                    checked[(cp_c4_item[0], cp_c4_item[1])] = [1]
                    current_check = cp_c4_item
                    corner_n_count += 1
        # print(corner_n_count)
        if corner_n_count > 1:
            # not clear area
            return c_segment
        elif corner_n_count == 1:
            # checked.append(current_check)
            c_segment.append(current_check)
            # recursion
            c_segment = check_2points_path(current_check, ppoint2, c_segment, checked, fnfp_map)
            return c_segment
        else:
            return c_segment


def generate_interaction_maps(segments, prmap_shape, num_points=3, encoding="disk", interaction="scribbles",
                              random_range=(0, 0), radius=3, op_way="middle", gausigma=0.04, final_mask=None, save_log=None):
    # print('segments:', segments)
    # assert len(segments) == num_points  # [[(x,y), (x,y)]]
    # print('[generate_interaction_maps] segments:', np.array(segments).shape, segments)
    # (a, x, 2) a表示交互的个数，x表示某次交互的坐标点数，2表示横纵坐标

    """
    1
    support pair (interaction, encoding): 
    (point/points, disk/eudistance), (bbox, bbox), (scribbles, scribbles/eudistance), (syn_scribbles, syn_scribbles/syn_scribbles_dis)
    2
    point + middle, disk/eudistance/gaudistance
    scribbles, scribbles/eudistance/gaudistance
    syn_scribbles（用户点2点，自动直线连接之）, syn_scribbles(选中的点为1，弱连接的点为0.5)/syn_scribbles_eudis（欧氏距离）/syn_scribbles_gaudis
    3
    not support yet: (nmst_syn_scribbles, syn_scribbles/syn_scribbles_dis)
    """
    

    if save_log is not None:
        print("Record the click info at ", save_log)
        print()
        with open(save_log, "a+") as record:
            record.write(interaction + "\t" + encoding + "\t" + str(segments))
            record.write("\n")

    # get points
    # points_iter = list()
    # p_weight = list()
    # for inter_seg in segments:
    #     if interaction == "two_points" or interaction == "bbox":
    #         points_iter.extend(generate_two_points(inter_seg, random_range))
    #     elif interaction == "multi_points":
    #         points_iter.extend(generate_multiple_points(inter_seg, random_range))
    #     elif interaction == "point":
    #         if op_way == "random":
    #             points_iter.append(generate_random_point(inter_seg))
    #         elif op_way == "middle":
    #             points_iter = segments #[(x, y)]
    #             # points_iter.append(generate_middle_point(inter_seg, random_range))
    #     elif interaction == "scribbles":
    #         points_iter.extend(generate_scribbles(inter_seg, random_range=random_range, keep_random_change=True))
    #         # points_iter.extend(generate_scribbles(inter_seg, random_range=(-2, 2), keep_random_change=False))
    #     elif interaction == "syn_scribbles":
    #         # print('inter_seg', inter_seg)
    #         p_weight.extend(inter_seg)  # generate_two_points(inter_seg, random_range)
    #         a, b = generate_syn_scribbles(inter_seg, random_range)
    #         points_iter.extend(a)
    #         # p_weight.extend(b)
    #     elif interaction == "nmst_syn_scribbles":
    #         print("Not support yet")
    #         # a = generate_nmst_scribbles(inter_segment, epmap, random_range=(-2, 2), threshold=0.5)
    #     else:
    #         raise ValueError("Not support yet")

        
    # generate mask
    if encoding == "disk":
        points_iter = segments[0]
        final_mask = disk_encoding(points_iter, radius, prmap_shape, final_mask)
    elif encoding == "scribbles":
        points_iter = segments[0]
        final_mask = scribbles_encoding_v2(points_iter, radius, prmap_shape, final_mask)
    elif encoding == "eudistance":
        points_iter = segments[0]
        final_mask = eudis_encoding(points_iter, prmap_shape, isdis_radius=5, mode="distance_limit") # full
        # final_mask = disk_encoding(points_iter, radius, prmap_shape, final_mask=final_mask)
    elif encoding == "gaudistance":
        points_iter = segments[0]
        final_mask = gaudis_encoding(points_iter, prmap_shape, sigma=gausigma, isdis_radius=5, mode="distance_limit")
    elif encoding == "syn_scribbles":
        p_weight = segments[0]
        # final_mask = scribbles_encoding_syn(points_iter, radius, prmap_shape, p_weight, final_mask=final_mask)
        final_mask = scribbles_encoding_synv2(p_weight, radius, prmap_shape, final_mask=final_mask)
        # final_mask = cv2.line(final_mask, points_iter, radius, prmap_shape, p_weight, final_mask=final_mask)

    elif encoding == "syn_scribbles_eudis":
        # problem, need to fix
        points_iter = segments[0]
        final_mask = eudis_encoding(points_iter, prmap_shape, isdis_radius=5, mode="distance_limit") # full
        # final_mask = scribbles_encoding_synv2(p_weight, radius, prmap_shape, final_mask=final_mask)
        # final_mask = disk_encoding(points_iter/p_weight, radius, prmap_shape, final_mask=final_mask)
    elif encoding == "syn_scribbles_gaudis":
        # problem, need to fix
        points_iter = segments[0]
        final_mask = gaudis_encoding(points_iter, prmap_shape, sigma=gausigma, isdis_radius=5, mode="distance_limit")
        # final_mask = scribbles_encoding_synv2(p_weight, radius, prmap_shape, final_mask=final_mask)
    else:
        raise ValueError

    return final_mask



def generate_random_inneg_maps(negpoints, prmap_shape, num_points=1, encoding="disk", radius=3):
    # assert len(negpoints) == num_points
    points_iter = negpoints
    # generate mask
    if encoding == "disk":
        final_mask = disk_encoding(points_iter, radius, prmap_shape)
    elif encoding == "scribbles":
        final_mask = scribbles_encoding_v2(points_iter, radius, prmap_shape)
    else:
        raise ValueError
    return final_mask


# def check_points_distance(p_choices, new_points)
def get_random_sample_click(binary_mask, radius, min_distance=30):
    # binary_mask = ((np.array(gt_map[:, :, 0]) > 0 ) * 1).reshape(1080, 1080, 1)
    radius = radius + 1
    negative_choices = list()
    for i in range(radius, binary_mask.shape[0]-radius, radius*2):
        for j in range(radius, binary_mask.shape[1]-radius, radius*2):
            area_count = None
            xmin = i - radius 
            xmax = i + radius  +1
            ymin = j - radius  
            ymax = j +radius +1 
            area_count = np.sum(binary_mask[xmin:xmax, ymin:ymax, :], axis=(0, 1))[0] 
            if area_count == 0: #  and (len(negtive_choices) == 0 or ):
                negative_choices.append((i, j))
          
    return negative_choices


def vis_fnfp_points(vis_img, n_candidate_ef_points, p_candidate_ef_points, store_path):
    # # print(n_candidate_ef)
    # n_vis_points = pr_map.copy()    
    # p_vis_points = pr_map.copy()  
    p_points = list()
    n_points = list()
    for item in n_candidate_ef_points:
        n_points.extend(item)
    for item in p_candidate_ef_points:
        p_points.extend(item)
    for i in range(vis_img.shape[0]):
        for j in range(vis_img.shape[1]):
            c_pixel = (i, j)
            if c_pixel in n_points:
                vis_img[i, j, :] = [255, 0, 0]
            if c_pixel in p_points:
                vis_img[i, j, :] = [0, 0, 255]                
    cv2.imwrite(store_path+"_points_fnfp.png", vis_img)


def vis_np_mask(vis_img, p_mask, n_mask, store_path):
    p_mask = p_mask * [1, 1, 1]
    n_mask = n_mask * [1, 1, 1]
    for i in range(vis_img.shape[0]):
        for j in range(vis_img.shape[1]):
            if p_mask[i, j, :].tolist() != [0, 0, 0]:
                vis_img[i, j, :] = [0, 0, 255]
            if n_mask[i, j, :].tolist() != [0, 0, 0]:
                vis_img[i, j, :] = [255, 0, 0]
    cv2.imwrite(store_path+"_fnfp.png", vis_img)
    # cv2.imwrite("./crop_examples/"+img_name+"_gt_crop.png", gt_crop*255)
    # cv2.imwrite(store_path+"_p_mask.png", p_mask)
    # cv2.imwrite(store_path+"_n_mask.png", n_mask)
    
    
def vis_inter(vis_img, p_mask, n_mask, store_path):
    mean = [0.485, 0.456, 0.406]  # [0.780, 0.771, 0.771]
    std = [0.229, 0.224, 0.225]  # [0.203, 0.206, 0.217]
    vis_img = (vis_img * std + mean) * 255
    p_mask = p_mask * [1, 1, 1]
    n_mask = n_mask * [1, 1, 1]
    for i in range(vis_img.shape[0]):
        for j in range(vis_img.shape[1]):
            if p_mask[i, j, :].tolist() != [0, 0, 0]:
                vis_img[i, j, :] = [0, 0, 255]
            if n_mask[i, j, :].tolist() != [0, 0, 0]:
                vis_img[i, j, :] = [255, 0, 0]
    cv2.imwrite(store_path+"_ifnfp.png", vis_img)  


if __name__ == '__main__':
    pr_map = cv2.imread("./mexample.png") # ("./bpynmst7_pr_gt.png")  # ("./mexample.png")
    p_candidate_ef, n_candidate_ef = split_edge_frament(pr_map)
    p_ef_s = get_soreted_edgeframent(p_candidate_ef)
    # print(len(p_ef_s[0]), len(p_ef_s[-1]))
    p_mask = generate_middle_point(p_ef_s[-1], pr_map.shape)
    cv2.imwrite("m_p_i3.png", p_mask)
    

