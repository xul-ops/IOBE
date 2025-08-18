import os
import cv2
import copy
import math
import time
import random
import logging
from decimal import Decimal
# import open3d as o3d
import numpy as np
from skimage import draw
# import scipy.linalg as linalg
import itertools
from collections import Counter
from collections import defaultdict
from tqdm import tqdm
# from tqdm.std import trange
import networkx as nx
from networkx.drawing.nx_pydot import write_dot
from scipy.spatial.transform import Rotation as R

import utils.configs as configs
from utils.tools import *
from downsampling import *


OBP_CLASS = ["background_obj_obp", "different_obj_obp", "self_obp"]
# logging config, if run several, change log name 
if not os.path.exists(configs.image_save_dir):
    os.makedirs(configs.image_save_dir)
logging.basicConfig(filename=configs.image_save_dir + "/program_geocc.log",
                    filemode='w+', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

if configs.use_down_sampling:
    configs.img_resolution_x *= configs.down_sampling_rate
    configs.img_resolution_y *= configs.down_sampling_rate
    print()   
    print(f"Towards better matched boundary, we will use the blender cycle oversampling, so generated image size is {configs.img_resolution_x}*{configs.img_resolution_y}, but this ops will loss some OB3 and OO infomation!")


def projection_skiimage(model_pose_infos, camera_matrix, model_dir, img_file, save_file):

    img = cv2.imread(img_file)
    img_idmap = img.copy()
    h_size, w_size, channel = img.shape
    
    assert configs.img_resolution_x == h_size
    assert configs.img_resolution_y == w_size

    dict_keys = [(i, j) for i in range(h_size) for j in range(w_size)]
    points_info_dict = dict.fromkeys(dict_keys, [])
    no_projection_mesh = list()
    f_covered_points = list()
    mesh_3d_dict = dict()
    vertex_map_dict = dict()
    obj_count = 1
    # polypoints = list()

    for index, model_pose_info in enumerate(model_pose_infos):
        model_id = model_pose_info['shape_id']
        trans_vec = np.array(model_pose_info['translation'])
        rot_mat = np.array(model_pose_info['rotation'])
        if model_pose_info['shape_id'] is None or model_pose_info['rotation'] is None \
                or model_pose_info['translation'] is None or model_pose_info['category_id'] is None:
            continue
        if model_pose_info['category_id'] in configs.category_must_skip:
            continue

        current_obj_dir = os.path.join(model_dir, model_id + "/")
        obj_file = os.path.join(current_obj_dir + 'raw_model.obj')

        # transformation w->c
        mesh_vertices, mesh_surfaces = get_obj_vertex_ali(obj_file)
        # mesh_vertices, mesh_surfaces = get_obj_vertex_ali(obj_file)
        mesh_vertices.astype(np.float64)
        rot_mat = rot_mat.astype(np.float64)
        trans_vec = trans_vec.astype(np.float64)
        
        if len(mesh_surfaces.shape) != 2:

            # print(mesh_surfaces.shape)
            count_list = list()
            is_print_info = False
            new_mesh_surfaces = list()
            for item_mesh in mesh_surfaces:
                count_list.append(len(item_mesh))
                if len(item_mesh) == 4:
                    new_mesh_surfaces.append(item_mesh[:3])
                    new_mesh_surfaces.append(np.array([item_mesh[2], item_mesh[3], item_mesh[0]]))
                elif len(item_mesh) == 3:
                    new_mesh_surfaces.append(item_mesh)
                else:
                    # try to use open3d sub mesh method
                    is_print_info = True
                    # raise ValueError
                    # actually only one element arrive here, e73ff703-adb1-4d3a-993d-60f6d148bec4  --- {3: 22761, 1: 1}
                    print("Need new sub method for mesh to get triangles!")
            if is_print_info:
                print(model_pose_info)
                print(dict(Counter(count_list)))
            mesh_surfaces = np.array(new_mesh_surfaces)


        if configs.extra_rotation:
            r = R.from_euler(configs.extra_rotation_mode, configs.extra_rotation_angle, degrees=True)
            extra_rot = r.as_matrix()
            extra_rot = np.array(extra_rot).astype(np.float64)
            mesh_vertices_trans = np.transpose(np.dot(extra_rot, np.transpose(mesh_vertices)))
        else:
            mesh_vertices_trans = np.transpose(np.dot(rot_mat, np.transpose(mesh_vertices)))

        mesh_vertices_trans = mesh_vertices_trans - (-trans_vec)
        mesh_vertices_trans = np.apply_along_axis(add_1_in_position, -1, mesh_vertices_trans)
        mesh_vertices_trans = np.apply_along_axis(transfer_wc, -1, mesh_vertices_trans, camera_matrix)

        screen_vertices = np.array([viewport(ndc(v)) for v in mesh_vertices_trans])
        h, w, d = screen_vertices.T


        for item in mesh_surfaces:
            point_c_3d = list()
            current_surface = list()
            current_surface_depth = list()
            current_s_with_depth = list()
            for j in range(len(item)):

                # obj vertices index start from 1, o3d index start from 0
                # current_x, current_y = np.round(w[item[j]]).astype(int), np.round(h[item[j]]).astype(int)
                # current_x, current_y = w[item[j]], h[item[j]]
                # current_surface.append([current_x, current_y])
                # point_w_3d.append(mesh_vertices_trans[item[j]])

                # current_x, current_y = np.round(w[int(item[j])-1]).astype(int), np.round(h[int(item[j])-1]).astype(int)
                current_x, current_y = w[int(item[j]) - 1], h[int(item[j]) - 1]
                current_surface.append([current_y, configs.img_resolution_y - current_x])
                current_surface_depth.append(d[int(item[j]) - 1])
                point_c_3d.append(mesh_vertices_trans[int(item[j]) - 1][0:3])
                current_s_with_depth.append((current_y, configs.img_resolution_y - current_x, d[int(item[j]) - 1]))

                # vertex_map_dict[mesh_vertices_trans[int(item[j]) - 1][0:3]] = [current_y, configs.img_resolution_y - current_x]
                vertex_map_dict[(current_y, configs.img_resolution_y - current_x)] = mesh_vertices_trans[int(item[j]) - 1][0:3]

            

            point_3d_list_tuple = item_to_tuple(point_c_3d, is_polygon=False)

            if calculate_polygon_area(current_surface, point_c_3d):
            
                # cv2.fillConvexPoly(img, np.array(current_surface), color=(255, 255, 255))
                # cv2.fillPoly(img, [np.array(current_surface)], color=(255, 255, 255))

                polygon = np.asarray(current_surface)
                vertex_row_coords, vertex_col_coords = polygon.T
                fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, (w_size, h_size))

                # If modify the skiimage source code, the calculation process will be faster
                if len(fill_row_coords) == 0 or len(fill_col_coords) == 0:
                    # polygons_tuple = item_to_tuple(current_surface, is_polygon=True)
                    # no_projection_mesh.append((polygons_tuple, point_3d_list_tuple, obj_count))
                    no_projection_mesh.append((current_surface, point_c_3d, obj_count))
                    mesh_3d_dict[point_3d_list_tuple] = ["small_face", [],  current_surface, current_surface_depth, obj_count]
                    continue                                   

                # polygon2mask_points = [[fill_col_coords[key_key], fill_row_coords[key_key]]
                # polypoints.extend(points)
                # one_point = np.array([fill_col_coords[0], fill_row_coords[0]]).astype(np.float64)
                # one_point = np.array([fill_row_coords[0], fill_col_coords[0]]).astype(np.float64)
                # current_bcoor = barycentric_coordinates(one_point, polygon[0], polygon[1], polygon[2])
                # print(current_bcoor)

                current_hwpoint = list()

                for point_index in range(len(fill_col_coords)):
                    current_h = fill_col_coords[point_index]
                    current_w = fill_row_coords[point_index]

                    # c_point_depth = 1 / (current_bcoor[0]/current_surface_depth[0] +
                    #                            current_bcoor[1]/current_surface_depth[1] +
                    #                            current_bcoor[2]/current_surface_depth[2])
                    
                    c_point_depth = get_depth_bc(current_w, current_h, current_s_with_depth)
                    points_info_dict, final_covered_points = update_depth_dict_v2((current_h, current_w), c_point_depth,
                                                                                  points_info_dict, f_covered_points,
                                                                                  current_surface, point_c_3d, obj_count)
                
                    current_hwpoint.append((current_h, current_w))
                
                mesh_3d_dict[point_3d_list_tuple] = ["porjected", current_hwpoint, current_surface, current_surface_depth, obj_count]

            else:

                mesh_3d_dict[point_3d_list_tuple] = ["backface", [], current_surface, current_surface_depth, obj_count]                
                

        obj_count += 1


    colors_id = ncolors(obj_count - 1)
    # print(colors_id)

    projected_points_info_dict = dict()
    for key, value in points_info_dict.items():
        if len(points_info_dict[key]) != 0:
            projected_points_info_dict[key] = value
            paint_one_point(key, img, color=[255, 255, 255])
            paint_one_point(key, img_idmap, color=colors_id[value[0][-1] - 1])

    # if configs.extra_rotation:
    #     save_file = save_file.replace("projection", configs.extra_rot_name + "_projection")
    
    
    cv2.imwrite(save_file, img)

    save_file = save_file.replace("projection", "idmap")
    # if configs.extra_rotation:
    #     save_file = save_file.replace("idmap", configs.extra_rot_name + "_idmap")
    cv2.imwrite(save_file, img_idmap)
    
    if configs.use_down_sampling:
        save_file = save_file.replace("idmap", "projection").replace("ss_", "")
        img = average_filter_downsample(img, configs.down_sampling_rate, save_file,
                              keep_ob3color=False, simple_filtered=False)
        save_file = save_file.replace("projection", "idmap").replace("ss_", "")                     
        img_idmap = average_filter_downsample(img_idmap, configs.down_sampling_rate, save_file,
                              keep_ob3color=False, simple_filtered=False)

    return projected_points_info_dict, no_projection_mesh, mesh_3d_dict, vertex_map_dict


def run_projection(model_pose_info_file, folder, projection_name):

    # intrinsic_file = configs.intrinsic_file
    # K = np.load(intrinsic_file, allow_pickle=True)

    ca_m = np.load(configs.camera_matrix_file, allow_pickle=True)
    ca_m = ca_m.astype(np.float64)
    model_pose_infos = np.load(model_pose_info_file, allow_pickle=True)
    # np.save(configs.image_save_dir + folder + "pose_info.npy", model_pose_infos)
    model_dir = configs.obj_model_dir

    # if configs.count_backface:
    #     save_file = configs.image_save_dir + folder + "countback_" + projection_name
    # else:
    #     save_file = configs.image_save_dir + folder + projection_name

    save_file = configs.image_save_dir + folder + projection_name
    img_file = configs.background_img_file

    # points_info_dict, no_projection_mesh, final_covered_points = projection_with_depth(model_pose_infos, K, model_dir, img_file, save_file, configs.depth_type)
    points_info_dict, no_projection_mesh, mesh_3d_dict, vertex_map_dict = projection_skiimage(model_pose_infos, ca_m, model_dir, img_file, save_file)

    return points_info_dict, no_projection_mesh, mesh_3d_dict, vertex_map_dict


def occulsion_boundary_definition(img_original, img_projected, points_info_dict, graph_projection,
                                  mesh_3d_dict, num_neighbor=4, vertex_map_dict=None):
    same_count = 0
    length_list = list()
    labels_info_dict = dict()
    orientation_label = dict()
    black_point = [0, 0, 0]
    white_point = [255, 255, 255]

    # delete one type bad projection point, i.e. one black point's 4 neighbors are all white.
    # other type cannot judge easily.
    bad_projection_points = list()
    for i in range(img_projected.shape[0]):
        for j in range(img_projected.shape[1]):
            if img_projected[i, j, :].tolist() == black_point:
                all_4_neighbors = get_point_neighbors((i, j), img_original.shape, 4)
                if len(all_4_neighbors) != 0:
                    for x, y in all_4_neighbors:
                        if img_projected[x, y, :].tolist() != white_point:
                            break
                    else:
                        bad_projection_points.append((i, j))

    # points_list = list(points_info_dict.keys())
    # for point in tqdm(points_list):
    for point in points_info_dict.keys():
        if len(points_info_dict[point]) != 1:
            same_count += 1

        # check point information
        orientation_label[point] = list()
        check_point_info = points_info_dict[point][0]
        check_neighbors = get_point_neighbors(point, img_original.shape, num_neighbor)

        # image boundary point, skip
        if len(check_neighbors) == 0:
            continue

        # not image boundary points
        for item in check_neighbors:
            # junction of projected object and the black background
            color_neighbor = img_projected[item[0], item[1], :].tolist()
            if color_neighbor == black_point and item not in bad_projection_points:
                # img_output[point[0], point[1], :] = white_point
                labels_info_dict = update_point_label(point, OBP_CLASS[0], labels_info_dict)
                orientation_label[point].append((point, item))
                continue

            if item in bad_projection_points:
                continue

            # different obj point
            if points_info_dict[item][0][-1] != check_point_info[-1]:
                # option 1: choose depth closer one
                if points_info_dict[item][0][0] < check_point_info[0]:
                    # make sure always point --> item
                    # orientation_label[point].append((item, point))
                    # img_output[item[0], item[1], :] = white_point
                    # labels_info_dict = update_point_label(item, OBP_CLASS[1], labels_info_dict)
                    continue
                elif points_info_dict[item][0][0] > check_point_info[0]:
                    orientation_label[point].append((point, item))
                    # img_output[point[0], point[1], :] = white_point
                    labels_info_dict = update_point_label(point, OBP_CLASS[1], labels_info_dict)
                else:
                    orientation_label, labels_info_dict = process_same_depth(item, point, points_info_dict,
                                                                             orientation_label, labels_info_dict)
                continue

            # self occlusion point
            # if check_z_fighting(point, item, points_info_dict, graph_projection, length_list):
            if graph_check_connection_new_v2(point, item, points_info_dict, graph_projection, length_list, mesh_3d_dict, vertex_map_dict):
                continue
            else:
                if points_info_dict[item][0][0] < check_point_info[0]:
                    # orientation_label[point].append((item, point))
                    # img_output[item[0], item[1], :] = self_obp_color
                    # labels_info_dict = update_point_label(item, OBP_CLASS[2], labels_info_dict)
                    # count_check += 1
                    continue
                elif points_info_dict[item][0][0] > check_point_info[0]:
                    orientation_label[point].append((point, item))
                    # img_output[point[0], point[1], :] = self_obp_color
                    labels_info_dict = update_point_label(point, OBP_CLASS[2], labels_info_dict)
                else:
                    orientation_label, labels_info_dict = process_same_depth(item, point, points_info_dict,
                                                                             orientation_label, labels_info_dict)

    # orientation_label_checked = dict()
    # for key, value in orientation_label.items():
    #     if len(orientation_label[key]) != 0:
    #         orientation_label_checked[key] = value
    # assert len(orientation_label_checked.keys()) == len(labels_info_dict.keys())
    # print(Counter(length_list))

    return same_count, labels_info_dict, orientation_label


def run_definition(points_info_dict, folder, projection_name, graph_projection, mesh_3d_dict, vertex_map_dict):
    img_original_file = configs.background_img_file
    img_original = cv2.imread(img_original_file)
    # if configs.count_backface:
    #     count_back_name = "countback_"
    # else:
    #     count_back_name = ""
    projection_dir = configs.image_save_dir + folder + projection_name
    img_projected = cv2.imread(projection_dir)

    same_count, labels_info_dict, orientation_label = occulsion_boundary_definition(img_original, img_projected,
                                                                                    points_info_dict, graph_projection,
                                                                                    mesh_3d_dict, 4, vertex_map_dict)

    return same_count, labels_info_dict, orientation_label


def filter_label(orientation_label, labels_info_dict, save_dir):
    img_original_file = configs.background_img_file
    img_original = cv2.imread(img_original_file)
    white_point = np.array([255, 255, 255])

    # unfiltered
    img_ob = img_original.copy()
    img_ob_color = img_original.copy()
    img_oo = np.ones((img_original.shape[0], img_original.shape[1], 3)) * 112  # img_original.copy()
    uf_orientation_label = dict()
    uf_boundary_label = dict()

    # # filtered
    # img_obf = img_original.copy()
    # img_obf_color = img_original.copy()
    # img_oof = np.ones((img_original.shape[0], img_original.shape[1], 3)) * 112  # img_original.copy()
    # f_orientation_label = dict()
    # f_boundary_label = dict()
    # # print(img_oo[0, 0, :])
    # threshold_count = 3
    # delete_points = list()
    # points_not_count_again = list()

    for key, value in orientation_label.items():
        if len(orientation_label[key]) != 0:
            # unfiltered
            uf_orientation_label[key] = value
            uf_boundary_label[key] = labels_info_dict[key]
            angle0 = get_angle_value(value)
            # angle = np.ceil(angle0 * 255 / 360)
            angle = angle0 * 255 / 360
            img_ob[key[0], key[1], :] = white_point
            img_oo[key[0], key[1]] = np.array([angle, angle, angle])
            img_ob_color[key[0], key[1], :] = get_3_colors(uf_boundary_label[key])

    #         # filtered  # filter single point or not a line point
    #         obp_count = 1
    #         check_neighbors = get_point_neighbors(key, img_original.shape, 8)
    #         # we could also creat a 5*5 mask, check obp_count>=5, one column count once
    #         for cn in check_neighbors:
    #             # x_list = list()
    #             # y_list = list()
    #             if cn in orientation_label.keys() and len(orientation_label[cn]) != 0 \
    #                     and cn not in points_not_count_again:
    #                 obp_count += 1
    #                 # x_list.append(cn[0])
    #                 # y_list.append(cn[1])
    #         if obp_count >= threshold_count:
    #             # delete not line points, not corners
    #             # if check_list_diff2(x_list) or check_list_diff2(y_list):
    #             f_orientation_label[key] = value
    #             f_boundary_label[key] = labels_info_dict[key]
    #         else:
    #             # continue
    #             points_not_count_again.append(key)
    #     else:
    #         # delete one situation, that's center point are white but not obp, yet all 4 neighbors are obp&white
    #         # not always right/good...
    #         all_4_neighbors = get_point_neighbors(key, img_original.shape, 4)
    #         all_4_corners = get_corner_neighbors(key, img_original.shape)
    #         if len(all_4_neighbors) != 0:
    #             for pp in all_4_neighbors:
    #                 try:
    #                     if len(orientation_label[pp]) == 0:
    #                         # another situation, not work well
    #                         # pp_4_neighbors = get_point_neighbors(pp, img_original.shape, 4)
    #                         # pp_3_neighbors = [item for item in pp_4_neighbors if item != key]
    #                         # for ppn in pp_3_neighbors:
    #                         #     if len(orientation_label[ppn]) == 0:
    #                         #         break
    #                         # else:
    #                         #     delete_points.extend(all_4_neighbors)
    #                         #     delete_points.extend(pp_3_neighbors)
    #                         break
    #                 except KeyError:
    #                     break
    #             else:
    #                 # other 4 corners are not white.
    #                 for cc in all_4_corners:
    #                     try:
    #                         if len(orientation_label[cc]) != 0:
    #                             break
    #                     except KeyError:
    #                         break
    #                 else:
    #                     delete_points.extend(all_4_neighbors)
    #
    # # filter bad line segments, not finish
    # # print(points_not_count_again)
    # # paint oo image and ob image
    # for key in f_boundary_label.keys():
    #     if key not in delete_points:
    #         angle0 = get_angle_value(f_orientation_label[key])
    #         # angle1 = int(angle0*255/361)
    #         # angle2 = int(angle0*255/360)
    #         # print(angle0, angle0*255/360, np.ceil(angle0*255/360), angle1, angle2)
    #         # angle = np.ceil(angle0*255/360)
    #         angle = angle0 * 255 / 360
    #         img_obf[key[0], key[1], :] = white_point
    #         img_oof[key[0], key[1]] = np.array([angle, angle, angle])
    #         img_obf_color[key[0], key[1], :] = get_3_colors(f_boundary_label[key])

    # ob_save_name = str(configs.threshold_shortest_len) + "_" + str(threshold_count) + "_ob2.png"
    # oo_save_name = str(configs.threshold_shortest_len) + "_" + str(threshold_count) + "_oo2.png"

    if configs.extra_rotation:
        save_dir = save_dir + configs.extra_rot_name + "_"

    # print(save_dir)
    ob_save_name = "dis_dOB.png"  # "occlusion_boundary.png"
    oo_save_name = "dis_dOO.png"  # "occlusion_orientation.png"
    ob3_save_name = "dis_dOB3.png"
    cv2.imwrite(save_dir + oo_save_name, img_oo)
    cv2.imwrite(save_dir + ob_save_name, img_ob)
    cv2.imwrite(save_dir + ob3_save_name, img_ob_color)

    # cv2.imwrite(save_dir + "f" + oo_save_name, img_oof)
    # cv2.imwrite(save_dir + "f" + ob_save_name, img_obf)
    # cv2.imwrite(save_dir + "f" + ob3_save_name, img_obf_color)
    return uf_boundary_label, uf_orientation_label  # f_boundary_label, f_orientation_label


def filter_label_downsampling(orientation_label, labels_info_dict, save_dir):
    img_original_file = configs.background_img_file
    img_original = cv2.imread(img_original_file)
    white_point = np.array([255, 255, 255])

    # unfiltered
    img_ob = img_original.copy()
    img_ob_color = img_original.copy()
    img_oo = np.ones((img_original.shape[0], img_original.shape[1], 3)) * 112  # img_original.copy()
    img_oo_down = np.zeros((img_original.shape[0], img_original.shape[1], 3)) # * (-1)
    
    uf_orientation_label = dict()
    uf_boundary_label = dict()


    for key, value in orientation_label.items():
        if len(orientation_label[key]) != 0:
            # unfiltered
            # uf_orientation_label[key] = value
            # uf_boundary_label[key] = labels_info_dict[key]
            angle0 = get_angle_value(value)
            # angle = np.ceil(angle0 * 255 / 360)
            angle = angle0 * 255 / 360
            img_ob[key[0], key[1], :] = white_point
            img_oo[key[0], key[1]] = np.array([angle, angle, angle])
            img_oo_down[key[0], key[1]] = np.array([angle0, angle0, angle0])
            img_ob_color[key[0], key[1], :] = get_3_colors(labels_info_dict[key])


    if configs.extra_rotation:
        save_dir = save_dir + configs.extra_rot_name + "_"

    # print(save_dir)
    ob_save_name = "ss_dis_OB.png"  # "occlusion_boundary.png"
    oo_save_name = "ss_dis_OO.png"  # "occlusion_orientation.png"
    ob3_save_name = "ss_dis_OB3.png"
    cv2.imwrite(save_dir + oo_save_name, img_oo)
    cv2.imwrite(save_dir + ob_save_name, img_ob)
    cv2.imwrite(save_dir + ob3_save_name, img_ob_color)

    average_filter_downsample(img_ob_color, configs.down_sampling_rate, save_dir + "dis_OB3.png", keep_ob3color=True, simple_filtered=False)

    average_filter_downsample(img_ob, configs.down_sampling_rate, save_dir  + "dis_OB.png", keep_ob3color=False, simple_filtered=False)
    
    average_filter_downsample(img_oo_down, configs.down_sampling_rate, save_dir  + "dis_OO.png", keep_ob3color=False, simple_filtered=False, is_oo_img=True)


    return uf_boundary_label, uf_orientation_label  



def run_one(model_pose_info_file, train_db=True):
    projection_name = "ss_projection.png"
    if configs.extra_rotation:
        projection_name = configs.extra_rot_name + "_" + projection_name

    # train imgae num 14761
    # if train_db:
    #     current_img_name = model_pose_info_file.split("/")[-1].replace(".npy", "")
    #     folder = "scene/" + current_img_name + "/"
    # else:
    #     current_img_name = model_pose_info_file.split("/")[-1].replace(".npy", "")
    #     current_img_name = str(int(current_img_name) + 14761 + 1).zfill(7)
    #     folder = "scene/" + current_img_name + "/"

    current_img_name = model_pose_info_file.split("/")[-2]
    # print(current_img_name)
    folder = current_img_name + "/"

    if not os.path.exists(configs.image_save_dir + folder):
        os.makedirs(configs.image_save_dir + folder)
        # os.makedirs(configs.image_save_dir + folder + "info/")

    # # step 1: finish projection and get points info dict
    logging.info("Start projection for " + folder + " : ")
    # logging.info("Projection img store at "+folder + projection_name)
    start = time.time()
    points_info_dict, no_projection_mesh, mesh_3d_dict, vertex_map_dict = run_projection(model_pose_info_file,
                                                                                folder, projection_name)
    logging.info("There are {} mesh don't have projection points! ".format(len(no_projection_mesh)))
    logging.info("Finish projection in {} seconds !".format(int((time.time() - start))))

    # # step 2: build projection graph
    # graph_projection = build_projection_graph(points_info_dict, no_projection_mesh)
    graph_projection = build_3dmesh_graph(mesh_3d_dict, no_projection_mesh)

    # # step 3: generate occlusion boundary relationships
    logging.info("Start ob judgement!")
    # print(configs.__dict__)
    start = time.time()
    same_count, labels_info_dict, orientation_label = run_definition(points_info_dict, folder, projection_name,
                                                                     graph_projection, mesh_3d_dict, vertex_map_dict)
    logging.info("{} points have same depth !".format(same_count))

    # # step 4: filter ob and oo label image
    ob_save_dir = configs.image_save_dir + folder
    if configs.use_down_sampling:
        uf_boundary_label, uf_orientation_label = filter_label_downsampling(orientation_label, labels_info_dict, ob_save_dir)
    else:
        uf_boundary_label, uf_orientation_label = filter_label(orientation_label, labels_info_dict, ob_save_dir)
    logging.info("Finish ob judgement in {} minutes !".format(int((time.time() - start) / 60)))
    logging.info("/n")
    # print((time.time() - start) / 60)

    # # save all info; use dtype=object deal with warning
    # np.save(configs.image_save_dir + folder + "/info/points_info.npy", np.array(list(points_info_dict.items())))
    # np.save(configs.image_save_dir + folder + "/info/no_projection_meshes.npy", np.array(no_projection_mesh))
    # np.save(configs.image_save_dir + folder + "/info/covered_points_list.npy", np.array(final_covered_points))
    # # pos = nx.nx_agraph.graphviz_layout(graph_projection)
    # # nx.draw(graph_projection, pos=pos)
    # write_dot(graph_projection, configs.image_save_dir + folder + "/info/graph_projection.dot")
    # np.save(configs.image_save_dir + folder +"/info/orientation_label.npy", np.array(list(orientation_label.items())))
    # np.save(configs.image_save_dir + folder + "/info/labels_info.npy", np.array(list(labels_info_dict.items())))
    # np.save(configs.image_save_dir + folder + "/info/filtered_OO.npy", np.array(list(f_orientation_label.items())))
    # np.save(configs.image_save_dir + folder + "/info/filtered_OB.npy", np.array(list(f_labels_info_dict.items())))

def vis_ob_main_diff(img_path, img_path2, name, radius=1):

    image1 = cv2.imread(img_path,  cv2.IMREAD_UNCHANGED)
    image2 = cv2.imread(img_path2,  cv2.IMREAD_UNCHANGED)
    img1_bmap = ((np.array(image1[:, :, 0]) > 0)*1).reshape(image1.shape[0], image1.shape[1], 1)    
    img2_bmap = ((np.array(image2[:, :, 0]) > 0)*1).reshape(image1.shape[0], image1.shape[1], 1)
        
    diff = np.zeros((image1.shape))

 
    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            if image1[i, j, :].tolist() != [0,0,0] or image2[i, j, :].tolist() != [0,0,0]:
                # points_neighbors = get_point_neighbors((i, j), image1.shape)
                img1_check = np.sum(img1_bmap[i-radius:i+radius+1, j-radius:j+radius+1 ], axis=(0, 1))[0] > 0
                img2_check = np.sum(img2_bmap[i-radius:i+radius+1, j-radius:j+radius+1 ], axis=(0, 1))[0] > 0
                if img1_check and img2_check:
                    continue
                elif img1_check and not img2_check:
                    image2[i, j, :] = [255, 0, 0]
                elif not img1_check and img2_check:
                    image2[i, j, :] = [0, 0, 255]
                               
    current_save_name = "./main_diff/ob_main_diff" + name + "_.png"
    cv2.imwrite(current_save_name, image2)


if __name__ == '__main__':
    print("Graph connection distance check --- ", configs.use_threshold_shortest)
    # random.seed(configs.random_seed)
    # t, t1, t2 = compare_two_images("obdb/scene/0014829/OB1.png",
    #                "obdb/scene/0014829/OB.png",
    #                "obdb/scene/0014829/", return_colored=True)

    start = time.time()
    print(configs.img_resolution_y, configs.img_resolution_x)
    # create_black_background((configs.img_resolution_y, configs.img_resolution_x))

    error_list = list()

    # img_list_path = "../occdata/synocc_split/synocc_train_3.txt"
    # img_list_path = "../occdata/synocc_split/synocc_fval.txt" 

    # with open(img_list_path, 'r') as f:
    #     names = f.readlines()
    # run_list = [x.replace('\n', '') for x in names]
    
    key_min =  52
    key_max =  53
    runner_list = []
    
    for i in range(key_min, key_max):
        # key = run_list[i]
        key = i

        model_pose_name = str(key).zfill(5)
        model_pose_info_file = configs.pose_info_dir + model_pose_name + "/pose_info.npy"
        model_pose_infos = np.load(model_pose_info_file, allow_pickle=True)
        print(model_pose_infos)
        print(name)
 
        # scene 18316 long  7fdd4112-7d50-4793-a18c-23c6920b7306
        # scene 00427 long  138c90bc-8505-4db6-8920-87a6421e66f0
       
        store_path = configs.image_save_dir + model_pose_name + "/"
        if not os.path.exists(store_path):
            os.makedirs(store_path)
        file_count = os.listdir(store_path)
        
        starttt = time.time()

        run_one(model_pose_info_file)

        # if len(file_count) < 14 :
        #     # try:
        #     run_one(model_pose_info_file)
        #     print(key, " --- time cost: " ,(time.time() - starttt) / 60)
        # except:
        # error_list.append(key)


        # s_path = configs.image_save_dir + model_pose_name + "/"
        # if configs.extra_rotation:
        #     c_path = configs.image_save_dir + model_pose_name + "/" + configs.extra_rot_name + "_brgba.png"
        # else:
        #     c_path = configs.image_save_dir + model_pose_name + "/brgba.png"

        # # # add multi random background
        # bg_index = random.sample(range(1, 5), configs.rgb_num)
        # count_bg = 0
        # for bg_i in bg_index:
        #     bg_i_path = "./data/bg_1080/" + str(bg_i) + ".png"
        #     # bg_i_path = "./data/bg_960/" + str(bg_i) + ".png"
        #     count_bg += 1
        #     save_name = "rgb_{}.png".format(count_bg)
        #     if configs.extra_rotation:
        #         save_name = configs.extra_rot_name + "_" + save_name

        #     add_rgba_background(c_path, bg_i_path, s_path, save_name)


    print(error_list)
    print((time.time() - start) / 60)
