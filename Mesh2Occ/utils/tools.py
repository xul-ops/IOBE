import os
import sys
import pdb
import cv2
import copy
import math
import time
import random
# import cython
import colorsys
import logging
import itertools
import numpy as np
import networkx as nx
from skimage import draw
# import scipy.linalg as linalg
# from collections import Counter
from PIL import Image
from decimal import *
from collections import defaultdict
from tqdm import tqdm
# from pywavefront import Wavefront
# from tqdm.std import trange
# from mlaa import mlaa_img_luminance

# sys.path.append(os.getcwd())
# print(getcontext())
import utils.configs as configs
OBP_CLASS = ["background_obj_obp", "different_obj_obp", "self_obp"]

if configs.is_decimal:
    getcontext().prec = configs.decimal_prec
    # cython test
    # def depth_z_value(z_eye: cython.double):
    #     f = 100.0
    #     n = 0.1
    #     # a: cython.int = 1
    #     # z_ndc = (-z_eye * (f + n) / (f - n) - 2 * f * n / (f - n)) / -z_eye
    #     return 1 / z_eye
    # def interpolation_depth(point_inte: (cython.int, cython.int), point1: (cython.int, cython.int),
    #                         point2: (cython.int, cython.int), point1_value: cython.double, point2_value: cython.double):
    #
    #     if point1[0] == point2[0]:
    #         if point1[1] > point2[1]:
    #             depth_current = point1_value - (point1_value - point2_value) * (point1[1] - point_inte[1]) / (
    #                     point1[1] - point2[1])
    #         else:
    #             depth_current = point2_value - (point2_value - point1_value) * (point2[1] - point_inte[1]) / (
    #                     point2[1] - point1[1])
    #     # elif point1[0] != point2[0]:
    #     else:
    #         if point1[0] > point2[0]:
    #             depth_current = point1_value - (point1_value - point2_value) * (point1[0] - point_inte[0]) / (
    #                     point1[0] - point2[0])
    #         else:
    #             depth_current = point2_value - (point2_value - point1_value) * (point2[0] - point_inte[0]) / (
    #                     point2[0] - point1[0])
    #
    #     return depth_current
    def dis2surface(surface, camera_point=(0, 0, 0)):
        x = Decimal(str(np.abs(surface[3])))
        y = np.sqrt(
            Decimal(str(surface[0])) * Decimal(str(surface[0])) + Decimal(str(surface[1])) * Decimal(str(surface[1])) +
            Decimal(str(surface[2])) * Decimal(str(surface[2])))
        return x / y


    def dis2camera(point_3d, camera_point=(0, 0, 0)):
        t = Decimal(str(point_3d[0])) * Decimal(str(point_3d[0])) + Decimal(str(point_3d[1])) * Decimal(
            str(point_3d[1])) \
            + Decimal(str(point_3d[2])) * Decimal(str(point_3d[2]))
        return np.sqrt(t)

    def interpolation_depth(point_inte, point1, point2, point1_value, point2_value):
        point1 = [Decimal(str(i)) for i in point1]
        point2 = [Decimal(str(i)) for i in point2]
        point_inte = [Decimal(str(i)) for i in point_inte]
        if point1[0] == point2[0]:  # and point_inte[0] == point1[0]:

            if point1[1] > point2[1]:
                length = (point1[1] - point_inte[1]) / (point1[1] - point2[1])
                depth_current = point1_value - (point1_value - point2_value) * length
            else:
                if point1[1] == point2[1] and point1[1] == point_inte[1]:
                    return point2_value
                length = (point2[1] - point_inte[1]) / (point2[1] - point1[1])
                depth_current = point2_value - (point2_value - point1_value) * length
        # elif point1[0] != point2[0]:
        else:
            if point1[0] > point2[0]:
                depth_current = point1_value - (point1_value - point2_value) * (point1[0] - point_inte[0]) / (
                        point1[0] - point2[0])
            else:
                depth_current = point2_value - (point2_value - point1_value) * (point2[0] - point_inte[0]) / (
                        point2[0] - point1[0])

        return depth_current

    def ndc(v):
        w = Decimal(str(v[-1]))
        x, y, z = Decimal(str(v[0])) / w, Decimal(str(v[1])) / w, Decimal(str(v[2])) / w
        return [x, y, z, 1 / w]


    def viewport(v):
        x = y = 0
        w, h = configs.img_resolution_x, configs.img_resolution_y
        n, f = Decimal(str(0.1)), Decimal(str(100))
        return np.array([w * Decimal(str(0.5)) * v[0] + x + w * Decimal(str(0.5)),
            h * Decimal(str(0.5)) * v[1] + y + h * Decimal(str(0.5)),
            Decimal(str(0.5)) * (f - n) * v[2] + Decimal(str(0.5)) * (f + n)])

else:

    def dis2surface(surface, camera_point=(0, 0, 0)):
        x = np.abs(surface[3])
        y = np.sqrt(surface[0] * surface[0] + surface[1] * surface[1] + surface[2] * surface[2])
        return x / y


    def dis2camera(point_3d, camera_point=(0, 0, 0)):
        point_3d = [np.float64(i) for i in point_3d]
        return np.sqrt(pow(point_3d[0], 2) + pow(point_3d[1], 2) + pow(point_3d[2], 2))


    def interpolation_depth(point_inte, point1, point2, point1_value, point2_value):
        # print(point1_value.dtype)
        point1 = [np.float64(i) for i in point1]
        point2 = [np.float64(i) for i in point2]
        point_inte = [np.float64(i) for i in point_inte]
        if point1[0] == point2[0]:  # and point_inte[0] == point1[0]:
            if point1[1] > point2[1]:
                depth_current = point1_value - (point1_value - point2_value) * (point1[1] - point_inte[1]) / (
                        point1[1] - point2[1])
            else:
                depth_current = point2_value - (point2_value - point1_value) * (point2[1] - point_inte[1]) / (
                        point2[1] - point1[1])
        # elif point1[0] != point2[0]:
        else:
            if point1[0] > point2[0]:
                depth_current = point1_value - (point1_value - point2_value) * (point1[0] - point_inte[0]) / (
                        point1[0] - point2[0])
            else:
                depth_current = point2_value - (point2_value - point1_value) * (point2[0] - point_inte[0]) / (
                        point2[0] - point1[0])

        return depth_current


    def ndc(v):
        w = v[-1]
        x, y, z = v[0] / w, v[1] / w, v[2] / w
        return np.array([x, y, z, 1 / w])


    def viewport(v):
        x = y = 0
        w, h = configs.img_resolution_x, configs.img_resolution_y
        n, f = 0.1, 100
        return np.array([w * 0.5 * v[0] + x + w * 0.5,
                         h * 0.5 * v[1] + y + h * 0.5,
                         0.5 * (f - n) * v[2] + 0.5 * (f + n)])


def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step

    return hls_colors


def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])

    return rgb_colors


def create_black_background(img_size=(1200, 1200)):
    black = np.zeros(img_size)
    cv2.imwrite('data/scene_data/black2.png', black)


def get_surface(mesh):
    x1, y1, z1 = mesh[0]
    x2, y2, z2 = mesh[1]
    x3, y3, z3 = mesh[2]
    a = ((y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1))
    b = ((z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1))
    c = ((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
    d = (0 - (a * x1 + b * y1 + c * z1))

    return [a, b, c, d]


def o3d_read_mesh(obj_file, rot_mat, trans_vec):
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(obj_file)
    mesh = mesh.rotate(rot_mat)
    mesh = mesh.translate(trans_vec)

    mesh_in = copy.deepcopy(mesh)
    # print("Current obj file at ", obj_file)
    print(f'Original mesh has {len(mesh_in.vertices)} vertices and {len(mesh_in.triangles)} triangles')
    # voxel_size = max(mesh_in.get_max_bound() - mesh_in.get_min_bound()) / configs.voxel_size_changes
    # print(f'voxel_size = {voxel_size:e}')
    # mesh_smp = mesh_in.simplify_quadric_decimation(target_number_of_triangles=len(mesh_in.triangles)-500)
    # if configs.contraction_method == "average":
    #     mesh_smp = mesh_in.simplify_vertex_clustering(
    #         voxel_size=voxel_size,
    #         contraction=o3d.geometry.SimplificationContraction.Average)
    # else:
    #     mesh_smp = mesh_in.simplify_vertex_clustering(
    #         voxel_size=voxel_size,
    #         contraction=o3d.geometry.SimplificationContraction.Quadric)
    if configs.submesh_method == "midpoint":
        mesh_smp = mesh_in.subdivide_midpoint(number_of_iterations=configs.number_of_iterations)
    else:
        mesh_smp = mesh_in.subdivide_loop(number_of_iterations=configs.number_of_iterations)
    print(f'Modified mesh has {len(mesh_smp.vertices)} vertices and {len(mesh_smp.triangles)} triangles')
    # o3d.visualization.draw_geometries([mesh_smp])

    # print(np.max(mesh.get_max_bound() - mesh.get_min_bound()))
    # print(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()))
    # scale 1.0 1.5 0.5
    # a slight scale will speed up the projection step but sometimes the resulting image is not good
    mesh.scale(1.0 / np.max(mesh.get_max_bound() - mesh.get_min_bound()), center=mesh.get_center())
    # mesh_vertices_trans, mesh_surfaces = np.asarray(mesh_smp.vertices), np.asarray(mesh_smp.triangles)
    return np.asarray(mesh.vertices), np.asarray(mesh.triangles)


# def assign_mesh_ids_and_coordinates(obj_file_path):
#     mesh_data = {}

#     # Load the OBJ file
#     scene = Wavefront(obj_file_path)

#     # Iterate through each mesh in the scene
#     for mesh_name, material in scene.meshes.items():
#         vertices = scene.vertices[mesh_name]
#         mesh_data[mesh_name] = {'id': len(mesh_data) + 1, 'coordinates': vertices}

#     return mesh_data


def add_1_in_position(a):
    a = a.tolist()
    a.append(1)
    return np.array(a)


def transfer_wc(a, T):
    return np.matmul(T, a)


def get_obj_vertex_ali(file):
    """
    get obj vertex, some obj file can not be loaded through open3d or trimesh
    """

    with open(file, 'r') as f:
        surface_group = []
        vertex_group = []
        part_vertex = []
        last_fisrt = ''

        lines = f.readlines()

        for i, line in enumerate(lines):
            line = line.strip()
            split_line = line.split(' ')
            curr_first = split_line[0]

            if curr_first != last_fisrt:
                if part_vertex != []:
                    vertex_group += part_vertex
                part_vertex = []

            if 'v' == curr_first:
                try:
                    vertex = [float(split_line[-3]), float(split_line[-2]), float(split_line[-1])]
                except:
                    continue
                    pdb.set_trace()
                    vertex = [float(split_line[-3]), float(split_line[-2]), float(split_line[-1])]
                part_vertex.append(vertex)

            last_fisrt = curr_first

            if 'f' == curr_first:
                current_f = []
                for j in range(1, len(split_line)):
                    current_f.append(split_line[j].split("/")[0])

                surface_group.append(np.array(current_f))

            last_fisrt = curr_first

        # remove shadow vertex
        if len(vertex_group[-1]) == 4 and len(vertex_group) != 0:
            print(vertex_group[-1])
            vertex_group.pop()
    return np.array(vertex_group), np.array(surface_group)   
        
    try:
        # some numpy version dont support this
        return np.array(vertex_group), np.array(surface_group)
    except ValueError as e:
        return vertex_group, surface_group

def get_obj_vertex_ali_idmap(file, obj_count):
    """
    get obj vertex, some obj file can not be loaded through open3d or trimesh
    """

    with open(file, 'r') as f:
        surface_group = []
        vertex_group = []
        part_vertex = []
        objects_mesh_dict = dict()
        last_fisrt = ''

        lines = f.readlines()

        for i, line in enumerate(lines):
            line = line.strip()
            split_line = line.split(' ')
            curr_first = split_line[0]

            if split_line[-1].startswith("solid_"): 
                objects_mesh_dict[obj_count] = np.array(surface_group)
                obj_count += 1
                surface_group = []
                # vertex_group = []
                # part_vertex = []

            if curr_first != last_fisrt:
                if part_vertex != []:
                    vertex_group += part_vertex
                part_vertex = []

            if 'v' == curr_first:
                try:
                    vertex = [float(split_line[-3]), float(split_line[-2]), float(split_line[-1])]
                except:
                    continue
                    pdb.set_trace()
                    vertex = [float(split_line[-3]), float(split_line[-2]), float(split_line[-1])]
                part_vertex.append(vertex)

            last_fisrt = curr_first

            if 'f' == curr_first:
                current_f = []
                for j in range(1, len(split_line)):
                    current_f.append(split_line[j].split("/")[0])

                surface_group.append(np.array(current_f))

            last_fisrt = curr_first

        # remove shadow vertex
        if len(vertex_group[-1]) == 4 and len(vertex_group) != 0:
            print(vertex_group[-1])
            vertex_group.pop()

        objects_mesh_dict[obj_count] = np.array(surface_group)
        obj_count += 1

    return objects_mesh_dict, obj_count , np.array(vertex_group)  
        
    # try:
    #     # some numpy version dont support this
    #     return np.array(vertex_group), np.array(surface_group)
    # except ValueError as e:
    #     return vertex_group, surface_group


def build_triangle_node(triangle, mesh, obj_count, graph):

    nodes_obj = [(point[0], point[1], obj_count) for point in triangle]
    nodes = [(str(nodes_obj[i]), {"3d_point": mesh[i]}) for i in range(len(nodes_obj))]
    edges = list(itertools.combinations(nodes_obj, 2))
    bi_edges = list()
    # weight_list = list()

    graph.add_nodes_from(nodes)

    for edge in edges:
        if configs.add_graph_edge_weight:
            c_weight = distance_points(edge[0], edges[1])
            # weight_list.append(weight)
            # bi_edges.extend([(str(edge[0]), str(edges[1])), (str(edge[1]), str(edges[0]))])
            # bi_edges.extend([(str(edge[0]), str(edges[1])), (str(edge[1]), str(edges[0]))])

        else:
            c_weight = 1
        
        
        c_edge = [(str(edge[0]), str(edges[1])), (str(edge[1]), str(edges[0]))]
        graph.add_edges_from(c_edge, weight=c_weight)

    # Pratically there won't has the projected coordinate points
    # with different 3d coordinates and the same 2d coordinates
    # 
    # for i in range(len(nodes)):
    #     try:
    #         # if exist, check 3d point is same or not
    #         if graph.nodes[nodes[i][0]]["3d_point"] == mesh[i]:
    #             continue
    #         else:
    #             print("Some points has same 2d coordinates but 3d positions are different.")
    #             logging.warning("Some points has same 2d coordinates but 3d positions are different.")
    #             continue
    #
    #     except KeyError:
    #         graph.add_nodes_from([nodes[i]])

    # graph.add_nodes_from(nodes)
    # if configs.add_graph_edge_weight:
    #     graph.add_edges_from(bi_edges. weight)
    # graph.add_edges_from(bi_edges)


    return graph


def build_projection_graph(points_info_dict, no_projection_mesh):
    graph_projection = nx.Graph()

    for key, value in points_info_dict.items():
        current_triangle = value[0][1]
        current_mesh = value[0][2]
        current_obj = value[0][3]
        graph_projection = build_triangle_node(current_triangle, current_mesh, current_obj, graph_projection)
        # break
    # print(name33)
    if not configs.count_small_faces: 
        return graph_projection

    for item in no_projection_mesh:
        current_triangle = item[0]
        current_mesh = item[1]
        current_obj = item[2]
        graph_projection = build_triangle_node(current_triangle, current_mesh, current_obj, graph_projection)

    return graph_projection


def calculate_3d_distance(point1, point2):
    """
    Calculate the Euclidean distance between two 3D points.
    Each point should be a tuple (x, y, z).
    """
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2 + (point2[2] - point1[2])**2)


# def calculate_3d_distance(point1, point2):

#     if len(point1) == 3 and len(point2) == 3:
#         return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2 + (point2[2] - point1[2])**2)
#     else:
#         print(len(point1), len(point2))
#         raise ValueError("Each point should be a tuple with three elements (x, y, z).")



def build_3dtriangle_node(graph, triangle_3d, triangle_info):

    nodes_obj = [(point[0], point[1], point[2]) for point in triangle_3d]
    nodes = [(str(nodes_obj[i]), {"3d_triangle_info": [triangle_info]}) for i in range(len(nodes_obj))]
    edges = list(itertools.combinations(nodes_obj, 2))
    bi_edges = list()

    # graph.add_nodes_from(nodes)

    for node in nodes:
        # print(node[0])
        # print(graph.nodes[node[0]]["3d_triangle_info"])
        try:
            exist_node_info =  graph.nodes[node[0]]["3d_triangle_info"]
            # print(exist_node_info)
            new_node_info = node[1]["3d_triangle_info"]
            new_node_info.extend(exist_node_info)
            graph.nodes[node[0]]["3d_triangle_info"] = new_node_info
            # print(graph.nodes[node[0]]["3d_triangle_info"])
            # pdb.set_trace()
        except KeyError:
            graph.add_nodes_from([node])

    for edge in edges:
        if configs.add_graph_edge_weight:
            c_weight = calculate_3d_distance(edge[0], edge[1])
        else:
            c_weight = 1
          
        c_edge = [(str(edge[0]), str(edges[1])), (str(edge[1]), str(edges[0]))]
        graph.add_edges_from(c_edge, weight=c_weight)

    return graph


def build_3dmesh_graph(mesh_3d_dict, no_projection_mesh):
    graph_projection = nx.Graph()

    # a = np.array(mesh_3d_dict.keys())
    # print(a.shape)
    # pdb.set_trace()
    # for i in mesh_3d_dict.keys():
    #     if len(i) != 3:
    #         print(i)
    #         print(mesh_3d_dict[i])
    #         pdb.set_trace()


    for key, value in mesh_3d_dict.items():
        # ["backface", [], current_surface, current_surface_depth, obj_count]        
        # current_3d_triangle = key
        # current_project_info = value[0]
        # current_projected_points = value[1]
        # current_2d_triangle = value[2]
        # current_2d_triangle_depth = value[3]
        # current_obj_count = value[-1]
        graph_projection = build_3dtriangle_node(graph_projection, key, value)


    return graph_projection


def build_local_graph(point1, point2, pixel_relatedFace_dict, **kwargs):

    graph_projection = nx.Graph()

    point1_related_face = pixel_relatedFace_dict[point1]
    point2_related_face = pixel_relatedFace_dict[point2]

    for item in point1_related_face:
        current_triangle = item[0]
        current_face = item[1]
        current_obj = item[2]

        graph_projection = build_triangle_node(current_triangle, current_face, current_obj, graph_projection)
 
    for item in point2_related_face:
        current_triangle = item[0]
        current_face = item[1]
        current_obj = item[2]

        graph_projection = build_triangle_node(current_triangle, current_face, current_obj, graph_projection)
 

    return graph_projection


def build_local_graph_v2(point1, point2, pixel_relatedFace_dict, no_projection_mesh):

    graph_projection = nx.Graph()

    point1_related_face = pixel_relatedFace_dict[point1]
    point2_related_face = pixel_relatedFace_dict[point2]

    for item in point1_related_face:
        current_triangle = item[0]
        current_face = item[1]
        current_obj = item[2]

        graph_projection = build_triangle_node(current_triangle, current_face, current_obj, graph_projection)
 
    for item in point2_related_face:
        current_triangle = item[0]
        current_face = item[1]
        current_obj = item[2]

        graph_projection = build_triangle_node(current_triangle, current_face, current_obj, graph_projection)
 

    for item in no_projection_mesh:
        current_triangle = item[0]
        current_mesh = item[1]
        current_obj = item[2]
        graph_projection = build_triangle_node(current_triangle, current_mesh, current_obj, graph_projection)

    return graph_projection


def graph_check_connection_v2(point1, point2, points_info_dict, 
                                graph_projection, length_list,
                                pixel_relatedFace_dict):
    point1_info = points_info_dict[point1][0]
    point2_info = points_info_dict[point2][0]
    
    one_point1 = (point1_info[1][0][0], point1_info[1][0][1], point1_info[-1])
    one_point2 = (point2_info[1][0][0], point2_info[1][0][1], point2_info[-1])

    # print("-"*100)
    # print(point1_info)
    # print(point1_info[1][0][0])

    # p1_tuple1 = item_to_tuple(point1_info[1][0][0], is_polygon=True)
    # p1_tuple2 = item_to_tuple(point1_info[1][0][1], is_polygon=False)
    # p2_tuple1 = item_to_tuple(point2_info[1][0][0], is_polygon=True)
    # p2_tuple2 = item_to_tuple(point2_info[1][0][1], is_polygon=False)
    # one_point1 = (p1_tuple1, p1_tuple2, point1_info[-1])
    # one_point2 = (p2_tuple1, p2_tuple2, point2_info[-1])

    # print()

    graph_projection = build_local_graph(point1, point2, pixel_relatedFace_dict)
    # different obj points will be processed in the previous step
    # chceck connection between these two triangles (same obj)
    if nx.has_path(graph_projection, str(one_point1), str(one_point2)):
        # threshold
        shortest_len = nx.shortest_path_length(graph_projection, source=str(one_point1), target=str(one_point2))
        length_list.append(shortest_len)
        if configs.use_threshold_shortest:
            if shortest_len <= configs.threshold_shortest_len:
                return True
            else:
                return False
        else:
            return True
    else:
        return False

def graph_check_connection(point1, point2, points_info_dict, graph_projection, length_list):
    point1_info = points_info_dict[point1][0]
    point2_info = points_info_dict[point2][0]

    one_point1 = (point1_info[1][0][0], point1_info[1][0][1], point1_info[-1])
    one_point2 = (point2_info[1][0][0], point2_info[1][0][1], point2_info[-1])

    # different obj points will be processed in the previous step
    # chceck connection between these two triangles (same obj)
    if nx.has_path(graph_projection, str(one_point1), str(one_point2)):
        # threshold
        shortest_len = nx.shortest_path_length(graph_projection, source=str(one_point1), target=str(one_point2))
        length_list.append(shortest_len)
        if configs.use_threshold_shortest:
            if shortest_len <= configs.threshold_shortest_len:
                return True
            else:
                return False
        else:
            return True
    else:
        return False


def check_easy_case(triangle1, triangle2):
    # equal or connecte directly
    t1 = [tuple(triangle1[0]), tuple(triangle1[1]), tuple(triangle1[2])]
    t2 = [tuple(triangle2[0]), tuple(triangle2[1]), tuple(triangle2[2])]
    intersection = list(set(t1) & set(t2))

    if len(intersection) > 0:
        return True
    else:
        return False

def check_easy_case_v2(point1_info, point2_info, mesh_3d_dict, points_info_dict, tolerance=1e-25):

    # equal or connecte directly
    # print(triangle1)
    triangle1, triangle2 = point1_info[1], point2_info[1]

    if triangle1 == triangle2:
        # print("11111111111")
        return 1

    t1 = [tuple(triangle1[0]), tuple(triangle1[1]), tuple(triangle1[2])]
    t2 = [tuple(triangle2[0]), tuple(triangle2[1]), tuple(triangle2[2])]
    intersection = list(set(t1) & set(t2))
    
    if len(intersection) > 0:

        triangle3d_1, triangle3d_2 = point1_info[2], point2_info[2]
        triangle3d_1 = item_to_tuple(triangle3d_1, is_polygon=False)
        triangle3d_2 = item_to_tuple(triangle3d_2, is_polygon=False)
        t1_p = mesh_3d_dict[triangle3d_1][1]
        t2_p = mesh_3d_dict[triangle3d_2][1]


        intersection_2 = list(set(t1_p) & set(t2_p))
        if len(intersection_2) != 0:
            # not consider if this two connected triangles fully visible
            # they must locally visible
            return 0
        else:
            return 1

        # slow and wrong

        # for point in t1_p:
        #     final_traingle_info = points_info_dict[point][0][1]
        #     if not are_points_equal(t1, final_traingle_info, tolerance=tolerance):
        #         return 0
        # for point in t2_p:
        #     final_traingle_info = points_info_dict[point][0][1]
        #     if not are_points_equal(t2, final_traingle_info, tolerance=tolerance):
        #         return 0

        # return 1
        # return True

    else:
        return 2

def distance_points(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def nearest_vertex(point, triangle):
    min_distance = float('inf')  # Initialize with positive infinity
    nearest_vertex = None

    for vertex in triangle:
        dist = distance_points(point, vertex)
        if dist < min_distance:
            min_distance = dist
            nearest_vertex = vertex

    return nearest_vertex

def max_distance_between_vertices(triangle1, triangle2):
    max_distance = 0.0

    for vertex1 in triangle1:
        for vertex2 in triangle2:
            dist = distance_points(vertex1, vertex2)
            # max_distance = max(max_distance, dist)
            max_distance = max(max_distance, dist)

    return max_distance

def min_distance_between_vertices(triangle1, triangle2):
    min_distance = 99999.9

    for vertex1 in triangle1:
        for vertex2 in triangle2:
            dist = distance_points(vertex1, vertex2)
            # max_distance = max(max_distance, dist)
            min_distance = min(min_distance, dist)

    return min_distance

def graph_check_connection_new(point1, point2, points_info_dict, graph_projection, length_list):
    point1_info = points_info_dict[point1][0]
    point2_info = points_info_dict[point2][0]
    # print("-"*100)
    # print(point1, point2)
    # print("-"*100)
    # print(point1_info[1], point2_info[1])
    # print("-"*100)

    # come from same triangle or directly connected one
    if check_easy_case(point1_info[1], point2_info[1]):
        return True
    
    # if configs.stop_at_easy: 
    #     return False

    # one_point1 = (point1_info[1][0][0], point1_info[1][0][1], point1_info[-1])
    # one_point2 = (point2_info[1][0][0], point2_info[1][0][1], point2_info[-1])

    # different obj points will be processed in the previous step
    # chceck connection between these two triangles (same obj)
    x1, y1 = nearest_vertex(point2, point1_info[1])
    one_point1 = (x1, y1, point1_info[-1])
    x2, y2 = nearest_vertex(point1, point2_info[1])
    one_point2 = (x2, y2, point2_info[-1])

    # else:
    if nx.has_path(graph_projection, str(one_point1), str(one_point2)):
        # threshold
        shortest_len = nx.shortest_path_length(graph_projection, source=str(one_point1), target=str(one_point2))
        shortest_path = nx.shortest_path(graph_projection, source=str(one_point1), target=str(one_point2))
        shortest_path = shortest_path[1:-2]


        max_dis = max_distance_between_vertices(point1_info[1], point2_info[1])
        min_dis = min_distance_between_vertices(point1_info[1], point2_info[1])

        # print(point1, point2)
        # print(one_point1, one_point2)
        # print(graph_projection)
        # print("-"*100)
        # print(shortest_path)
        # print("-"*100)
        start_point = one_point1[:2]
        end_point = one_point2[:2]

        path_points = list()

        distance_add = 0.0
        # can use edge weight, or calulated by the network
        for item in shortest_path:
            item = eval(item)
            if isinstance(item[0], tuple):
                continue
            else:
                c_point = item[:2]
                distance_add += distance_points(start_point, c_point)
                start_point = c_point
        
        distance_add += distance_points(start_point, end_point)

        # print(max_dis, min_dis, distance_add)
        # max_dis
        # np.pi * max_dis / 2
        # (max_dis+min_dis)/2
        # (max_dis+min_dis) / 2.25    3  4
        if distance_add < (max_dis+min_dis) / 4:  # * 3:  # /4:
            
            return True
        else:
            return False
                
        # print(item)
        # print(len(item))
        # print("-"*100)
        # pdb.set_trace()
        # length_list.append(shortest_len)
        # if configs.use_threshold_shortest:
        #     if shortest_len <= configs.threshold_shortest_len:
        #         return True
        #     else:
        #         return False
        # else:
        #     return True
    else:
        return False

def get_check_vis_area(point1, point2, area_radius=2):
    x_min = min(point1[0], point2[0]) - area_radius
    x_max = max(point1[0], point2[0]) + area_radius
    y_min = min(point1[1], point2[1]) - area_radius
    y_max = max(point1[1], point2[1]) + area_radius

    return [ x_min, x_max, y_min, y_max ]


def graph_check_connection_new_v2(point1, point2, points_info_dict, graph_projection, length_list, mesh_3d_dict, vertex_map_dict, check_area_radius=3):
    point1_info = points_info_dict[point1][0]
    point2_info = points_info_dict[point2][0]

    # come from same triangle 
    flag = check_easy_case_v2(point1_info, point2_info, mesh_3d_dict, points_info_dict)
    
    if flag == 1:
        return True
    elif flag == 0:
        return False
    
    x1, y1 = nearest_vertex(point2, point1_info[1])
    one_point1 =  vertex_map_dict[(x1, y1)]
    # print(one_point1[0])
    one_point1 = (one_point1[0], one_point1[1], one_point1[2])   # item_to_tuple(list(one_point1), is_polygon=False)
    x2, y2 = nearest_vertex(point1, point2_info[1])
    one_point2 = vertex_map_dict[(x2, y2)]
    one_point2 = (one_point2[0], one_point2[1], one_point2[2])

    # check_area = get_check_vis_area(point1, point2, area_radius=check_area_radius)

    if nx.has_path(graph_projection, str(one_point1), str(one_point2)):
 
        # shortest_len = nx.shortest_path_length(graph_projection, source=str(one_point1), target=str(one_point2))
        shortest_path = nx.shortest_path(graph_projection, source=str(one_point1), target=str(one_point2))
        shortest_path = shortest_path[1:-2]

        # start_point = one_point1
        # end_point = one_point2
        # print(start_point, end_point)

        path_points = list()
        path_edges = list()

        for item in shortest_path:
            item = eval(item)
            # print("-"*100)
            # print(item)
            if isinstance(item[0], tuple):
                path_edges.append(item)
                continue
            else:
                c_point = item  # [:2]
                path_points.append(c_point)     
        # print("-"*100)       
        # (-1.4155736466962934, 2.6564398914517713, 3.6935309643617176)
        # ["backface", [], current_surface, current_surface_depth, obj_count] 
        # print(path_edges)
        # print("-"*100)
        # print(graph_projection[str(path_edges[0][0])][str(path_edges[0][1])])
        # pdb.set_trace()
        if len(path_points) > 10:
            return False
            print(len(path_points))
        return check_all_triangles_visibility(path_points, points_info_dict, graph_projection)

    else:
        return False


def graph_check_connection_new_v3(point1, point2, points_info_dict, graph_projection, length_list, mesh_3d_dict, vertex_map_dict, triangles_visible_buffer, check_area_radius=3):
    point1_info = points_info_dict[point1][0]
    point2_info = points_info_dict[point2][0]

    # come from same triangle 
    flag = check_easy_case_v2(point1_info, point2_info, mesh_3d_dict, points_info_dict)
    
    if flag == 1:
        return True
    elif flag == 0:
        return False
    
    x1, y1 = nearest_vertex(point2, point1_info[1])
    one_point1 =  vertex_map_dict[(x1, y1)]
    # print(one_point1[0])
    one_point1 = (one_point1[0], one_point1[1], one_point1[2])   # item_to_tuple(list(one_point1), is_polygon=False)
    x2, y2 = nearest_vertex(point1, point2_info[1])
    one_point2 = vertex_map_dict[(x2, y2)]
    one_point2 = (one_point2[0], one_point2[1], one_point2[2])

    # check_area = get_check_vis_area(point1, point2, area_radius=check_area_radius)

    if nx.has_path(graph_projection, str(one_point1), str(one_point2)):
 
        # shortest_len = nx.shortest_path_length(graph_projection, source=str(one_point1), target=str(one_point2))
        shortest_path = nx.shortest_path(graph_projection, source=str(one_point1), target=str(one_point2))
        shortest_path = shortest_path[1:-2]

        # start_point = one_point1
        # end_point = one_point2
        # print(start_point, end_point)

        path_points = list()
        path_edges = list()

        for item in shortest_path:
            item = eval(item)
            # print("-"*100)
            # print(item)
            if isinstance(item[0], tuple):
                path_edges.append(item)
                continue
            else:
                c_point = item  # [:2]
                path_points.append(c_point)     
        # print("-"*100)       
        # (-1.4155736466962934, 2.6564398914517713, 3.6935309643617176)
        # ["backface", [], current_surface, current_surface_depth, obj_count] 
        # print(path_edges)
        # print("-"*100)
        # print(graph_projection[str(path_edges[0][0])][str(path_edges[0][1])])
        # pdb.set_trace()
        
        # if configs.use_threshold_shortest:
        #     if configs.threshold_shortest_len == -1:
        #         # has path --- > ob (self-occ are ignored)
        #         return True
        #     if len(path_points) > configs.threshold_shortest_len:
        #         return False
        #         # print(len(path_points))
        #     return check_all_triangles_visibility_v2(path_points, points_info_dict, graph_projection, triangles_visible_buffer)
        # else:
        #     return False
        
        if len(path_points) > 10:
            return False
            print(len(path_points))
        return check_all_triangles_visibility_v2(path_points, points_info_dict, graph_projection, triangles_visible_buffer)

    else:
        return False

def check_all_triangles_visibility_v2(path_points, points_info_dict, graph_projection, triangles_visible_buffer, check_area=[], tolerance=1e-25):

    # x_min, x_max, y_min, y_max = check_area


    for node_3d in path_points:
        node_3d_graph = str(node_3d)

        # node_3d_info = graph_projection.nodes[node_3d_graph]["3d_triangle_info"][0]

        # print(len(node_3d_info))
        # print(node_3d_info[0])
        # print("-"*100)
        # print(node_3d_info[1])

        node_3d_infos = graph_projection.nodes[node_3d_graph]["3d_triangle_info"]

        for node_3d_info in node_3d_infos:

            if node_3d_info[0] == "backface":
                return False
            
            # cannot make sure if all small faces are visible (covered by other triangles)
            # configs.count_small_faces
            # if node_3d_info[0] == "small_face":
            #     return False    

            current_triangle_projected_points = node_3d_info[1]
            current_surface = node_3d_info[2]
            # print(len(current_triangle_projected_points))
            polygons_tuple = item_to_tuple(current_surface, is_polygon=True)
            if triangles_visible_buffer[polygons_tuple]:
                continue
            else:
                return False
    
    return True

def are_points_equal(points1, points2, tolerance=1e-25):
    """
    Check if two sets of points are approximately equal.
    """
    # print(points1)
    # print(points2)
    # pdb.set_trace()

    # Decimal
    # 334.97503853095179765513286849074178
    # 334.79499311407402552713587168361825

    # np.float
    # 334.9750385309518   168.27443195889157  334.79499311407403
    # 334.79499311407403  168.251900361352    334.65073031328586

    if len(points1) != len(points2):
        return False
    
    if points1 == points2:
        return True
    else:
        return False

    # or use np.allclose 1e-25
    for p1, p2 in zip(points1, points2):
        if not np.allclose(p1, p2, atol=tolerance):
            return False

    return True

def check_all_triangles_visibility(path_points, points_info_dict, graph_projection, check_area=[], tolerance=1e-25):

    # x_min, x_max, y_min, y_max = check_area


    for node_3d in path_points:
        node_3d_graph = str(node_3d)

        # node_3d_info = graph_projection.nodes[node_3d_graph]["3d_triangle_info"][0]

        # print(len(node_3d_info))
        # print(node_3d_info[0])
        # print("-"*100)
        # print(node_3d_info[1])

        node_3d_infos = graph_projection.nodes[node_3d_graph]["3d_triangle_info"]

        for node_3d_info in node_3d_infos:

            if node_3d_info[0] == "backface":
                return False
            
            # cannot make sure if all small faces are visible (covered by other triangles)
            # configs.count_small_faces
            # if node_3d_info[0] == "small_face":
            #     return False    

            current_triangle_projected_points = node_3d_info[1]
            current_surface = node_3d_info[2]
            # print(len(current_triangle_projected_points))

            # continue
            for point in current_triangle_projected_points:

                # c_x, c_y = point
                # if not (x_min <= c_x <= x_max and y_min <= c_y <= y_max):
                #     continue

                final_traingle_info = points_info_dict[point][0][1]

                # option 1: Not fully visible, make sure all path's triangles' projected points are not replaced.
                # option 2: For a big triangle (conntecd area not replaced, far area has replaced), 
                #           option1 still match definition but not so ..., 
                #           maybe we just need to check the closest trangle proejected point (1 or 2 or 3?) is replaced or not.
                if not are_points_equal(current_surface, final_traingle_info, tolerance=tolerance):
                    return False
            
            # do not check all triangles.         
            break
            # pdb.set_trace()
    
    return True


def check_z_fighting(point1, point2, points_info_dict, graph_projection, length_list):
    # one way to check z_fighting in self obj judgement:
    # if this two point are not directly (path_length=0) connectted, they must be cutted by their own traingle edge
    # if z_fighting, skip and pass?  ---> check_z_fighting
    # or still check their connection by nx.has_path? ---> graph_check_connection
    point1_info = points_info_dict[point1][0]
    point2_info = points_info_dict[point2][0]

    len_list_in = exist_lists_intersection(point1_info[2], point2_info[2])
    if len_list_in > 0:
        # == 3 means same surface
        # == 2 or 1 means connected, even have z_fighting, won't influence results
        return True
    else:
        mesh_2d = point1_info[1]
        check_line = (point1, point2)
        polygon1, polygon2 = return_right_mesh_edge(mesh_2d, check_line)

        if polygon1 == 0 and polygon2 == 0:
            # some other problems like z_fighting occurred
            return True
        else:
            mesh_2d = point2_info[1]
            polygon1, polygon2 = return_right_mesh_edge(mesh_2d, check_line)
            if polygon1 == 0 and polygon2 == 0:
                # and
                return True
            else:
                # two edge cutted
                return graph_check_connection(point1, point2, points_info_dict, graph_projection, length_list)


def process_same_depth(item, point, points_info_dict, orientation_label, labels_info_dict):
    # # basically won't happen
    # option 1: don't process, ignore
    # pass
    # option 2: near surface
    # print(points_info_dict[item])
    dis1 = get_surface(points_info_dict[item][0][2])
    # print(points_info_dict[item])
    dis1 = dis2surface(dis1)
    dis2 = get_surface(points_info_dict[point][0][2])
    dis2 = dis2surface(dis2)
    if dis1 <= dis2:
        # orientation_label[point].append((item, point))
        # img_output[item[0], item[1], :] = self_obp_color
        # labels_info_dict = update_point_label(item, OBP_CLASS[2], labels_info_dict)
        count = 1
    else:
        # checked_list.append(item)
        orientation_label[point].append((point, item))
        # img_output[point[0], point[1], :] = self_obp_color
        labels_info_dict = update_point_label(point, OBP_CLASS[1], labels_info_dict)
    # option 3: near neighbors depth
    # neighbors1 = get_point_neighbors(item, img_output.shape, 4)
    # neighbors1_depth = sum([points_info_dict[nn] for nn in neighbors1])
    #
    # neighbors2 = get_point_neighbors(point, img_output.shape, 4)
    # neighbors2_depth = sum([points_info_dict[nn] for nn in neighbors2])
    #
    # if neighbors1_depth <= neighbors2_depth:
    #     orientation_label[point].append((item, point))
    #     img_output[item[0], item[1], :] = self_obp_color
    #     labels_info_dict = update_point_label(item, OBP_CLASS[2], labels_info_dict)
    # else:
    #     orientation_label[point].append((point, item))
    #     img_output[point[0], point[1], :] = white_point
    #     labels_info_dict = update_point_label(point, OBP_CLASS[1], labels_info_dict)

    # option 4: always choose x,y smaller one
    # x1, y1 = item
    # x2, y2 = point
    # if x1 + y1 < x2 + y2:
    #     orientation_label[point].append((item, point))
    #     img_output[item[0], item[1], :] = self_obp_color
    #     labels_info_dict = update_point_label(item, OBP_CLASS[2], labels_info_dict)
    # else:
    #     orientation_label[point].append((point, item))
    #     img_output[point[0], point[1], :] = white_point
    #     labels_info_dict = update_point_label(point, OBP_CLASS[1], labels_info_dict)

    return orientation_label, labels_info_dict  # , img_output


def compare_one_pair(pair):
    y1, x1 = pair[0]
    y2, x2 = pair[1]
    if x1 == x2:
        if y1 > y2:
            return 270  # -90
        else:
            return 90
    else:
        # y1==y2
        if x1 > x2:
            return 180
        else:
            return 0


def get_angle_value(angles_info):
    # angle with positive x direction
    # opencv h,w --> y,x
    # deal with 4n, not 8n
    # There will be several special cases in the actual calculation:
    # the length is 2, the two directions are exactly opposite, we will get 0;
    # the length is 4, all four directions are two-two opposite;
    # we can give a special value is used to distinguish 0 from the normal direction like 360.
    if len(angles_info) == 1:
        return compare_one_pair(angles_info[0])
    elif len(angles_info) == 4:
            return 360
    elif len(angles_info) == 2:
        d1 = angles_info[0][1]
        d2 = angles_info[1][1]
        if d1[0] == d2[0] or d1[1] == d2[1]:
            return 360

        # we can process 270 and 0 directly
        x, y = 0, 0
        angle = 0
        for item in angles_info:
            y1, x1 = item[0]
            y2, x2 = item[1]
            # move to origin and add vectors
            y += y2-y1
            x += x2-x1
            if x1 == x2:
                angle += 90
            else:
                # y1==y2
                if x1 > x2:
                    angle += 180
                else:
                    angle += 0

        angle = angle / len(angles_info)
        if y < 0:
            angle = - angle
        angle = (angle + 360) % 360
        return angle
    else:
        # length = 3
        # we create the neighbors list in an order, so for length 2 or 3, we need check 0,1 or -1, -2
        d1 = angles_info[0][1]
        d2 = angles_info[1][1]
        d3 = angles_info[-2][1]
        d4 = angles_info[-1][1]
        if d1[0] == d2[0] or d1[1] == d2[1]:
            return compare_one_pair(angles_info[2])
        elif d3[0] == d4[0] or d3[1] == d4[1]:
            return compare_one_pair(angles_info[0])


def barycentric_coordinates(point, a, b, c):
    # T = np.hstack(((a - c)[:, None], (b - c)[:, None]))
    # alpha, beta = np.linalg.lstsq(T, point - c)[0]
    # gamma = 1.0-alpha-beta
    # print(alpha, beta)

    gamma = ( (a[1] - b[1])*point[0] + (b[0]-a[0])*point[1] + a[0]*b[1]-b[0]*a[1] ) / \
            ( (a[1] - b[1])*c[0]     + (b[0]-a[0])*c[1]     + a[0]*b[1]-b[0]*a[1] )
    beta =  ( (a[1] - c[1])*point[0] + (c[0]-a[0])*point[1] + a[0]*c[1]-c[0]*a[1] ) / \
            ( (a[1] - c[1])*b[0]     + (c[0]-a[0])*b[1]     + a[0]*c[1]-c[0]*a[1] )
    alpha = 1 - beta - gamma

    # v0, v1, v2 = a,b,c
    # denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
    # w0 = ((v1[1] - v2[1]) * (x - v2[0]) + (v2[0] - v1[0]) * (y - v2[1])) / denom
    # w1 = ((v2[1] - v0[1]) * (x - v2[0]) + (v0[0] - v2[0]) * (y - v2[1])) / denom
    # w2 = 1 - w0 - w1
    # # print(alpha, beta, gamma)
    return alpha, beta, gamma


def get_depth_bc(x, y, vertices):
    """
    Compute the depth value of a point inside a triangle defined by its vertices.

    Args:
    x, y (float): The coordinates of the point.
    vertices (list of tuple): The vertices of the triangle, each defined as (x, y, z).

    Returns:
    float: The depth value of the point inside the triangle.
    """
    # Compute the barycentric coordinates of the point.
    v0, v1, v2 = vertices
    denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
    w0 = ((v1[1] - v2[1]) * (x - v2[0]) + (v2[0] - v1[0]) * (y - v2[1])) / denom
    w1 = ((v2[1] - v0[1]) * (x - v2[0]) + (v0[0] - v2[0]) * (y - v2[1])) / denom
    w2 = 1 - w0 - w1

    # Compute the depth value of the point using the barycentric coordinates and the depth values of the vertices.
    # depth = w0 * v0[2] + w1 * v1[2] + w2 * v2[2]
    depth = 1 / (w0 / v0[2] + w1 / v1[2] + w2 / v2[2])
    return depth


def line_equation(point1, point2):
    x1, y1 = point1[0], point1[1]
    x2, y2 = point2[0], point2[1]

    # k = (y2-y1)/(x2-x1)
    return y2 - y1, x1 - x2, x2 * y1 - x1 * y2


def find_true_border_line(point1, vertex_2d):
    """
    We know this point is a border point
    """
    border_lines = list(itertools.combinations(vertex_2d, 2))
    y = list()
    for border_line in border_lines:
        # interpolation
        a, b, c = line_equation((border_line[0][1], border_line[0][0]), (border_line[1][1], border_line[1][0]))
        # print((a*point1[0]+c)/(-b), point1[1])
        if b == 0 and point1[0] == border_line[0][1]:    # for item in polypoints:
    #     # project 3D points
    #     try:
    #         img[item[0], item[1], 2] = 255
    #         img[item[0], item[1], 1] = 255
    #         img[item[0], item[1], 0] = 255
    #     except:
    #         print(item)
    # cv2.imwrite("./p4_skiimage.png", img)
            point_1 = (border_line[0][1], border_line[0][0])
            point_2 = (border_line[1][1], border_line[1][0])
            return point_1, point_2
        elif b == 0 and point1[0] != border_line[0][1]:
            continue

        y.append(np.abs((a * point1[0] + c) / (-b) - point1[1]))
        if (a * point1[0] + c) / (-b) == point1[1]:
            # print((border_line[0][1],border_line[0][0]), (border_line[1][1],border_line[1][0]))
            point_1 = (border_line[0][1], border_line[0][0])
            point_2 = (border_line[1][1], border_line[1][0])
            return point_1, point_2
    else:
        # print(y, point1)
        # float problem: float a != float a
        min_index = y.index(min(y))
        point_1 = (border_lines[min_index][0][1], border_lines[min_index][0][0])
        point_2 = (border_lines[min_index][1][1], border_lines[min_index][1][0])
        return point_1, point_2


def item_to_tuple(list1, is_polygon=True):
    results = list()
    for item in list1:
        if is_polygon:
            results.append((item[1], item[0]))
        else:
            results.append((item[0], item[1], item[2]))
    return tuple(results)


def get_vertex_depth_value(vertex_2d, vertex_3d, depth_type):
    depth_dict = dict()
    for i in range(len(vertex_2d)):
        if depth_type == "z_buffer":
            depth_dict[(vertex_2d[i][1], vertex_2d[i][0])] = depth_z_value(vertex_3d[i][-1])
        else:
            depth_dict[(vertex_2d[i][1], vertex_2d[i][0])] = dis2camera(vertex_3d[i])
    return depth_dict


def get_depth_value(border_point, vertex_2d, depth_dict):
    # far
    border_depth = 0.0

    if border_point in list(depth_dict.keys()):
        return border_point, depth_dict[border_point]

    # border_lines = list(itertools.combinations(vertex_2d, 2))
    # border_lines = [(vertex_2d[0], vertex_2d[1]), (vertex_2d[1], vertex_2d[2]), (vertex_2d[0], vertex_2d[2])]

    # interpolation
    point1, point2 = find_true_border_line(border_point, vertex_2d)
    depth_1 = depth_dict[point1]
    depth_2 = depth_dict[point2]
    border_depth = interpolation_depth(border_point, point1, point2, depth_1, depth_2)

    if border_depth == 0.0:
        logging.info(str(border_point) + "\t in \t " + str(vertex_2d))
        logging.warning("The above projected points do not have depth, check results")
        # print(border_point, border_depth)
        # print("some projected points do not have depth, check results")
        return border_point, border_depth
    else:
        return border_point, border_depth

def dot(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]

def subtract(v1, v2):
    return (v1[0] - v2[0], v1[1] - v2[1])

def perpendicular(v):
    return (-v[1], v[0])

def interval_overlap(interval1, interval2):
    return interval1[1] >= interval2[0] and interval2[1] >= interval1[0]

def project_triangle_onto_axis(triangle, axis):
    dot_products = [dot(vertex, axis) for vertex in triangle]
    return min(dot_products), max(dot_products)

def triangles_intersect(triangle1, triangle2):
    axes = [
        perpendicular(subtract(triangle1[0], triangle1[1])),
        perpendicular(subtract(triangle1[1], triangle1[2])),
        perpendicular(subtract(triangle2[0], triangle2[1])),
        perpendicular(subtract(triangle2[1], triangle2[2])),
    ]

    for axis in axes:
        projection1 = project_triangle_onto_axis(triangle1, axis)
        projection2 = project_triangle_onto_axis(triangle2, axis)

        if not interval_overlap(projection1, projection2):
            return False

    return True


def update_point_label(point, point_label, labels_info_dict):
    if point in list(labels_info_dict.keys()):
        if point_label not in labels_info_dict[point]:
            labels_info_dict[point].append(point_label)
    else:
        labels_info_dict[point] = [point_label]
    return labels_info_dict


def get_3_colors(list1):
    colors_id = ncolors(3)
    OBP_CLASS = ["background_obj_obp", "different_obj_obp", "self_obp"]

    if len(list1) == 1:
        return colors_id[OBP_CLASS.index(list1[0])]
    else:
        if "background_obj_obp" in list1:
            return colors_id[0]
        elif "different_obj_obp" in list1:
            return colors_id[0]


def update_depth_dict_v2(point, point_depth, final_depth_dict, covered_points, polygons, point_3d_list, obj_count):
    if int(point[0]) == point[0] and int(point[1]) == point[1]:
        point = (int(point[0]), int(point[1]))
        # paint_one_point(point, img, color=[255, 255, 255])
        if len(final_depth_dict[point]) == 0:
            final_depth_dict[point] = [[point_depth, polygons, point_3d_list, obj_count]]
        else:
            # covered by other projection points
            covered_points.append(point)
            depth_new = point_depth
            # if depth_type == "z_buffer":
            if final_depth_dict[point][0][0] > depth_new:
                final_depth_dict[point] = [[depth_new, polygons, point_3d_list, obj_count]]

            elif final_depth_dict[point][0][0] == depth_new:
                final_depth_dict[point].append([depth_new, polygons, point_3d_list, obj_count])

            # else:
            #     if final_depth_dict[point][0][0] > depth_new:
            #         final_depth_dict[point] = [[depth_new, polygons, point_3d_list, obj_count]]
            #
            #     elif final_depth_dict[point][0][0] == depth_new:
            #         final_depth_dict[point].append([depth_new, polygons, point_3d_list, obj_count])

    return final_depth_dict, covered_points # , # empty_projection #, img


def update_depth_dict_v5(point, point_depth, final_depth_dict, covered_points, polygons, point_3d_list, obj_count, triangles_visible_buffer):

    if int(point[0]) == point[0] and int(point[1]) == point[1]:
        point = (int(point[0]), int(point[1]))

        if len(final_depth_dict[point]) == 0:
            final_depth_dict[point] = [[point_depth, polygons, point_3d_list, obj_count]]
        else:
            # covered by other projection points
            covered_points.append(point)
            depth_new = point_depth
            # if depth_type == "z_buffer":
            if final_depth_dict[point][0][0] > depth_new:
                polygons_tuple = item_to_tuple(final_depth_dict[point][0][1], is_polygon=True)
                triangles_visible_buffer[polygons_tuple] = False  

                final_depth_dict[point] = [[depth_new, polygons, point_3d_list, obj_count]]


            elif final_depth_dict[point][0][0] == depth_new:
                final_depth_dict[point].append([depth_new, polygons, point_3d_list, obj_count])

            # else:
            #     if final_depth_dict[point][0][0] > depth_new:
            #         final_depth_dict[point] = [[depth_new, polygons, point_3d_list, obj_count]]
            #
            #     elif final_depth_dict[point][0][0] == depth_new:
            #         final_depth_dict[point].append([depth_new, polygons, point_3d_list, obj_count])

    return final_depth_dict, covered_points, triangles_visible_buffer


def find_intersection(triangle1, triangle2):
    intersection_points = []

    for edge1 in get_edges(triangle1):
        for edge2 in get_edges(triangle2):
            intersection_point = find_intersection_point(edge1, edge2)
            if intersection_point:
                intersection_points.append(intersection_point)

    # The intersection triangle vertices are the intersection points
    intersection_triangle = intersection_points[:3]

    return intersection_triangle


def get_edges(triangle):
    # Helper function to get the edges of a triangle
    edges = []
    for i in range(3):
        edges.append((triangle[i], triangle[(i + 1) % 3]))
    return edges


def find_intersection_point(edge1, edge2):
    # Helper function to find the intersection point of two edges
    x1, y1 = edge1[0]
    x2, y2 = edge1[1]
    x3, y3 = edge2[0]
    x4, y4 = edge2[1]

    # Use the line intersection formula
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denominator != 0:
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator

        # Check if the intersection point is within the edges
        if 0 <= t <= 1 and 0 <= u <= 1:
            intersection_x = x1 + t * (x2 - x1)
            intersection_y = y1 + t * (y2 - y1)
            return (intersection_y, intersection_x)

    return None

    
def linear_interpolation(p1, p2, t):
    return (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))

def generate_boundary_points(triangle, step=0.1):
    boundary_points = []

    for i in np.arange(0, 1 + step, step):
        edge_points = [
            linear_interpolation(triangle[0], triangle[1], i),
            linear_interpolation(triangle[1], triangle[2], i),
            linear_interpolation(triangle[2], triangle[0], i)
        ]

        for point in edge_points:
            rounded_point = (round(point[0]), round(point[1]))
            if rounded_point not in boundary_points:
                boundary_points.append(rounded_point)

    return boundary_points

def update_depth_dict_v3(polygon, polygon_iter_dict, final_depth_dict, covered_points, polygons, point_3d_list, obj_count):

    covered_polygon_list = list()
    for info in polygon:
        point, point_depth = info[0], info[1]
        if int(point[0]) == point[0] and int(point[1]) == point[1]:
            point = (int(point[0]), int(point[1]))
            # paint_one_point(point, img, color=[255, 255, 255])
            if len(final_depth_dict[point]) == 0:
                final_depth_dict[point] = [[point_depth, polygons, point_3d_list, obj_count]]
            else:
                # covered by other projection points
                covered_points.append(point)
                depth_new = point_depth
                # if depth_type == "z_buffer":

                # previous_info

                previous_info = final_depth_dict[point]
                # covered_polygon_list.append(previous_info)

                if final_depth_dict[point][0][0] > depth_new:
                    final_depth_dict[point] = [[depth_new, polygons, point_3d_list, obj_count]]
                    covered_polygon_list.append(previous_info)

                elif final_depth_dict[point][0][0] == depth_new:
                    final_depth_dict[point].append([depth_new, polygons, point_3d_list, obj_count])
                else:
                    continue


                # else:
                #     if final_depth_dict[point][0][0] > depth_new:
                #         final_depth_dict[point] = [[depth_new, polygons, point_3d_list, obj_count]]
                #
                #     elif final_depth_dict[point][0][0] == depth_new:
                #         final_depth_dict[point].append([depth_new, polygons, point_3d_list, obj_count])


    # covered_polygon_list = list(set(covered_polygon_list))
    current_polygon_points = polygon_iter_dict[str(polygons)]

    covered_points = list()

    # covered_polygon_points_list = [polygon_iter_dict[str(item[1])] for item in covered_polygon_list]
    if len(covered_polygon_list) != 0:

        for item in covered_polygon_list:
            previous_polygon_points = polygon_iter_dict[str(item[0][1])]
            # print(current_polygon_points)

            check = triangles_intersect(polygons, item[0][1])
            if check:
                # print(polygons)
                # print(item[0][1])
                # print("-"*70)
                pp = find_intersection(polygons, item[0][1])

                polygon = np.asarray(pp)
                if len(polygon) != 3:
                    continue


                bb_points = generate_boundary_points(polygon, step=0.001)

                covered_points.extend(bb_points)

                # vertex_row_coords, vertex_col_coords = polygon.T
                # fill_row_coords, fill_col_coords = draw.polygon(
                #     vertex_row_coords, vertex_col_coords, (1080, 1080))

                # if len(fill_row_coords) == 0 or len(fill_col_coords) == 0:
                #     continue

                # current_polygon_points = list()
                # for point_index in range(len(fill_col_coords)):
                #     current_h = fill_col_coords[point_index]
                #     current_w = fill_row_coords[point_index]                    
                #     current_polygon_points.append((current_h, current_w))

                # covered_points.extend(current_polygon_points)
         
            #     print(pp)
            #     print("-"*70)            
            # print(111)
            # pdb.set_trace()
            # intersection = list(set(current_polygon_points) & set(previous_polygon_points))
            # print(intersection)

            # if len(intersection) != len(current_polygon_points) and len(intersection) != len(previous_polygon_points):
            #     # partily visible
            #     covered_points.extend(intersection)
            # elif len(intersection) == len(current_polygon_points):
            #     covered_points.extend(intersection)
            # elif len(intersection) == len(previous_polygon_points):
            #     # check = triangles_intersect(polygons, item[0][1])
            #     # if check:
            #     covered_points.extend(intersection)
            #     # print(1111)
            # else:
            #     print(22222)


    return final_depth_dict, covered_points  # , # empty_projection #, img


def update_depth_dict_v4(polygon, polygon_iter_dict, final_depth_dict, 
                    covered_points, polygons, point_3d_list, 
                    obj_count, pixel_relatedFace_dict):

    covered_polygon_list = list()
    for info in polygon:
        point, point_depth = info[0], info[1]
        polygons_tuple = item_to_tuple(polygons, is_polygon=True)
        point_3d_list_tuple = item_to_tuple(point_3d_list, is_polygon=False)
        if int(point[0]) == point[0] and int(point[1]) == point[1]:
            point = (int(point[0]), int(point[1]))
            # paint_one_point(point, img, color=[255, 255, 255])
            if len(final_depth_dict[point]) == 0:
                final_depth_dict[point] = [[point_depth, polygons_tuple, point_3d_list_tuple, obj_count]]

                try:
                    pixel_relatedFace_dict[point].append((polygons_tuple, point_3d_list_tuple, obj_count))
                except:
                    pixel_relatedFace_dict[point] = [(polygons_tuple, point_3d_list_tuple, obj_count)] 


            else:
                # covered by other projection points
                covered_points.append(point)
                depth_new = point_depth
                # if depth_type == "z_buffer":

                # previous_info

                previous_info = final_depth_dict[point]

                pre_tuple1 = previous_info[0][1] # item_to_tuple(previous_info[0][1], is_polygon=True)
                pre_tuple2 = previous_info[0][2] # item_to_tuple(previous_info[0][2], is_polygon=False)

                pre_tuple = (pre_tuple1, pre_tuple2, previous_info[0][3])

                # covered_polygon_list.append(previous_info)

                if final_depth_dict[point][0][0] > depth_new:
                    final_depth_dict[point] = [[depth_new, polygons_tuple, point_3d_list_tuple, obj_count]]
                    covered_polygon_list.append(previous_info)

                    # print("-"*100)
                    # print(final_depth_dict[point])
                    # print("-"*100)
                    # print(pixel_relatedFace_dict[point])
                    # print("-"*100)

                    pixel_relatedFace_dict[point].remove(pre_tuple)
                    pixel_relatedFace_dict[point].append((polygons_tuple, point_3d_list_tuple, obj_count))

                    # print("-"*100)
                    # print(pixel_relatedFace_dict[point])
                    # print("-"*100)


                elif final_depth_dict[point][0][0] == depth_new:
                    final_depth_dict[point].append([depth_new, polygons_tuple, point_3d_list_tuple, obj_count])
                    # pixel_relatedFace_dict[point].remove(pre_tuple)
                    pixel_relatedFace_dict[point].append((polygons_tuple, point_3d_list_tuple, obj_count))

                else:
                    continue

        # print(point)

        # point = [int(point[0]), int(point[1])]

        # try:
        #     pixel_relatedFace_dict[point].append((polygons, point_3d_list, obj_count))
        # except:
        #     pixel_relatedFace_dict[point] = [(polygons, point_3d_list, obj_count)] 

 

    return final_depth_dict, pixel_relatedFace_dict  # , # empty_projection #, img


def update_depth_dict(point, point_depth, final_depth_dict, covered_points, empty_projection, depth_type,
                      polygons, point_3d_list, obj_count):
    if int(point[0]) == point[0] and int(point[1]) == point[1]:
        point = (int(point[0]), int(point[1]))
        empty_projection += 1
        # paint_one_point(point, img, color=[255, 255, 255])
        if len(final_depth_dict[point]) == 0:
            final_depth_dict[point] = [[point_depth, polygons, point_3d_list, obj_count]]

        else:
            # covered by other projection points
            covered_points.append(point)
            depth_new = point_depth
            # if depth_type == "z_buffer":
            if final_depth_dict[point][0][0] > depth_new:
                final_depth_dict[point] = [[depth_new, polygons, point_3d_list, obj_count]]

            elif final_depth_dict[point][0][0] == depth_new:
                final_depth_dict[point].append([depth_new, polygons, point_3d_list, obj_count])

            # else:
            #     if final_depth_dict[point][0][0] > depth_new:
            #         final_depth_dict[point] = [[depth_new, polygons, point_3d_list, obj_count]]
            #
            #     elif final_depth_dict[point][0][0] == depth_new:
            #         final_depth_dict[point].append([depth_new, polygons, point_3d_list, obj_count])

    return final_depth_dict, covered_points, empty_projection #, img


def cal_slope(p1, p2, return_d=True, return_k=True):
    # pi is a tuple:(xi, yi)
    # d > 0 : p1->p2 towards down
    if return_k:
        if p1[0] == p2[0]:
            k = None
        else:
            k = (p2[1] - p1[1]) * 1.0 / (p2[0] - p1[0])
    else:
        k = None
    if return_d:
        if p2[1] != p1[1]:
            d = (p2[1] - p1[1]) / abs(p2[1] - p1[1])
        else:
            d = 0
    else:
        d = None
    return k, d


def scanline_depth(polygons, point_3d_list, depth_type, obj_count, final_depth_dict, img):
    image_size = img.shape
    polygons_tuple = item_to_tuple(polygons, is_polygon=True)
    point_3d_list_tuple = item_to_tuple(point_3d_list, is_polygon=False)

    def find_border_points(last_p, current_p, border_dict):
        results = [(last_p[1], last_p[0]), (current_p[1], current_p[0])]
        x1, y1 = last_p
        x2, y2 = current_p

        k, d = cal_slope(last_p, current_p)

        if k == 0:
            if int(y1) == y1:
                x_start = min(x1, x2)
                x_end = max(x1, x2)
                if int(y1) < image_size[0]:
                    for i in range(int(x_start), int(x_end) + 1):
                        results.append((int(y1), i))
                        border_dict[(int(y1), i)] = [(last_p[1], last_p[0]), (current_p[1], current_p[0])]
                return results
            else:
                return results

        else:
            if d > 0:
                x_start, y_start = x1, y1
                x_end, y_end = x2, y2
            else:
                x_start, y_start = x2, y2
                x_end, y_end = x1, y1
            y_index_start = int(y_start) + 1
            y_index_end = int(np.ceil(y_end))

            if k is None:
                for y_index in range(y_index_start, y_index_end):
                    scan_border_dict[y_index].append(x_start)
                    scan_border_dict[y_index].sort()
                    border_dict[(y_index, x_start)] = [(last_p[1], last_p[0]), (current_p[1], current_p[0])]
            else:
                k_trans = 1.0 / k
                for y_index in range(y_index_start, y_index_end):
                    x = k_trans * (y_index - y_start) + x_start
                    scan_border_dict[y_index].append(x)
                    scan_border_dict[y_index].sort()
                    border_dict[(y_index, x)] = [(last_p[1], last_p[0]), (current_p[1], current_p[0])]

            return results

    # initialization
    points_num = len(polygons)
    # scan_border: y ->b order_points
    scan_border_dict = defaultdict(list)
    covered_points = list()
    empty_projection = 0
    depth_dict = dict()

    # depth_dict = get_vertex_depth_value(polygons, point_3d_list, depth_type)
    for p in range(len(polygons)):
        current_vertex = (polygons[p][1], polygons[p][0])
        if depth_type == "z_buffer":
            current_vertex_depth = depth_z_value(point_3d_list[p][-1])
            # print(current_vertex_depth)
        else:
            current_vertex_depth = dis2camera(point_3d_list[p])
        depth_dict[current_vertex] = current_vertex_depth
        # all_depth_dict = update_all_depth_dict(current_vertex, current_vertex_depth, all_depth_dict, depth_type)
        final_depth_dict, covered_points, empty_projection = update_depth_dict(current_vertex, current_vertex_depth,
                                                                                    final_depth_dict, covered_points,
                                                                                    empty_projection, depth_type,
                                                                                    polygons_tuple, point_3d_list_tuple,
                                                                                    obj_count)
    results = list()
    border_dict = dict()
    for i in range(points_num):
        last_point = polygons[i - 1]
        current_point = polygons[i]
        results.extend(find_border_points(last_point, current_point, border_dict))
        # print(scan_border_dict)

    results = list(set(results))
    # assert len(results) == 3
    if len(results) != 3:
        for point in results:
            if point not in list(depth_dict.keys()):
                this_depth = interpolation_depth(point, border_dict[point][0], border_dict[point][1],
                                                 depth_dict[border_dict[point][0]],
                                                 depth_dict[border_dict[point][1]])
                depth_dict[point] = this_depth
                # all_depth_dict = update_all_depth_dict(point, this_depth, all_depth_dict, depth_type)
                final_depth_dict, covered_points, empty_projection = update_depth_dict(point, this_depth, final_depth_dict,
                                                                                       covered_points, empty_projection,
                                                                                       depth_type, polygons_tuple,
                                                                                       point_3d_list_tuple, obj_count,)

    for r in scan_border_dict.keys():
        c_list = scan_border_dict[r]
        # assertion error occurred in some scenes. we can comment the code below(asslist is in ge_oc.py):
        # assert len(c_list) % 2 == 0

        nps = int(len(c_list) / 2)
        for i in range(nps):
            c_start = c_list[2 * i]
            c_end = c_list[2 * i + 1]
            if r < image_size[0]:
                border_1_x = int(np.ceil(c_start))
                border_2_x = int(c_end) + 1
                key1 = (r, c_start)
                depth_1 = interpolation_depth(key1, border_dict[key1][0], border_dict[key1][1],
                                              depth_dict[border_dict[key1][0]],
                                              depth_dict[border_dict[key1][1]])
                depth_dict[key1] = depth_1
                # all_depth_dict = update_all_depth_dict(key1, depth_1, all_depth_dict, depth_type)
                final_depth_dict, covered_points, empty_projection = update_depth_dict(key1, depth_1, final_depth_dict,
                                                                                       covered_points, empty_projection,
                                                                                       depth_type, polygons_tuple,
                                                                                       point_3d_list_tuple, obj_count)
                key2 = (r, c_end)
                depth_2 = interpolation_depth(key2, border_dict[key2][0], border_dict[key2][1],
                                              depth_dict[border_dict[key2][0]],
                                              depth_dict[border_dict[key2][1]])
                depth_dict[key2] = depth_2
                # all_depth_dict = update_all_depth_dict(key2, depth_2, all_depth_dict, depth_type)
                final_depth_dict, covered_points, empty_projection = update_depth_dict(key2, depth_2, final_depth_dict,
                                                                                       covered_points, empty_projection,
                                                                                       depth_type, polygons_tuple,
                                                                                       point_3d_list_tuple, obj_count)
                for j in range(border_1_x, border_2_x):
                    results.append((r, j))
                    depth_rj = interpolation_depth((r, j), key1, key2, depth_1, depth_2)
                    depth_dict[(r, j)] = depth_rj
                    # all_depth_dict = update_all_depth_dict((r, j), depth_rj, all_depth_dict, depth_type)
                    final_depth_dict, covered_points, empty_projection = update_depth_dict((r, j), depth_rj,
                                                                                            final_depth_dict,
                                                                                            covered_points,
                                                                                            empty_projection,
                                                                                            depth_type,
                                                                                            polygons_tuple,
                                                                                            point_3d_list_tuple,
                                                                                            obj_count)

    return final_depth_dict, covered_points, empty_projection


def paint_one_point(item, img, color=[255, 255, 255]):
    # try:
    if 0<=item[0]<img.shape[0] and 0<=item[1]<img.shape[1]:
        img[item[0], item[1], 2] = color[2]
        img[item[0], item[1], 1] = color[1]
        img[item[0], item[1], 0] = color[0]
    # except Exception as e:
    #     logging.info(str(item) + " cannot be painted.")
    #     logging.warning("Exception occurred", exc_info=e)
    #     print(e)
    #     print(item)


def calculate_polygon_area(points_list, point_w_3d):
    """
    need area < 0,  x,y  -  y,x
    """
    n = len(points_list)
    if n < 3:
        logging.warning(str(points_list) + " Mesh not a triangle, check data.")
        return False
    # if configs.count_backface:
    #     return True
    # else:
    if configs.is_decimal:
        points_list = [(Decimal(str(point[0])), Decimal(str(point[1]))) for point in points_list]
        area = Decimal(str(0.0))
    else:
        points_list = [(np.float128(point[0]), np.float128(point[1])) for point in points_list]
        area = np.float128(0.0)

    for i in range(n):
        x = points_list[i][0]
        y = points_list[i][1]
        area += x * points_list[(i + 1) % n][1] - y * points_list[(i + 1) % n][0]
    # print(area)
    if area < 0.0  or check_backface_uvsight(point_w_3d):
        return True
    else:
        return False


def check_backface_uvsight(mesh):

    # u = [point_w_3d[1][0] - point_w_3d[0][0], point_w_3d[1][1] - point_w_3d[0][1], point_w_3d[1][2] - point_w_3d[0][2]]
    # v = [point_w_3d[2][0] - point_w_3d[1][0], point_w_3d[2][1] - point_w_3d[1][1], point_w_3d[2][2] - point_w_3d[1][2]]
    # uv = [u[0] * v[0], u[1] * v[1], u[2] * v[2]]
    # vector_sight = [-point_w_3d[0][0], -point_w_3d[0][1], -point_w_3d[0][2]]
    # value_angle = vector_sight[0] * uv[0] + vector_sight[1] * uv[1] + vector_sight[2] * uv[2]

    p = mesh
    a = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]]  # alpha
    b = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]]  # beta

    norm = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]  # normal from product

    # calculate the center of the polygonal face
    center = [(p[0][0] + p[1][0] + p[2][0]) / 3, (p[0][1] + p[1][1] + p[2][1]) / 3,
              (p[0][2] + p[1][2] + p[2][2]) / 3]

    view = [center[0] - 0, center[1] - 0, center[2] - 0]  # calculate the vector between the viewer and the polygon

    if (view[0] * norm[0] + view[1] * norm[1] + view[2] * norm[2]) < 0:    # <= 0:# vector product to check orientation
        return False
    else:
        return True


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


def get_neighbor_nearest_2neighbor(pixel_point, other_neighbor_list):
    results = list()
    distance_check = np.inf
    for item in other_neighbor_list:
        if item != pixel_point:
            # current_dis = point2d_distance(pixel_point, item)
            current_dis = np.abs(pixel_point[0] - item[0]) + np.abs(pixel_point[1] - item[1])
            if current_dis < distance_check:
                distance_check = current_dis
                results = [item]
            elif current_dis == distance_check:
                results.append(item)
            else:
                continue
    return results


def exist_lists_intersection(x, y):
    # x = [(xx.tolist()[0], xx.tolist()[1], xx.tolist()[2]) for xx in list1]
    # y = [(xx.tolist()[0], xx.tolist()[1], xx.tolist()[2]) for xx in list2]
    intersection = list(set(x) & set(y))
    if len(intersection) >= 1:
        return len(intersection) # True
    else:
        return False


def check_point_in_segments(intersection_point, check_line, line2):
    x, y = intersection_point
    x1 = check_line[0][0]
    y1 = check_line[0][1]
    x2 = check_line[1][0]
    y2 = check_line[1][1]

    x3 = line2[0][0]
    y3 = line2[0][1]
    x4 = line2[1][0]
    y4 = line2[1][1]
    if (y1 <= y <= y2 or y2 <= y <= y1) and (y3 <= y <= y4 or y4 <= y <= y3) \
            and (x1 <= x <= x2 or x2 <= x <= x1) and (x3 <= x <= x4 or x4 <= x <= x3):
        return True
    else:
        return False


def check_lines_intersection_4n(check_line, line2):
    x1 = check_line[0][0]
    y1 = check_line[0][1]
    x2 = check_line[1][0]
    y2 = check_line[1][1]
    x3 = line2[0][0]
    y3 = line2[0][1]
    x4 = line2[1][0]
    y4 = line2[1][1]
    # n=4 either x1=x2 or y1=y2
    if x1 == x2 and y1 != y2:
        if (x4 - x3) == 0:
            # not happen x3=x1
            return False
        else:
            x = x1
            k2 = (y4 - y3) / (x4 - x3)
            b2 = y3 - x3 * k2
            y = k2 * x + b2
            if check_point_in_segments((x, y), check_line, line2):
                return (x, y)
            else:
                return False
    elif y1 == y2 and x1 != x2:
        if (x4 - x3) == 0:
            x = x3
            y = y1
            # print(x,y)
            if check_point_in_segments((x, y), check_line, line2):
                return (x, y)
            else:
                return False

        else:
            y = y1
            if y3 == y4:
                # not happen y1=y3
                return False
            k2 = (y4 - y3) / (x4 - x3)
            b2 = y3 - x3 * k2
            # print(k2,b2)
            x = (y - b2) / k2
            # print(x,y)
            if check_point_in_segments((x, y), check_line, line2):
                return (x, y)
            else:
                return False
    # 8n
    elif x1 != x2 and y1 != y2:
        if (x4 - x3) == 0:
            x = x3
            if x1 <= x <= x2 or x2 <= x <= x1:
                k1 = (y2 - y1) / (x2 - x1)
                b1 = y1 - x1 * k1
                y = k1 * x + b1
                if y1 <= y <= y2 or y2 <= y <= y1:
                    return (x, y)
                else:
                    return False
            else:
                return False
        else:
            k1 = (y2 - y1) / (x2 - x1)
            b1 = y1 - x1 * k1
            k2 = (y4 - y3) / (x4 - x3)
            b2 = y3 - x3 * k2
            if k1 == k2:
                return False
            x = (b2 - b1) / (k1 - k2)
            y = k1 * x + b1
            if (y1 <= y <= y2 or y2 <= y <= y1) and (x1 <= x <= x2 or x2 <= x <= x1):
                return (x, y)
            else:
                return False


def return_right_mesh_edge(mesh_2d, check_line):
    # polygons = [(item[1], item[0]) for item in mesh_2d]
    lines = list(itertools.combinations(mesh_2d, 2))
    for line in lines:
        intersection = check_lines_intersection_4n(check_line, line)
        if intersection:
            return (line[0][0], line[0][1]), (line[1][0], line[1][1])
    else:
        return 0, 0


def check_list_diff2(list1):
    if np.max(list1) - np.min(list1) >= 2:
        return True
    else:
        return False


def add_rgba_background(rgba_path, bg_path, save_path, save_name="default"):
    render_img = Image.open(rgba_path)
    # random choose
    # bg_name = "1.png"
    bg_img = Image.open(bg_path)
    # bg_img = bg_img.transpose(Image.FLIP_TOP_BOTTOM)

    assert render_img.mode == 'RGBA'
    assert bg_img.width == render_img.width
    assert bg_img.height == render_img.height

    bg_img.paste(render_img, (0, 0), mask=render_img)

    save_file = save_path + save_name
    # print(bg_img.mode)
    if bg_img.mode != 'RGB':
        bg_img.convert('RGB').save(save_file)
    else:
        bg_img.save(save_file)


def read_npy_dict(data_path):
    points_info = np.load(data_path, allow_pickle=True)
    points_info_dict = dict()
    # skip_list = list()
    for item in points_info:
        points_info_dict[item[0]] = item[1]

    return points_info_dict


def read_info_data(model_pose_info_file, train_db):
    if train_db:
        current_img_name = model_pose_info_file.split("/")[-1].replace(".npy", "")
        folder = "scene/" + current_img_name + "/"
    else:
        current_img_name = model_pose_info_file.split("/")[-1].replace(".npy", "")
        current_img_name = str(int(current_img_name) + 14761 + 1).zfill(7)
        folder = "scene/" + current_img_name + "/"

    folder = configs.image_save_dir + folder

    # read all info
    points_info_dict = read_npy_dict(folder + "/info" + "/points_info.npy")
    labels_info_dict = read_npy_dict(folder + "/info" + "/labels_info.npy")
    orientation_label = read_npy_dict(folder + "/info" + "/orientation_label.npy")
    no_projection_mesh = np.load(folder + "/info" + "/no_projection_meshes.npy", allow_pickle=True)
    final_covered_points = np.load(folder + "/info" + "/covered_points_list.npy", allow_pickle=True)
    # graph_projection = nx.Graph(nx.nx_pydot.read_dot(folder + "/info/graph_projection.dot"))

    return points_info_dict, labels_info_dict,orientation_label,  no_projection_mesh, final_covered_points # , graph_projection


def compare_two_images(image1, image2, img_address, return_colored=True):
    """
    i.e.
    # t, t1, t2 = compare_two_images("obdb/scene/0014829/homotopy_35_newbackface2.png",
    #                "obdb/scene/0014829/homotopy_35_newbackface.png",
    #                "obdb/scene/0014829/", return_colored=True)
    """
    save_name = "compare_" + image1.split("/")[-1].replace(".png", "") + "_and_" + image2.split("/")[-1]
    count_obp_image1 = 0
    count_obp_image2 = 0
    results1 = list()
    results2 = list()
    image1 = cv2.imread(image1)
    image2 = cv2.imread(image2)
    img_original_file = configs.background_img_file
    img_original = cv2.imread(img_original_file)
    img_output = img_original.copy()
    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            if image1[i, j, :].tolist() == image2[i, j, :].tolist():
                continue
            else:
                if image1[i, j, :].tolist() == [0, 0, 0] and image2[i, j, :].tolist() != [0, 0, 0]:
                    print(i, j, image1[i, j, :].tolist(), image2[i, j, :].tolist())
                    count_obp_image2 += 1
                    results2.append((i, j))
                elif image1[i, j, :].tolist() != [0, 0, 0] and image2[i, j, :].tolist() == [0, 0, 0]:
                    print(i, j, image1[i, j, :].tolist(), image2[i, j, :].tolist())
                    count_obp_image1 += 1
                    results1.append((i, j))
                else:
                    # [255,255,255] [0,0,255]
                    continue

                # results.append((i,j))

    if return_colored:
        for p in results1:
            img_output[p[0], p[1], :] = np.array([255, 0, 0])
        for p in results2:
            img_output[p[0], p[1], :] = np.array([0, 0, 255])
        save_file = os.path.join(img_address, save_name)
        cv2.imwrite(save_file, img_output)
        return results1.extend(results2), count_obp_image1, count_obp_image2
    else:
        return results1.extend(results2), count_obp_image1, count_obp_image2


def scanline_polygons(polygons):
    image_size = (1200, 1200)

    def find_border_points(last_p, current_p, border_dict):
        results = [(last_p[1], last_p[0]), (current_p[1], current_p[0])]
        x1, y1 = last_p
        x2, y2 = current_p

        k, d = cal_slope(last_p, current_p)

        if k == 0:
            if int(y1) == y1:
                x_start = min(x1, x2)
                x_end = max(x1, x2)
                if int(y1) < image_size[0]:
                    for i in range(int(x_start), int(x_end) + 1):
                        results.append((int(y1), i))
                        border_dict[(int(y1), i)] = [(last_p[1], last_p[0]), (current_p[1], current_p[0])]
                return results
            else:
                return results

        else:
            if d > 0:
                x_start, y_start = x1, y1
                x_end, y_end = x2, y2
            else:
                x_start, y_start = x2, y2
                x_end, y_end = x1, y1
            y_index_start = int(y_start) + 1
            y_index_end = int(np.ceil(y_end))

            if k is None:
                for y_index in range(y_index_start, y_index_end):
                    scan_border_dict[y_index].append(x_start)
                    scan_border_dict[y_index].sort()
                    border_dict[(y_index, x_start)] = [(last_p[1], last_p[0]), (current_p[1], current_p[0])]
            else:
                k_trans = 1.0 / k
                for y_index in range(y_index_start, y_index_end):
                    x = k_trans * (y_index - y_start) + x_start
                    scan_border_dict[y_index].append(x)
                    scan_border_dict[y_index].sort()
                    border_dict[(y_index, x)] = [(last_p[1], last_p[0]), (current_p[1], current_p[0])]

            return results

    # initialization
    points_num = len(polygons)
    # scan_border: y ->b order_points
    scan_border_dict = defaultdict(list)

    results = list()
    border_dict = dict()
    for i in range(points_num):
        last_point = polygons[i - 1]
        current_point = polygons[i]
        results.extend(find_border_points(last_point, current_point, border_dict))

    for r in scan_border_dict.keys():
        c_list = scan_border_dict[r]
        assert len(c_list) % 2 == 0

        nps = int(len(c_list) / 2)
        for i in range(nps):
            c_start = c_list[2 * i]
            c_end = c_list[2 * i + 1]
            if r < image_size[0]:
                border_1_x = int(np.ceil(c_start))
                border_2_x = int(c_end) + 1

                for j in range(border_1_x, border_2_x):
                    results.append((r, j))
    polygons = list()
    for point in results:
        if int(point[0]) == point[0] and int(point[1]) == point[1]:
            polygons.append(point)
    return list(set(polygons))


def projection_with_depth(model_pose_infos, K, model_dir, img_file, save_file, depth_type):
    img = cv2.imread(img_file)
    img_idmap = img.copy()
    h_size, w_size, channel = img.shape
    no_projection_mesh = list()
    final_covered_points = list()
    dict_keys = [(i, j) for i in range(h_size) for j in range(w_size)]
    points_info_dict = dict.fromkeys(dict_keys, [])
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]
    if configs.camera_extrinsic:
        r = R.from_euler('xyz', [-configs.camera_rx * (np.pi / 180.0), -configs.camera_ry * (np.pi / 180.0),
                                 -configs.camera_rz * (np.pi / 180.0)], degrees=False)
        r = r.as_matrix()
        t = np.array([[-configs.camera_tx], [-configs.camera_ty], [-configs.camera_tz]])
        T = np.concatenate((r, t), axis=1)

    # print(T)
    obj_count = 1
    for index, model_pose_info in enumerate(model_pose_infos):
        model_id = model_pose_info['shape_id']
        trans_vec = np.array(model_pose_info['translation'])
        rot_mat = np.array(model_pose_info['rotation'])
        # still happen...
        if model_pose_info['shape_id'] is None or model_pose_info['rotation'] is None \
                or model_pose_info['translation'] is None or model_pose_info['category_id'] is None:
            continue
        if model_pose_info['category_id'] in configs.category_must_skip:
            continue

        current_obj_dir = os.path.join(model_dir, model_id + "/")
        obj_file = os.path.join(current_obj_dir + 'raw_model.obj')

        # transformation w->c
        mesh_vertices, mesh_surfaces = get_obj_vertex_ali(obj_file)
        if configs.extra_rotation:
            r = R.from_euler(configs.extra_rotation_mode, configs.extra_rotation_angle, degrees=True)
            extra_rot = r.as_matrix()
            mesh_vertices_trans = np.transpose(np.dot(extra_rot, np.transpose(mesh_vertices)))
        else:
            mesh_vertices_trans = np.transpose(np.dot(rot_mat, np.transpose(mesh_vertices)))

        mesh_vertices_trans = mesh_vertices_trans - (-trans_vec)
        if configs.camera_extrinsic:
            mesh_vertices_trans = np.apply_along_axis(add_1_in_position, -1,  mesh_vertices_trans)
            # print(mesh_vertices_trans.shape)
            # print(mesh_vertices_trans[0])
            mesh_vertices_trans = np.apply_along_axis(transfer_wc, -1,  mesh_vertices_trans, T)

        # project 3d->2d
        X, Y, Z = mesh_vertices_trans.T
        h = (-Y) / (-Z) * fy + cy
        w = X / (-Z) * fx + cx
        h = np.minimum(np.maximum(h, 0), h_size - 1)
        w = np.minimum(np.maximum(w, 0), w_size - 1)

        # project 3D mesh
        # for item in tqdm(mesh_surfaces):
        for item in mesh_surfaces:
            point_c_3d = list()
            current_surface = list()
            for j in range(len(item)):
                # obj vertices index start from 1, o3d index start from 0
                # current_x, current_y = np.round(w[item[j]]).astype(int), np.round(h[item[j]]).astype(int)
                # current_x, current_y = w[item[j]], h[item[j]]
                # current_surface.append([current_x, current_y])
                # point_w_3d.append(mesh_vertices_trans[item[j]])

                # current_x, current_y = np.round(w[int(item[j])-1]).astype(int), np.round(h[int(item[j])-1]).astype(int)
                current_x, current_y = w[int(item[j]) - 1], h[int(item[j]) - 1]
                current_surface.append([current_x, current_y])
                point_c_3d.append(mesh_vertices_trans[int(item[j]) - 1])

            if calculate_polygon_area(current_surface, point_c_3d):
                points_info_dict, covered_points, empty_projection = scanline_depth(current_surface, point_c_3d,
                                                                                    depth_type, obj_count,
                                                                                    points_info_dict, img)
                final_covered_points.extend(covered_points)
                # if empty_projection :
                if empty_projection > 0:
                    polygons_tuple = item_to_tuple(current_surface, is_polygon=True)
                    point_3d_list_tuple = item_to_tuple(point_c_3d, is_polygon=False)
                    no_projection_mesh.append((polygons_tuple, point_3d_list_tuple, obj_count))

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

    return projected_points_info_dict, no_projection_mesh, final_covered_points


# def projection_cv2(model_pose_infos, K, ca_m,  model_dir, img_file):
#     img = cv2.imread(img_file)
#     getcontext().prec = 65
#     h_size, w_size, channel = img.shape
#     fx = K[0][0]
#     fy = K[1][1]
#     cx = K[0][2]
#     cy = K[0][2]
#
#     obj_count = 1
#     polypoints = list()
#     for index, model_pose_info in enumerate(model_pose_infos):
#         model_id = model_pose_info['shape_id']
#         trans_vec = np.array(model_pose_info['translation'])
#         rot_mat = np.array(model_pose_info['rotation'])
#         # still happen...
#         if model_pose_info['shape_id'] is None or model_pose_info['rotation'] is None \
#                 or model_pose_info['translation'] is None or model_pose_info['category_id'] is None:
#             continue
#         if model_pose_info['category_id'] in configs.category_must_skip:
#             continue
#
#         current_obj_dir = os.path.join(model_dir, model_id + "/")
#         obj_file = os.path.join(current_obj_dir + 'raw_model.obj')
#
#         # transformation w->c
#         mesh_vertices, mesh_surfaces = get_obj_vertex_ali(obj_file)
#
#         mesh_vertices.astype(np.float32)
#         rot_mat = rot_mat.astype(np.float32)
#         trans_vec = trans_vec.astype(np.float32)
#         print(mesh_vertices[0])
#         mesh_vertices_trans = np.transpose(np.dot(rot_mat, np.transpose(mesh_vertices)))
#         # print(type(mesh_vertices_trans[0][0]))
#         mesh_vertices_trans = mesh_vertices_trans - (-trans_vec)
#         # print(mesh_vertices_trans[0])
#
#         # print(name33)
#
#         mesh_vertices_trans = np.apply_along_axis(add_1_in_position, -1, mesh_vertices_trans)
#         mesh_vertices_trans = np.apply_along_axis(transfer_wc, -1, mesh_vertices_trans, ca_m)
#         # print(mesh_vertices_trans[:, :3].shape)
#         # print(type(mesh_vertices_trans[0][0]))
#
#         screen_vertices = np.array([viewport(ndc(v)) for v in mesh_vertices_trans])
#         # print(screen_vertices.shape)
#         h, w, d = screen_vertices.T
#         # print(d[0])
#         # print(h[0],w[0])
#         # print(name333)
#
#         # project 3d->2d
#         # # http://www.songho.ca/opengl/gl_projectionmatrix.html
#         # h = np.minimum(np.maximum(h, 0), h_size - 1)
#         # w = np.minimum(np.maximum(w, 0), w_size - 1)
#
#
#         # project 3D face and fill poly
#
#         for item in mesh_surfaces:
#             point_c_3d = list()
#             current_surface = list()
#             for j in range(len(item)):
#                 # obj vertices index start from 1, o3d index start from 0
#                 # current_x, current_y = np.round(w[item[j]]).astype(int), np.round(h[item[j]]).astype(int)
#                 # current_x, current_y = w[item[j]], h[item[j]]
#                 # current_surface.append([current_x, current_y])
#                 # point_w_3d.append(mesh_vertices_trans[item[j]])
#
#                 # current_x, current_y = np.round(w[int(item[j])-1]).astype(int), np.round(h[int(item[j])-1]).astype(int)
#                 current_x, current_y = w[int(item[j]) - 1], h[int(item[j]) - 1]
#                 current_surface.append([current_y, 1000-current_x])
#                 point_c_3d.append(mesh_vertices_trans[int(item[j]) - 1])
#
#             if calculate_polygon_area(current_surface, point_c_3d):
#                 # print(current_surface)
#                 # cv2.fillConvexPoly(img, np.array(current_surface), color=(255, 255, 255))
#                 # cv2.fillPoly(img, [np.array(current_surface)], color=(255, 255, 255))
#                 polygon = np.asarray(current_surface)
#                 vertex_row_coords, vertex_col_coords = polygon.T
#                 fill_row_coords, fill_col_coords = draw.polygon(
#                     vertex_row_coords, vertex_col_coords, (h_size, w_size))
#                 cc = [[fill_col_coords[key_key], fill_row_coords[key_key]] for key_key in range(len(fill_col_coords))]
#
#                 # cc = scanline_polygons(polygon, (h_size, w_size))
#                 polypoints.extend(cc)
#     #
#     # for item in polypoints:
#     #     # project 3D points
#     #     try:
#     #         img[item[0], item[1], 2] = 255
#     #         img[item[0], item[1], 1] = 255
#     #         img[item[0], item[1], 0] = 255
#     #     except:
#     #         print(item)
#
#         obj_count += 1
#
#     cv2.imwrite("./p4_skiimage.png", img)



