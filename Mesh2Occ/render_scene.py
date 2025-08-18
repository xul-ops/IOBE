import sys, os, time
import numpy as np
import json
import math
import shutil
# import random
import colorsys
import bpy
import argparse
import bpy_extras
from mathutils import Matrix
from math import radians
from bpy import context, data, ops

# add current path to env
sys.path.append(os.getcwd())

from utils.blender_utils import *
import utils.configs as configs

"""
render depth, normal map and rgba

"""

# # from scipy.spatial.transform import Rotation as R
if configs.extra_rotation:
    # import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--extrot', type=str, default=str(np.eye(3).tolist()), help='extra rotation matrix for obj')
    argv = sys.argv[sys.argv.index('--') + 1:]
    args = parser.parse_args(argv)


def init_blender(key):
    scene = bpy.context.scene
    scene.render.resolution_x = configs.img_resolution_x
    scene.render.resolution_y = configs.img_resolution_y
    scene.render.resolution_percentage = 100
    scene.render.engine = 'CYCLES'  # 'BLENDER_EEVEE' not in this version
    # when cannot add background image
    scene.cycles.film_transparent = True
    # scene.render.alpha_mode = 'TRANSPARENT'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '8'
    scene.render.image_settings.file_format = 'PNG'
    scene.render.use_overwrite = True
    scene.render.use_file_extension = True

    if configs.render_depth or configs.render_normal:
        bpy.context.scene.use_nodes = True
        tree = bpy.context.scene.node_tree
        links = tree.links

        # clear default nodes
        for n in tree.nodes:
            tree.nodes.remove(n)

    # node depth
    if configs.render_depth:
        render_layer_node = tree.nodes.new('CompositorNodeRLayers')
        map_value_node = tree.nodes.new('CompositorNodeMapValue')
        file_output_node = tree.nodes.new('CompositorNodeOutputFile')
        # clip_start default 0.1 clip_end default 100
        map_value_node.offset[0] = -0.1
        # # (10 -0.1 )
        map_value_node.size[0] = 1 / (10 - 0.1)
        map_value_node.use_min = True
        map_value_node.use_max = True
        map_value_node.min[0] = 0.1  # 0001
        map_value_node.max[0] = 1.0  # 255.0

        file_output_node.format.color_mode = 'BW'
        file_output_node.format.color_depth = '8'
        file_output_node.format.file_format = 'PNG'
        file_output_node.base_path = configs.image_save_dir + str(key).zfill(5) + "/"

        links.new(render_layer_node.outputs[2], map_value_node.inputs[0])
        links.new(map_value_node.outputs[0], file_output_node.inputs[0])
        file_output_node = bpy.context.scene.node_tree.nodes[2]  # depth
        file_output_node.file_slots[0].path = 'blender-######.depth.png'  # blender placeholder #


    if configs.render_normal:
        # Add passes for additionally dumping albedo and normals.
        bpy.context.scene.render.layers["RenderLayer"].use_pass_normal = True
        # bpy.context.scene.render.layers["RenderLayer"].use_pass_color = True
        bpy.context.scene.render.layers["RenderLayer"].use_pass_environment = True

        # Create input render layer node.
        render_layers = tree.nodes.new('CompositorNodeRLayers')

        # # default depth, not so clear in the far plane
        # depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
        # depth_file_output.label = 'Depth Output'
        # # links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
        # # Remap as other types can not represent the full range of depth.
        # normalize = tree.nodes.new(type="CompositorNodeNormalize")
        # links.new(render_layers.outputs['Depth'], normalize.inputs[0])
        # links.new(normalize.outputs[0], depth_file_output.inputs[0])

        scale_normal = tree.nodes.new(type="CompositorNodeMixRGB")
        scale_normal.blend_type = 'MULTIPLY'
        # scale_normal.use_alpha = True
        # 0.5?
        scale_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
        links.new(render_layers.outputs['Normal'], scale_normal.inputs[1])

        bias_normal = tree.nodes.new(type="CompositorNodeMixRGB")
        bias_normal.blend_type = 'ADD'
        # bias_normal.use_alpha = True
        bias_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
        links.new(scale_normal.outputs[0], bias_normal.inputs[1])

        normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
        normal_file_output.label = 'Normal Output'
        links.new(bias_normal.outputs[0], normal_file_output.inputs[0])
        normal_file_output.file_slots[0].path = 'blender-######.normal.png' # normal
        normal_file_output.base_path = configs.image_save_dir + str(key).zfill(5) + "/"


    if configs.blender_use_gpu:
        # only cycles engine can use gpu
        scene.render.tile_x = 512
        scene.render.tile_y = 512
        scene.cycles.device = 'GPU'
        # bpy.data.scenes["Scene"].cycles.device = 'GPU'
        bpy.types.CyclesRenderSettings.device = 'GPU'
        # bpy.context.user_preferences.system.compute_device_type = 'CUDA'
        # bpy.context.user_preferences.system.compute_device = 'CUDA_0'
        # scene.cycles.max_bounces = 5
        # scene.cycles.caustics_reflective = False
        # scene.cycles.caustics_refractive = False

    # Delete default obj
    # bpy.data.objects.remove(bpy.data.objects['Camera'])
    # bpy.data.objects.remove(bpy.data.objects['Cube'])
    # bpy.data.objects.remove(bpy.data.objects['Light'])
    for obj in bpy.data.objects:
        obj.select = True
    bpy.ops.object.delete()


def render_function(model_pose_infos, model_dir, save_dir):
    uv_material_list = list()
    # set_png_bg(bg_dir+"bg1.png")
    # model_pose_infos = [model_pose_infos[0], model_pose_infos[1],  model_pose_infos[2]]
    for index, model_pose_info in enumerate(model_pose_infos):
        # get object params
        model_id = model_pose_info['shape_id']
        trans_vec = np.array(model_pose_info['translation'])
        rot_mat = np.array(model_pose_info['rotation'])
        # fov = math.radians(model_pose_info['fov'])
        if model_pose_info['category_id'] in configs.category_must_skip:
            continue

        # load object
        # obj_file = os.path.join(model_dir, model_id + '/normalized_model.obj')
        obj_file = os.path.join(model_dir, model_id + '/raw_model.obj')
        if not os.path.exists(obj_file):
            raise FileNotFoundError(model_id + " obj file not exists!")
        bpy.ops.import_scene.obj(filepath=obj_file)

        # add uv texture
        # in linux, blender will add uvtexture directly/automatically...
        mat_name = "UVTexture_" + model_id
        if mat_name in uv_material_list:
            pass
        else:
            uv_material_list.append(mat_name)
            texture_file = os.path.join(model_dir, model_id + '/texture.png')
            cTex = bpy.data.materials.new(mat_name)
            # cTex = CreateMaterialFromImage(mat_name, texture_file)
            bpy.data.materials[mat_name].use_nodes = True
            texture_tree = bpy.data.materials[mat_name].node_tree
            texture_links = texture_tree.links
            texture_node = texture_tree.nodes.new("ShaderNodeTexImage")
            texture_node.image = bpy.data.images.load(texture_file)
            # texture_links.new(texture_node.outputs[0], texture_tree.nodes['Diffuse BSDF'].inputs[0]) Base Color
            texture_links.new(texture_node.outputs["Color"], texture_tree.nodes['Diffuse BSDF'].inputs[0])
            # bpy.data.scenes['Scene'].render.layers['RenderLayer'].material_override = bpy.data.materials[mat_name]

        # transform object
        # if we use extra rotation, will replace the original rotation
        if configs.extra_rotation:
            rot_mat = np.array(eval(args.extrot))

        transform_object(trans_vec, rot_mat, mat_name, name=model_id + '_' + str(index))


    # set camera 0.734843811495514
    camera = add_camera((0, 0, 0), configs.camera_fov, 'camera')

    # world lighting
    world = bpy.data.worlds['World']
    world.light_settings.use_ambient_occlusion = True
    world.light_settings.ao_factor = 1.

    # camera extrinsic, False
    if configs.camera_extrinsic:
        # fov * (pi / 180.0)
        # camera.data.angle = 0.6 # 1 # fov
        camera.location = (configs.camera_tx, configs.camera_ty, configs.camera_tz)
        camera.rotation_mode = 'XYZ'
        camera.rotation_euler[0] = configs.camera_rx * (np.pi / 180.0)
        camera.rotation_euler[1] = configs.camera_ry * (np.pi / 180.0)
        camera.rotation_euler[2] = configs.camera_rz * (np.pi / 180.0)


    # # get intrinsic matrix
    K_blender = get_calibration_matrix_K_from_blender(camera.data)
    # np.save('1080_intrinsic.npy', np.array(K_blender))
    # np.save('./data/scene_data/1080_intrinsic.npy', np.array(K_blender))
    
    print(K_blender)
    camera_matrix = get_blender_camera_matrix(camera)
    print(camera_matrix)

    # np.save('./data/scene_data/ca_pm_720.npy', np.array(camera_matrix).astype(np.float64))


    # render scene image
    save_img_name = "brgba"
    if configs.extra_rotation:
        save_img_name = configs.extra_rot_name + "_" + save_img_name
    save_file = os.path.join(save_dir, save_img_name)
    render_image_depth(save_file)

    # clear sys/scene
    clear_scene()


if __name__ == '__main__':

    # train_db = False
    model_dir = configs.blender_model_dir
    # if train_db:
    #     model_pose_info_file = configs.blender_model_pose_info_file + "train/"
    # else:
    #     model_pose_info_file = configs.blender_model_pose_info_file + "test/"
    
    # model_pose_info_file = configs.pose_info_dir +  "/pose_info.npy"
    error_list = list()
    
    # img_list_path = "../occdata/synocc_split/synocc_fval.txt"

    # with open(img_list_path, 'r') as f:
    #     names = f.readlines()
    # run_list = [x.replace('\n', '') for x in names]

    for key in range(52, 53):

        model_pose_name = str(key).zfill(5) + "/"
        # c_model_pose_info_file = model_pose_info_file + model_pose_name + "pose_info.npy"
        c_model_pose_info_file = configs.pose_info_dir +  "/pose_info.npy"

        # if train_db:
        #     current_img_name = c_model_pose_info_file.split("/")[-1].replace(".npy", "")
        #     folder = "scene/" + current_img_name + "/"
        # else:
        #     current_img_name = c_model_pose_info_file.split("/")[-1].replace(".npy", "")
        #     current_img_name = str(int(current_img_name) + 14761 + 1).zfill(5)
        #     folder = "scene/" + current_img_name + "/"

        if not os.path.exists(configs.image_save_dir + model_pose_name):
            os.makedirs(configs.image_save_dir + model_pose_name)

        folder = model_pose_name
        try:
            model_pose_infos = np.load(c_model_pose_info_file, allow_pickle=True)
            # np.save(configs.image_save_dir + folder + "pose_info.npy", model_pose_infos)
        except FileNotFoundError:
            continue

        init_blender(key)
        try:
            render_function(model_pose_infos, model_dir, configs.image_save_dir + folder)
        except RuntimeError:
            # import obj zero division error
            error_list.append(key)
            print(key)
            continue

        # shutil
        depth_name = "bdepth.png" # "blender_depth.png"
        if configs.extra_rotation:
            depth_name = configs.extra_rot_name + "_" + depth_name
        if configs.render_depth:
            os.rename(configs.image_save_dir + folder + "blender-000001.depth.png",
                      configs.image_save_dir  + depth_name)

        normal_name = "bnormal.png" # "blender_normal.png"
        if configs.extra_rotation:
            normal_name = configs.extra_rot_name + "_" + normal_name
        if configs.render_normal:
            os.rename(configs.image_save_dir + folder + "blender-000001.normal.png",
                      configs.image_save_dir + normal_name)

    # print(error_list)
