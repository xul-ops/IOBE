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



# # from scipy.spatial.transform import Rotation as R
if configs.extra_rotation:
    # import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--extrot', type=str, default=str(np.eye(3).tolist()), help='extra rotation matrix for obj')
    argv = sys.argv[sys.argv.index('--') + 1:]
    args = parser.parse_args(argv)


def init_blender():
    scene = bpy.context.scene
    scene.render.resolution_x = 1200
    scene.render.resolution_y = 1200
    scene.render.resolution_percentage = 100

    # scene.render.image_settings.color_mode = 'RGBA'
    scene.render.image_settings.color_depth = '8'
    scene.render.image_settings.file_format = 'PNG'

    # Delete default obj
    # bpy.data.objects.remove(bpy.data.objects['Camera'])
    # bpy.data.objects.remove(bpy.data.objects['Cube'])
    # bpy.data.objects.remove(bpy.data.objects['Light'])
    for obj in bpy.data.objects:
        obj.select = True
    bpy.ops.object.delete()


def render_image_depth(save_file):
    for o in bpy.data.objects:
        if o.type == 'CAMERA':
            bpy.context.scene.camera = o
            break

    scene = bpy.context.scene
    scene.render.filepath = save_file  # rgba
    bpy.ops.render.render(write_still=True)


def transform_object_depth(trans_vec, rot_mat, name='obj'):
    for i, obj in enumerate(bpy.context.selected_objects):
        if name is not None:
            if len(bpy.context.selected_objects) == 1:
                obj.name = name
            else:
                obj.name = name + '_' + str(i)

        for mtl in range(len(bpy.data.materials)):
            # Include this material and geometry that uses it in raytracing calculations
            bpy.data.materials[mtl].use_raytrace = True
            # Make this material insensitive to light or shadow
            bpy.data.materials[mtl].use_shadeless = False
            # bpy.data.materials[mtl].ambient = 0.5
            # bpy.data.materials[mtl].use_transparency = False

        # transformation
        trans_4x4 = Matrix.Translation(trans_vec)
        rot_4x4 = Matrix(rot_mat).to_4x4()
        scale_4x4 = Matrix(np.eye(4))
        obj.matrix_world = trans_4x4 * rot_4x4 * scale_4x4


def render_function(model_pose_infos, model_dir, save_dir):

    # set_png_bg(bg_dir+"bg1.png")
    # model_pose_infos = [model_pose_infos[0], model_pose_infos[1],  model_pose_infos[2]]
    colors = ncolors(len(model_pose_infos))
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

        # transform object
        # if we use extra rotation, will replace the original rotation
        if configs.extra_rotation:
            rot_mat = np.array(eval(args.extrot))

        # transform_object(trans_vec, rot_mat, mat_name, name=model_id + '_' + str(index))
        t = transform_object_idmap(trans_vec, rot_mat, index, colors, name='obj')

    # set camera
    camera = add_camera((0, 0, 0), 0.734843811495514, 'camera')

    # world lighting
    world = bpy.data.worlds['World']
    world.light_settings.use_ambient_occlusion = True
    world.light_settings.ao_factor = 1.
    world.use_nodes = True
    # not work...
    world.node_tree.nodes['Background'].inputs[0].default_value[0:3] = (0, 0, 0)

    # camera extrinsic, False
    if configs.camera_extrinsic:
        # fov * (pi / 180.0)
        # camera.data.angle = 0.6 # 1 # fov
        camera.location = (configs.camera_tx, configs.camera_ty, configs.camera_tz)
        camera.rotation_mode = 'XYZ'
        camera.rotation_euler[0] = configs.camera_rx * (np.pi / 180.0)
        camera.rotation_euler[1] = configs.camera_ry * (np.pi / 180.0)
        camera.rotation_euler[2] = configs.camera_rz * (np.pi / 180.0)

    # render scene image
    save_img_name = "blender_idmap"
    if configs.extra_rotation:
        save_img_name = configs.extra_rot_name + "_" + save_img_name
    save_file = os.path.join(save_dir, save_img_name)
    render_image_depth(save_file)

    # clear sys/scene
    clear_scene()


if __name__ == '__main__':

    train_db = False
    model_dir = configs.blender_model_dir
    # if train_db:
    #     model_pose_info_file = configs.blender_model_pose_info_file + "train/"
    # else:
    #     model_pose_info_file = configs.blender_model_pose_info_file + "test/"
    model_pose_info_file = "../occ/"

    for key in range(1, 2):
        # model_pose_name = str(key).zfill(7) + ".npy"
        model_pose_name = str(key).zfill(5) + "/"
        c_model_pose_info_file = model_pose_info_file + model_pose_name + "pose_info.npy"

        # if train_db:
        #     current_img_name = c_model_pose_info_file.split("/")[-1].replace(".npy", "")
        #     folder = "scene/" + current_img_name + "/"
        # else:
        #     current_img_name = c_model_pose_info_file.split("/")[-1].replace(".npy", "")
        #     current_img_name = str(int(current_img_name) + 14761 + 1).zfill(7)
        #     folder = "scene/" + current_img_name + "/"

        # if not os.path.exists(configs.image_save_dir + folder):
        #     os.makedirs(configs.image_save_dir + folder)
        #     os.makedirs(configs.image_save_dir + folder + "info/")

        folder = model_pose_name
        try:
            model_pose_infos = np.load(c_model_pose_info_file, allow_pickle=True)
        except FileNotFoundError:
            continue
        # print(model_pose_infos)

        init_blender()
        render_function(model_pose_infos, model_dir, configs.image_save_dir + folder)


