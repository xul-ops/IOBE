import sys, os, time
import numpy as np
import json
import math
import random
import colorsys
import bpy
import bpy_extras
from mathutils import Matrix
from math import radians
from bpy import context, data, ops


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


def CreateMaterialFromImage(matName, imgPath):
    """
    Creates and returns a material from an Image
    https://blender.stackexchange.com/questions/157531/blender-2-8-python-add-texture-image
    https://blender.stackexchange.com/questions/157531/blender-2-8-python-add-texture-image
    https://blender.stackexchange.com/questions/118646/add-a-texture-to-an-object-using-python-and-blender-2-8
    :param matName:
    :param imgPath:
    :return:
    """
    # realpath = os.path.expanduser(imgPath)
    try:
        img = bpy.data.images.load(imgPath)
    except:
        raise NameError("Cannot load image %s" % imgPath)

    # Create image texture from image
    cTex = bpy.data.textures.new(matName, type='IMAGE')
    cTex.image = img

    # Create material
    mat = bpy.data.materials.new(matName)

    # Add texture slot for color texture
    mtex = mat.texture_slots.add()
    mtex.texture = cTex
    mtex.texture_coords = 'UV'
    mtex.use_map_color_diffuse = True
    mtex.use_map_color_emission = True
    mtex.emission_color_factor = 0.5
    mtex.use_map_density = True
    mtex.mapping = 'FLAT'
    return mat


def get_blender_camera_matrix(camera):
    projection_matrix = camera.calc_matrix_camera(
        bpy.context.scene.render.resolution_x,
        bpy.context.scene.render.resolution_y,
        bpy.context.scene.render.pixel_aspect_x,
        bpy.context.scene.render.pixel_aspect_y,
    )
    # print(projection_matrix)
    return projection_matrix


def get_calibration_matrix_K_from_blender(camd):
    '''
    get camera intrinsic matrix
    '''
    f_in_mm = camd.lens
    scene = bpy.context.scene
    resolution_x_in_px = scene.render.resolution_x
    resolution_y_in_px = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100
    sensor_width_in_mm = camd.sensor_width
    sensor_height_in_mm = camd.sensor_height
    pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    if (camd.sensor_fit == 'VERTICAL'):
        s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
        s_v = resolution_y_in_px * scale / sensor_height_in_mm
    else: # 'HORIZONTAL' and 'AUTO'
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
        s_u = resolution_x_in_px * scale / sensor_width_in_mm
        s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_width_in_mm
        #s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((alpha_u, skew,    u_0),
        (    0  , alpha_v, v_0),
        (    0  , 0,        1 )))
        
    # K = np.array([
    #     [alpha_u, skew, u_0],
    #     [0, alpha_v, v_0],
    #     [0, 0, 1]
    # ], dtype=np.float64)
    return K
   
    
def get_calibration_matrix_K_from_blender_v2(camdata, mode='simple'):
    scene = bpy.context.scene

    scale = scene.render.resolution_percentage / 100
    width = scene.render.resolution_x * scale  # px
    height = scene.render.resolution_y * scale  # px

    if mode == 'simple':
        aspect_ratio = width / height
        K = np.zeros((3, 3), dtype=np.float32)
        K[0][0] = width / 2 / np.tan(camdata.angle / 2)
        K[1][1] = height / 2. / np.tan(camdata.angle / 2) * aspect_ratio
        K[0][2] = width / 2.
        K[1][2] = height / 2.
        K[2][2] = 1.
        K.transpose()

    if mode == 'complete':

        focal = camdata.lens  # mm
        sensor_width = camdata.sensor_width  # mm
        sensor_height = camdata.sensor_height  # mm
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

        if (camdata.sensor_fit == 'VERTICAL'):
            # the sensor height is fixed (sensor fit is horizontal),
            # the sensor width is effectively changed with the pixel aspect ratio
            s_u = width / sensor_width / pixel_aspect_ratio
            s_v = height / sensor_height
        else:  # 'HORIZONTAL' and 'AUTO'
            # the sensor width is fixed (sensor fit is horizontal),
            # the sensor height is effectively changed with the pixel aspect ratio
            pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
            s_u = width / sensor_width
            s_v = height * pixel_aspect_ratio / sensor_height

        # parameters of intrinsic calibration matrix K
        alpha_u = focal * s_u
        alpha_v = focal * s_v
        u_0 = width / 2
        v_0 = height / 2
        skew = 0  # only use rectangular pixels

        K = np.array([
            [alpha_u, skew, u_0],
            [0, alpha_v, v_0],
            [0, 0, 1]
        ], dtype=np.float64)

    return K


def try2_get_camera_matrix(cam):
    # get the relevant data
    scene = bpy.context.scene

    f_in_mm = cam.lens
    sensor_width_in_mm = cam.sensor_width

    w = scene.render.resolution_x
    h = scene.render.resolution_y

    pixel_aspect = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x

    f_x = np.float64(f_in_mm) / sensor_width_in_mm * w
    f_y = f_x * pixel_aspect

    # yes, shift_x is inverted. WTF blender?
    c_x = w * (0.5 - cam.shift_x)
    # and shift_y is still a percentage of width..
    c_y = h * 0.5 + w * cam.shift_y
    # print(f_in_mm, sensor_width_in_mm, w)
    # print(pixel_aspect, h)
    # print(cam.shift_x, cam.shift_y)
    K = np.array([
        [f_x, 0, c_x],
        [0, f_y, c_y],
        [0, 0, 1]
    ], dtype=np.float64)

    # K = Matrix(((f_x, 0, c_x)),
    #      [0, f_y, c_y],
    #      [0, 0, 1]])
    return K


# Clear all nodes in a mat
def clear_material(material):
    if material.node_tree:
        material.node_tree.links.clear()
        material.node_tree.nodes.clear()


def clear_scene():
    '''
    clear blender system for scene and object with pose, refer to 3D-R2N2
    '''
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_pattern(pattern="RotCenter")
    bpy.ops.object.select_pattern(pattern="Lamp*")
    bpy.ops.object.select_pattern(pattern="Camera")
    bpy.ops.object.select_all(action='INVERT')
    bpy.ops.object.delete()

    # The meshes still present after delete
    for item in bpy.data.meshes:
        bpy.data.meshes.remove(item)
    for item in bpy.data.materials:
        bpy.data.materials.remove(item)


def add_camera(xyz=(0, 0, 0), fov=1, name=None, proj_model='PERSP', sensor_fit='HORIZONTAL'):
    # https://mcarletti.github.io/articles/blenderintrinsicparams/
    # https://gist.github.com/autosquid/8e1cddbc0336a49c6f84591d35371c4d
    bpy.ops.object.camera_add()
    cam = bpy.context.active_object

    # cam.rotation_mode = 'XYZ'
    cam.rotation_euler[0] = radians(0)
    cam.rotation_euler[1] = radians(0)
    cam.rotation_euler[2] = radians(0)

    if name is not None:
        cam.name = name

    cam.location = xyz
    cam.data.type = proj_model
    cam.data.angle = fov
    cam.data.sensor_fit = sensor_fit
    # near default 0.1
    # cam.data.clip_start = 0.1
    # far default 100
    # cam.data.clip_end = 255
    return cam


def transform_object_idmap(trans_vec, rot_mat, index, colors, name='obj'):
    rgb = colors[index]
    blender_rgb = np.array(rgb) / 255

    for i, obj in enumerate(bpy.context.selected_objects):
        if name is not None:
            if len(bpy.context.selected_objects) == 1:
                obj.name = name
            else:
                obj.name = name + '_' + str(i)

        # material color
        if len(obj.data.materials) == 0:
            mat = bpy.data.materials.new(obj.name)
            mat.diffuse_color = (blender_rgb[0], blender_rgb[1], blender_rgb[2])
            mat.diffuse_shader = 'FRESNEL'
            mat.diffuse_intensity = 1.0
            mat = bpy.data.materials[obj.name]
            obj.data.materials.append(mat)
        else:
            obj.active_material.diffuse_color = (blender_rgb[0], blender_rgb[1], blender_rgb[2])

        for mtl in range(len(bpy.data.materials)):
            bpy.data.materials[mtl].use_shadeless = True

        color_inferior = obj.active_material.diffuse_color

        real_rgb = float(255.999 * pow(color_inferior.r, 1 / 2.2)), float(
            255.999 * pow(color_inferior.g, 1 / 2.2)), float(255.999 * pow(color_inferior.b, 1 / 2.2))

        # transformation
        trans_4x4 = Matrix.Translation(trans_vec)
        rot_4x4 = Matrix(rot_mat).to_4x4()
        scale_4x4 = Matrix(np.eye(4))
        obj.matrix_world = trans_4x4 * rot_4x4 * scale_4x4

    return real_rgb


def set_depth_path(new_path):
    """
    set depth output path to new_path
    Args: new rendered depth output path
    """
    file_output_node = bpy.context.scene.node_tree.nodes[2]
    file_output_node.base_path = new_path


def transform_object(trans_vec, rot_mat, mat_name, name='obj'):

    for i, obj in enumerate(bpy.context.selected_objects):
        obj.material_slots[0].material = bpy.data.materials[mat_name]
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
        # print(scale_4x4)
        obj.matrix_world = trans_4x4 * rot_4x4 * scale_4x4
        # print(obj.matrix_world)


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


def render_image(save_file):
    for o in bpy.data.objects:
        if o.type == 'CAMERA':
            bpy.context.scene.camera = o
            break

    scene = bpy.context.scene
    scene.render.filepath = save_file
    bpy.ops.render.render(write_still=True)  # render still


def render_image_depth(save_file):
    for o in bpy.data.objects:
        if o.type == 'CAMERA':
            bpy.context.scene.camera = o
            break

    scene = bpy.context.scene
    scene.render.filepath = save_file  # rgba
    # file_output_node = bpy.context.scene.node_tree.nodes[2]  # depth
    # file_output_node.file_slots[0].path = 'blender-######.depth.png'  # blender placeholder #

    bpy.ops.render.render(write_still=True)


def node_setting_init():
    """node settings for render rgb images
    mainly for compositing the background images
    """

    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    for node in tree.nodes:
        tree.nodes.remove(node)

    image_node = tree.nodes.new('CompositorNodeImage')
    scale_node = tree.nodes.new('CompositorNodeScale')
    alpha_over_node = tree.nodes.new('CompositorNodeAlphaOver')
    render_layer_node = tree.nodes.new('CompositorNodeRLayers')
    # file_output_node = tree.nodes.new('CompositorNodeOutputFile')

    scale_node.space = 'RENDER_SIZE'
    # file_output_node.base_path = "syn_rgb"

    links.new(image_node.outputs[0], scale_node.inputs[0])
    links.new(scale_node.outputs[0], alpha_over_node.inputs[1])
    links.new(render_layer_node.outputs[0], alpha_over_node.inputs[2])
    # links.new(alpha_over_node.outputs[0], file_output_node.inputs[0])


def scene_setting_init(use_gpu=False):
    """initialize blender setting configurations
    """
    sce = bpy.context.scene.name
    bpy.data.scenes[sce].render.engine = "CYCLES"
    bpy.data.scenes[sce].cycles.film_transparent = True
    #output
    bpy.data.scenes[sce].render.image_settings.color_mode = "RGB"
    bpy.data.scenes[sce].render.image_settings.color_depth = "16"
    bpy.data.scenes[sce].render.image_settings.file_format = 'PNG'

    #dimensions
    bpy.data.scenes[sce].render.resolution_x = 1200
    bpy.data.scenes[sce].render.resolution_y = 1200
    bpy.data.scenes[sce].render.resolution_percentage = 100

    if use_gpu:
        # only cycles engine can use gpu
        # bpy.data.scenes[sce].render.engine = 'CYCLES'
        bpy.data.scenes[sce].render.tile_x = 512
        bpy.data.scenes[sce].render.tile_y = 512
        bpy.context.scene.cycles.device = 'GPU'
        bpy.types.CyclesRenderSettings.device = 'GPU'
        bpy.data.scenes[sce].cycles.device = 'GPU'
        # bpy.context.user_preferences.system.compute_device_type = 'CUDA'
        # bpy.context.user_preferences.system.compute_device = 'CUDA_0'
        # bpy.data.scenes['Scene'].cycles.max_bounces = 5
        # bpy.data.scenes['Scene'].cycles.caustics_reflective = False
        # bpy.data.scenes['Scene'].cycles.caustics_refractive = False


def clear_mesh():
    """ clear all meshes in the secene
    """
    bpy.ops.object.select_all(action='DESELECT')
    # for obj in bpy.data.objects:
    #     if obj.type == 'MESH':
    #         obj.select = True
    # bpy.ops.object.delete()
    # Delete default obj
    # bpy.data.objects.remove(bpy.data.objects['Camera'])
    # bpy.data.objects.remove(bpy.data.objects['Cube'])
    # bpy.data.objects.remove(bpy.data.objects['Light'])
    for obj in bpy.data.objects:
        obj.select = True
    bpy.ops.object.delete()


def set_png_bg(img_dir):
    scene_setting_init()
    node_setting_init()
    image_node = bpy.context.scene.node_tree.nodes[0]
    image_node.image = bpy.data.images.load(img_dir)


def build_rgb_background():
    # # bpy.types.World
    # world.use_nodes = True
    # node_tree = world.node_tree
    #
    # rgb_node = node_tree.nodes.new(type="ShaderNodeRGB")
    # rgb_node.outputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    # node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
    # node_tree.links.new(rgb_node.outputs["Color"], node_tree.nodes["Background"].inputs["Color"])

    bpy.data.worlds['World'].use_nodes = True
    bpy.data.worlds['World'].node_tree.nodes['Background'].inputs[0].default_value[0:3] = (0, 0, 0)


def set_env_hdr(hdr_dir):
    """
    hdr file work
    img-png can't
    """
    world = bpy.data.worlds['World']
    world.use_nodes = True
    node_tree = world.node_tree
    environment_texture_node = node_tree.nodes.new(type="ShaderNodeTexEnvironment")
    environment_texture_node.image = bpy.data.images.load(hdr_dir+"bg1.hdr")
    # environment_texture_node.image = bpy.data.images.load(bg_dir)
    mapping_node = node_tree.nodes.new(type="ShaderNodeMapping")
    mapping_node.rotation[2] = 0.0
    tex_coord_node = node_tree.nodes.new(type="ShaderNodeTexCoord")
    node_tree.links.new(tex_coord_node.outputs["Generated"], mapping_node.inputs["Vector"])
    node_tree.links.new(mapping_node.outputs["Vector"], environment_texture_node.inputs["Vector"])
    node_tree.links.new(environment_texture_node.outputs["Color"], node_tree.nodes["Background"].inputs["Color"])

    # cTex = bpy.data.textures.new(matName, type='IMAGE')
    # cTex.image = bpy.data.images.load(img_dir+"bg1.png")
    # print(cTex.image)
    # # world.active_texture(cTex)
    # slot = world.texture_slots.add()
    # slot.texture = cTex
    # slot.texture_coords = 'OBJECT'
    # slot.use_map_horizon = True


#######
#light
#######

# bpy.ops.object.lamp_add(type='SPOT')
# # bpy.ops.object.lamp_add(type='SUN')
# # bpy.ops.object.lamp_add(type='POINT')
# bpy.data.lamps['Spot'].energy = 500
#
# # bpy.data.objects['Lamp'].data.energy = 50
# # bpy.ops.object.lamp_add(type='SUN')
# # bpy.ops.object.lamp_add(type='POINT')
# # self.translate('Point', [0, -10, 3])
# # bpy.data.lamps['Point'].use_specular = True
# # bpy.data.worlds['World'].horizon_color = (0, 0, 0)
# # # light_data = bpy.data.lamps.new('light', type='POINT')
# # # light = bpy.data.objects.new('light', light_data)
# # # scene.objects.link(light)
# # # light.location = mathutils.Vector((0, 0, 8))
def remove_all_lights():
    for l in filter(lambda o: o.type == 'LIGHT', bpy.data.objects):
        bpy.data.objects.remove(l)


def set_background_light(color, strength):
    world_node_tree = bpy.data.worlds['World'].node_tree
    for link in world_node_tree.nodes['Background'].inputs['Color'].links:
        world_node_tree.links.remove(link)
    world_node_tree.nodes['Background'].inputs['Color'].default_value = color
    world_node_tree.nodes['Background'].inputs['Strength'].default_value = strength


def add_light(name, light_type, location, energy):
    light_data = bpy.data.lights.new(name=name + '_data', type=light_type)
    light_data.energy = energy
    light_object = bpy.data.objects.new(name=name, object_data=light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = location
    return light_object


def studio_light(color, strength):
    remove_all_lights()
    add_light('PointLight1', 'POINT', (-1., -0.5, 2.), 70.0 * strength)
    add_light('PointLight2', 'POINT', (1., -0.5, 2.), 70.0 * strength)
    set_background_light(color, strength)


def studio_light_zfront(color, strength):
    remove_all_lights()
    add_light('PointLight1', 'POINT', (-1., -1, 0.), 70.0 * strength)
    add_light('PointLight2', 'POINT', (1., -1, 0.), 70.0 * strength)
    set_background_light(color, strength)


def hdr_light(path, strength, rotation=100):
    remove_all_lights()
    world_node_tree = bpy.data.worlds['World'].node_tree
    if 'Environment Texture' not in world_node_tree.nodes:
        world_node_tree.nodes.new('ShaderNodeTexEnvironment').name = 'Environment Texture'
    if 'Mapping' not in world_node_tree.nodes:
        world_node_tree.nodes.new('ShaderNodeMapping').name = 'Mapping'
    if 'Texture Coordinate' not in world_node_tree.nodes:
        world_node_tree.nodes.new('ShaderNodeTexCoord').name = 'Texture Coordinate'
    world_node_tree.nodes['Environment Texture'].image = bpy.data.images.load(path)
    world_node_tree.links.new(world_node_tree.nodes['Environment Texture'].outputs['Color'],
                              world_node_tree.nodes['Background'].inputs['Color'])
    world_node_tree.links.new(world_node_tree.nodes['Mapping'].outputs['Vector'],
                              world_node_tree.nodes['Environment Texture'].inputs['Vector'])
    world_node_tree.links.new(world_node_tree.nodes['Texture Coordinate'].outputs['Generated'],
                              world_node_tree.nodes['Mapping'].inputs['Vector'])
    world_node_tree.nodes['Mapping'].inputs['Rotation'].default_value[2] = math.radians(rotation)
    world_node_tree.nodes['Background'].inputs['Strength'].default_value = strength

