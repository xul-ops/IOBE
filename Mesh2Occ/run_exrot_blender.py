import os
import sys
import subprocess
# add current path to env
sys.path.append(os.getcwd())
import utils.configs as configs
from scipy.spatial.transform import Rotation as R

r = R.from_euler(configs.extra_rotation_mode, configs.extra_rotation_angle, degrees=True)
r = r.as_matrix()
# print(type(r))
# print(r.tolist())

blender_path = configs.blender_path

# # rgba
# py_file = "render_rgba.py"

# # depth and rgba
# py_file = "render_all.py"

# depth, normal, rgba
py_file = "render_scene.py"  # "r960.py"

# # idmap, should delete textures
# py_file = "render_idmap.py"

# example render single obj, mode: tract to
# py_file = "render_obj_withall.py"

if configs.extra_rotation:
    # extra rot for obj, not correct render_rgba.py
    # command = [blender_path, '--background', '--python', py_file, "--"]
    command = [blender_path, '--background', '--python', py_file, "--", "--extrot", str(r.tolist())]

else:
    # cmd = '{} -b -P {} '.format(blender_path, py_file)
    command = [blender_path, '--background', '--python', py_file]

    # render_obj_withall.py
    # command = [blender_path, '--background', '--python', py_file, "--", "--output_folder", "./data/mv_data/"]

subprocess.run(command)


