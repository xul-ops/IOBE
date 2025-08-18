
# use_neighbor_4 = True

# depth_type = "z_buffer" # "z_buffer" # "distance"

# intrinsic_file = 'data/scene_data/intrinsic.npy'
camera_matrix_file = 'data/scene_data/ca_pm_1080.npy'
# camera_matrix_file = 'data/scene_data/ca_pm_960.npy'

obj_model_dir = 'data/scene_00052_objs/'

image_save_dir = './buffer_test/'
# image_save_dir = "../occdata/soccdsf/"
# pose_info_dir = "../occdata/soccdsf/"
pose_info_dir = "data/scene_00052_files/"

# simple or sub mesh
# use_simplemesh = True
# voxel_size_changes = 200
# contraction_method = "quadric" # "average" # "quadric"
# use_simplify_quadric_decimation = False
# target_mesh_num = None
# submesh_method = "midpoint" # "loop" # "midpoint"
# number_of_iterations = 2

# count_backface = False

# boundary_thick = True

# check_self_obp = False

# check_z_fighting = False

use_threshold_shortest = True 
threshold_shortest_len = 10

add_graph_edge_weight = True
count_small_faces = True
# deafault should be false
stop_at_easy = False

# filter_ob_image = False
# use_covered_points_list = False

is_decimal = True # False  # False -> np.float64
decimal_prec = 35  # default 28 prec

random_seed = 42

# generate rgb img for training
multi_rgb = False
rgb_num = 1

# lamp  [33, 34]
category_must_skip = []

# use downsampling for label_gt
use_down_sampling =  False
# default blender cycle aa samples is 4, but default blender engine is 5, we didn't try 5
down_sampling_rate = 4

if use_down_sampling:
    background_img_file = 'data/scene_data/black2.png'
else:
    background_img_file = 'data/scene_data/black1.png'


# blender setting
img_resolution_x = 1080
img_resolution_y = 1080
# default 0.734843811495514
camera_fov = 0.85 # 0.6 - 1.0

# multi_frame = False
camera_extrinsic = False
# euler
camera_rx = 10
camera_ry = 10
camera_rz = 10
camera_tx = 0
camera_ty = 0
camera_tz = 0

extra_rotation = False
extra_rot_name = "rot_z45"
extra_rotation_mode = 'xyz'
extra_rotation_angle = [0, 0, 45]

# is_twolinear_interplation = True
# z_value_linear = True
# use_raw_obj = True

# blender configs
blender_use_gpu = True # False # does this really use gpu ? 
# render_idmap = False # not work in Linux, should delete textures files
render_depth =  True
render_normal = True # False


blender_path = "../GeOB/geocc/blender279b/blender"

blender_model_pose_info_file = 'data/'
blender_model_dir = obj_model_dir


