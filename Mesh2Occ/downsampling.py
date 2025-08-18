import os
import cv2
import numpy as np
from PIL import Image
from scipy.signal import fftconvolve
from scipy.signal.windows import blackmanharris
from collections import Counter
from utils.tools import get_point_neighbors, ncolors

from tqdm import tqdm
from tqdm.std import trange

"""
Average filter; Blender default box_filter; Blender cycle blackmanharris.
Those filters won't change the pixel coordinates, only the rgb color.
We can filter the single point after down sampling;
We can match the down sampling image with the directly OB3
"""



def nearest_distance(pixel_point):
    # work better in average filter
    colors_id = ncolors(3)
    # print(pixel_point, colors_id)
    # colors_distance = [np.linalg.norm(ccolor, pixel_point) for ccolor in colors_id]
    colors_distance = [np.sqrt(sum(np.power((ccolor - pixel_point), 2))) for ccolor in colors_id]
    
    return np.array(colors_id[colors_distance.index(np.min(colors_distance))])


def mean_ignore_num(arr, num, blender_superss=4):
    # Get count of invalid ones
    invc = np.count_nonzero(arr==num)
    if invc == blender_superss* blender_superss:
        return [112, 112, 112]

    # Get the average value for all numbers and remove contribution from num
    return (arr.sum() - invc*num)/float(arr.size-invc)    


def average_filter_downsample(img, blender_superss, img_save_name, keep_ob3color=True, simple_filtered=False, is_oo_img=False):
    supersample_size = img.shape

    # Compute the output image size
    output_size = (supersample_size[0] // blender_superss, supersample_size[1] // blender_superss)

    # Cont5rwqvert the image to a numpy array
    # img_arr = np.array(img)

    # Create an empty output array
    output_arr = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)

    # Loop over the output image pixels
    for y in range(output_size[1]):
        for x in range(output_size[0]):
            # Compute the range of pixels to average
            
            xmin = x * blender_superss
            xmax = xmin + blender_superss
            ymin = y * blender_superss
            ymax = ymin + blender_superss

            # Average the pixels in the range
            if is_oo_img:
                # new_color = np.nanmean(img[ymin:ymax, xmin:xmax], axis=(0, 1))
                # if new_color.tolist() == [np.nan, np.nan, np.nan]:
                #     new_color = np.array([112, 112, 112])
                # else:
                #     new_color = new_color * 255 / 360
                new_color = np.mean(img[ymin:ymax, xmin:xmax], axis=(0, 1)) * 255 / 360
            else:
                new_color = np.mean(img[ymin:ymax, xmin:xmax], axis=(0, 1))
            if keep_ob3color and list(new_color) != [0,0,0]:
                new_color = nearest_distance(new_color)
            #if is_oo_img:
            #    if new_color.tolist() == [0,0,0]:
            #        new_color = np.array([112, 112, 112])
            #    else:
            #        new_color = np.max(img[ymin:ymax, xmin:xmax], axis=(0, 1))  # * 255 / 360
           
            output_arr[y, x] = new_color

    # Convert the output array to an image and save it
    if simple_filtered:
        output_arr = downsampling_filter(output_arr)
    # imageio.imwrite(img_save_name, output_arr)
    cv2.imwrite(img_save_name, output_arr)
    # output_img.save(f'{blender_superss}_AvDownsampled_OB3_filtered.png')


def box_filter_downsample(img, blender_superss):
    supersample_size = img.shape
    # Compute the output image size
    output_size = (supersample_size[0] // blender_superss, supersample_size[1] // blender_superss)

    # Create an empty output array
    output_arr = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)

    # Define the box filter
    box_filter = np.ones((blender_superss, blender_superss)) / blender_superss*blender_superss
    box_filter = box_filter.reshape(blender_superss, blender_superss, 1)
    # Loop over the output image pixels
    for y in range(output_size[1]):
        for x in range(output_size[0]):
            # Compute the range of pixels to average
            xmin = x * blender_superss
            xmax = xmin + blender_superss
            ymin = y * blender_superss
            ymax = ymin + blender_superss

            # Average the pixels in the range using the box filter
            # print(img_arr[ymin:ymax, xmin:xmax].shape, box_filter.shape)
            output_arr[y, x] = np.sum(img[ymin:ymax, xmin:xmax] * box_filter, axis=(0, 1))

    # Convert the output array to an image and save it
    cv2.imwrite(img_save_name, output_arr)
    # output_img = Image.fromarray(output_arr)
    # output_img.save(f'{blender_superss}_BoxDownsampled_OB3.png')


def bmh_filter_downsample(img, blender_superss):
    # cycle blender engine default
    supersample_size = img.size
    # Compute the output image size
    output_size = (supersample_size[0] // blender_superss, supersample_size[1] // blender_superss)


    # Define the antialiasing filter
    bmh_weight = blackmanharris(blender_superss)
    bmh_filter = np.outer(bmh_weight, bmh_weight).reshape(blender_superss,blender_superss,-1)

    # Apply the antialiasing filter to the supersampled image
    # filtered_arr = fftconvolve(img_arr, antialias_filter[:, :, None], mode='same')
    # filtered_img = Image.fromarray(filtered_arr.astype(np.uint8))

    output_arr = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)

    for y in range(output_size[1]):
        for x in range(output_size[0]):
            # Compute the range of pixels to average
            xmin = x * blender_superss
            xmax = xmin + blender_superss
            ymin = y * blender_superss
            ymax = ymin + blender_superss

            # Average the pixels in the range using the box filter
            output_arr[y, x] = np.sum(img[ymin:ymax, xmin:xmax] * bmh_filter, axis=(0, 1))

    # Convert the output array to an image and save it
    cv2.imwrite(img_save_name, output_arr)
    # output_img = Image.fromarray(output_arr)
    # output_img.save(f'{blender_superss}_BMHdownsampled_OB3.png')


def downsampling_filter(img):
    img_filtered = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j].tolist() == [0, 0, 0]:
                continue
            else:
                obp_count = 1
                # print(img[i, j])
            check_neighbors = get_point_neighbors((i, j), img.shape, 8)
            for cn in check_neighbors:
                if img[cn[0], cn[1]].tolist() != [0, 0, 0]:
                    obp_count += 1
            if obp_count > 2:
                img_filtered[i, j, :] = np.array(img[i, j, :])
    return img_filtered


def filter_with_square(img_array, radius, delete_threshold):
    binary_mask = ((np.array(img_array[:, :, 0]) > 0 ) * 1).reshape(1080, 1080, 1)
    # print(binary_mask.shape)
    # print(Counter(binary_mask.reshape(-1)))
    for i in range(radius, img_array.shape[0]-radius):
        for j in range(radius, img_array.shape[1]-radius):
            area_count = None
            xmin = i - radius  # np.min(i - calulating_radius / 2, 0) 
            xmax = i + radius  # +1 weird yet effective
            ymin = j - radius  # np.min(j - calulating_radius / 2, 0)
            ymax = j + radius  # +1 
            if img_array[i, j, :].tolist() != [0, 0, 0]:
                # print(binary_mask[xmin:xmax, ymin:ymax, :])
                area_count = np.sum(binary_mask[xmin:xmax, ymin:ymax, :], axis=(0, 1))[0]
                # print(area_count)
            if area_count is None:
                continue
            # if radius == 1 and area_count == 1:
            #     binary_mask[i, j] = 0
            #     continue
            if area_count < delete_threshold:
                binary_mask[i, j] = 0
                # binary_mask[xmin:xmax, ymin:ymax] = 0 # np.zeros((xmax-xmin, ymax-ymin, 1))

    return img_array * binary_mask


def vis_disob(img):
    disob = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j,:].tolist() != [0,0,0]:
                disob[i, j, :] = [255,255,255]
    # print(disob)
    return disob
    

def filter_and_vis(root_img_path, img_list):
  
    # img_list = os.listdir(root_img_path)
    for i in tqdm(img_list): 
    # for i in img_list: 
       
        key = str(i).zfill(5)
        c_path = root_img_path + key + "/"
        
        file_list = os.listdir(c_path)

        if "dis_fOB.png" in file_list:
            # print("Already have filtered OB, ", key)
            continue

        if "dis_OB.png" not in file_list:
            print("Do not have OB, ", key)
            continue

        f_img = cv2.imread(c_path+"dis_OB.png", cv2.IMREAD_UNCHANGED)
        f_img = vis_disob(f_img)
        # cv2.imwrite(c_path + "dis_OB_vis.png", f_img)

        if key in ["18316", "00427", "14699", "17419"]:
            f_img = f_img
        else:
            f_img = filter_with_square(f_img, 6, 10)
            f_img = filter_with_square(f_img, 4, 6)
            f_img = filter_with_square(f_img, 9, 15)  
            # # f_img = filter_with_square(f_img, 4, 5)
            f_img = filter_with_square(f_img, 2, 3)
            f_img = filter_with_square(f_img, 2, 3)
            # f_img = filter_with_square(f_img, 4, 3)  

        # f_img = cv2.imread(c_path+"dis_fOB.png", cv2.IMREAD_UNCHANGED)

        b_mask = ((np.array(f_img[:, :, 0]) > 0 ) * 1).reshape(1080, 1080, 1)
        ob_vis = b_mask * [255, 255, 255]
        cv2.imwrite(c_path+"dis_fOB.png", ob_vis)

        oo_vis = cv2.imread(c_path+"dis_OO.png", cv2.IMREAD_UNCHANGED)
        oo_vis = vis_oo(oo_vis, ob_vis)
        cv2.imwrite(c_path+"dis_fOO.png", oo_vis)



def vis_oo(oo, ob):
    vis_oo = np.ones((ob.shape[0], ob.shape[1], 3))*112
    for i in range(ob.shape[0]):
        for j in range(ob.shape[1]):
            if ob[i, j, :].tolist() != [0,0,0]:
                vis_oo[i,j,:] = oo[i,j,:]

    return vis_oo




if __name__ == '__main__':

    # Load the supersampled image
    img = cv2.imread("../occdata/occ1080downsamples/00000/ss_OB3.png")

    blender_superss = 4
    save_dir = "../occdata/occ1080downsamples/00000/"
    
    # cv2 resize test
    # resize_img = cv2.resize(img, (img.shape[0]//blender_superss, img.shape[1]//blender_superss),interpolation = cv2.INTER_LINEAR) 
    # cv2.imwrite("../occdata/occ1080downsamples/00000/resize_OB3.png", resize_img)              
    
    img = np.array(img)
    # average_filter_downsample(img, blender_superss, save_dir + "down_OB3.png", keep_ob3color=True, simple_filtered=False)

    img = cv2.imread("../occdata/occ1080downsamples/00000/OB.png")
    img = np.array(img)
    # average_filter_downsample(img, blender_superss, save_dir + "down_OB.png", keep_ob3color=False, simple_filtered=False)
    
    img = cv2.imread("../occdata/occ1080downsamples/00000/ss_OO.png")
    img = np.array(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j].tolist() == [112,112,112]:
                img[i,j] = [np.nan, np.nan, np.nan]
    average_filter_downsample(img, blender_superss, save_dir + "down_OO_test2.png", keep_ob3color=False, simple_filtered=False, is_oo_img=True)
    

