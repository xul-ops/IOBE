import time
import re
import os
import sys
import json
import cv2
import argparse
import numpy as np
import matlab.engine
_CURR_DIR = os.path.dirname(os.path.realpath(__file__))


def run_cmd(cmd):
    print("cmd: %s" % cmd)
    os.system(cmd)


def get_matlab_eng(works_num=0):
    eng = matlab.engine.start_matlab()
    # eng.myCluster = eng.parcluster('local')
    # eng.delete(eng.myCluster.Jobs)
    eng.addpath(eng.genpath(_CURR_DIR))
    if works_num == 0:
        eng.parpool('local')
    else:
        eng.parpool('local', works_num)
    return eng
  

def imresize_test(img_path, out_path, matlab_eng=None):
    if matlab_eng is None:
        eng = get_matlab_eng()
    else:
        eng = matlab_eng
        
    eval_res = eng.Imresize_test(img_path, out_path, nargout=0)

    if matlab_eng is None:
        eng.quit()
        
    return
        

def save_m_nms(result_dir, dataset='PIOD',  matlab_eng=None, ep_delete=0):
    if matlab_eng is None:
        eng = get_matlab_eng()
    else:
        eng = matlab_eng
        
    eval_res = eng.EdgeNMS(result_dir, list_dir, nargout=0)

    if matlab_eng is None:
        eng.quit()
        
    return
      
        
def save_m_nms_old(result_dir, dataset='PIOD',  matlab_eng=None, ep_delete=0):
    if matlab_eng is None:
        eng = get_matlab_eng()
    else:
        eng = matlab_eng
        
    if args.dataset not in ['nyuv2']:
        list_dir = os.listdir(result_dir)
        list_dir = [item.replace(".jpg", "") for item in list_dir if item.endswith(".jpg")]
        # list_dir = [item.replace("_ob.png", "") for item in list_dir if item.endswith("_ob.png")]
        eval_res = eng.EdgeNMS(result_dir, list_dir, nargout=0)
    else:
        eval_res = eng.EdgeNMS(dataset, result_dir, nargout=0)

    if matlab_eng is None:
        eng.quit()
        
    return


def get_nmst(result_dir, ep_threhold=0.7):


    list_dir = os.listdir(result_dir)
    names_list_dir = [ item.replace("_ob.png", "") for item in list_dir if item.endswith("_ob.png")]


    for img_id in range(len(names_list_dir)):
        image1 = cv2.imread(os.path.join(result_dir, names_list_dir[img_id] + "_nms.png"),  cv2.IMREAD_UNCHANGED)
        nmst = np.zeros((image1.shape[0], image1.shape[1]))        
        for i in range(image1.shape[0]):
            for j in range(image1.shape[1]):
                if (255 -image1[i, j]) / 255 >=  ep_threhold:  
                    nmst[i, j] = 255
        current_save_name = result_dir+"/" + names_list_dir[img_id]+"_nms_t7.png"
        # os.remove(result_dir+"/" + names_list_dir[img_id]+"_final.png")
        cv2.imwrite(current_save_name, nmst)
        
    return 
    
    
def get_obt(result_dir, ep_threhold=0.9):


    list_dir = os.listdir(result_dir)
    names_list_dir = [ item.replace("_ob.png", "") for item in list_dir if item.endswith("_ob.png")]


    for img_id in range(len(names_list_dir)):
        image1 = cv2.imread(os.path.join(result_dir, names_list_dir[img_id] + "_ob.png"),  cv2.IMREAD_UNCHANGED)
        nmst = np.zeros((image1.shape[0], image1.shape[1]))        
        for i in range(image1.shape[0]):
            for j in range(image1.shape[1]):
                if (255 -image1[i, j]) / 255 >=  ep_threhold:  
                    nmst[i, j] = 255
        current_save_name = result_dir+"/" + names_list_dir[img_id]+"_ob_t9.png"
        # os.remove(result_dir+"/" + names_list_dir[img_id]+"_final.png")
        cv2.imwrite(current_save_name, nmst)
        
    return     


def adding_nms2rgb(result_dir, ep_threhold=0.9):


    list_dir = os.listdir(result_dir)
    names_list_dir = [ item.replace("_ob.png", "") for item in list_dir if item.endswith("_ob.png")]
    # nms_list_dir = [ os.path.join(result_dir, item) for item in list_dir if item.endswith("_nms.png")]
    # rgb_list_dir = [ os.path.join(result_dir, item) for item in list_dir if item.endswith("_rgb.png")]

    for img_id in range(len(names_list_dir)):
        # image1 = cv2.imread(rgb_list_dir[img_id], cv2.IMREAD_UNCHANGED)
        # image2 = cv2.imread(nms_list_dir[img_id],  cv2.IMREAD_UNCHANGED)
        image1 = cv2.imread(os.path.join(result_dir, names_list_dir[img_id] + "_rgb.png"), cv2.IMREAD_UNCHANGED)
        image2 = cv2.imread(os.path.join(result_dir, names_list_dir[img_id] + "_nms.png"),  cv2.IMREAD_UNCHANGED)        
        try:
            assert image1.shape[0] == image2.shape[0] and (image1.shape[1] == image2.shape[1])
        except AssertionError:
            # print("Img pair shape not match at ", result_dir+names_list_dir[img_id]) 
            # print(image1.shape, image2.shape)
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]), interpolation=cv2.INTER_LINEAR)
            # print("New shape pair is")
            # print(image1.shape, image2.shape)
        for i in range(image1.shape[0]):
            for j in range(image1.shape[1]):
                if (255 -image2[i, j]) / 255 >=  ep_threhold:  # image2[i, j] != 0
                    image1[i, j, :] = [255, 0, 0] 
        current_save_name = result_dir+"/" + names_list_dir[img_id]+"_rgbgt_t9.png"
        # os.remove(result_dir+"/" + names_list_dir[img_id]+"_final.png")
        cv2.imwrite(current_save_name, image1)
        
    return     


def parse_args():
    parser = argparse.ArgumentParser('Matlab Edge NMS')
    # parser.add_argument('--occ', type=int, default=0, help='occ')
    # parser.add_argument('--zip-dir', type=str, default=None, help='zip-dir')
    parser.add_argument('--result_dir', type=str, default=None, help='result_dir')
    # parser.add_argument('--zipfile', type=str, default=None, help='zipfile')
    parser.add_argument('--dataset', type=str, default='PIOD', help='dataset')
    # parser.add_argument('--maxdist', type=float, default=0.0075, help='maxdist')
    parser.add_argument('--works_num', type=int, default=2, help='works_num')
    return parser.parse_args()


if __name__ == '__main__':
    # imresize_test("./11879/pred.png", './11879/downsampling_test')
    # import pdb
    # pdb.set_trace()
    args = parse_args()
    if args.result_dir is not None:
        print('@' * 80 + '\n')
        start_time = time.time()
        eng = get_matlab_eng(args.works_num)
        list_dir = os.listdir(args.result_dir)
        list_dir = [args.result_dir]
        # multi dir nms
        for item in list_dir:
            current_dir = args.result_dir # + item +"/pre/"
            # cc_list = os.listdir(current_dir)
            # for item in cc_list:
            #     if item.startswith("epoch") and not item.endswith(".tar"):
            #         current_dir = current_dir +item +"/"
            #         break
            print(current_dir)
            save_m_nms_old(current_dir, dataset=args.dataset, matlab_eng=eng, ep_delete=0)
            print("Finish the nms process, store at ", current_dir)
            get_nmst(current_dir)
            
            pdb.set_trace()
            
            # get_obt(current_dir)
            # adding_nms2rgb(current_dir)
        print('time using: %f' % (time.time() - start_time))
        print('@' * 80 + '\n')
