import numpy as np
import cv2
import os

 
# img_h, img_w = 108, 108
img_h, img_w = 54, 54
means, stdevs = [], []
img_list = []
 
imgs_path = './occdata/synocc/'
imgs_path_list = os.listdir(imgs_path)
 
len_ = len(imgs_path_list)

for item in imgs_path_list:
    img = cv2.imread(os.path.join(imgs_path, item+"/rgb_1.png"))
    img = cv2.resize(img,(img_w,img_h))
    img = img[:, :, :, np.newaxis]
    img_list.append(img)

 
imgs = np.concatenate(img_list, axis=3)
imgs = imgs.astype(np.float32) / 255.
 
for i in range(3):
    pixels = imgs[:, :, i, :].ravel()
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))
 
# BGR --> RGB
means.reverse()
stdevs.reverse()
 
print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))

# 108
# normMean = [0.7801186, 0.7713451, 0.7718106]
# normStd = [0.20322366, 0.20654093, 0.21719876]

# 54
# normMean = [0.7801772, 0.77142173, 0.77184254]
# normStd = [0.20326443, 0.20657785, 0.21720478]