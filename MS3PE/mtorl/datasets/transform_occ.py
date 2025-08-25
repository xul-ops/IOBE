import cv2
import math
import random
import numpy as np
from torch import Tensor
import torchvision.transforms.functional as F
from torchvision import transforms as tv_transforms
import albumentations as A
# from albumentations.pytorch import ToTensorV2


def get_A_transform(crop_size):
    """
    https://albumentations.ai/
    """
    # Declare an augmentation pipeline
    data_transforms = {
        "co_trans": A.Compose([
            # # A.Resize(720, 720, interpolation=cv2.INTER_NEAREST, p=1.0),
            # A.RandomScale(scale_limit=(-0.3, 0.1), p=0.5),
            A.RandomCrop(width=crop_size, height=crop_size, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5)], p=1.0, additional_targets={'GT_occ': 'image'}),

        "train_trans": A.Compose([
            A.Blur(p=0.2),
            A.RandomBrightnessContrast(p=0.2),
            # A.ColorJitter(p=0.2)
            # A.OneOf([
            #     # A.GridDistortion(num_steps=5, distort_limit=0.05, p=0.2),
            #     # A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.2),
            #     # A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2)
            #     A.IAAAdditiveGaussianNoise(p=0.2),
            #     A.GaussNoise(p=0.2) ], p=0.5),
            # # A.CoarseDropout(max_holes=8, max_height=, max_width=, min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
        ], p=0.7)}

    return data_transforms


def transform_with_A(data_transforms, img, img_label, additional_train_trans=True):
    # Augment an image
    transformed_co = data_transforms["co_trans"](image=img, GT_occ=img_label)

    if additional_train_trans:
        transformed_train = data_transforms["train_trans"](image=transformed_co["image"])
        return transformed_train["image"], transformed_co["GT_occ"]
    else:
        return transformed_co["image"], transformed_co["GT_occ"]


def crop_with_label(img, label, crop_size):
    w, h = img.size
    img_center = np.array([h / 2, w / 2]).astype(np.int32)
    img = np.array(img)
    avg_chans = np.mean(img, axis=(0, 1))  # (3,)

    offset_x, offset_y = 0, 0
    offset = True
    if offset:
        # [ 0, 1 )
        center_off = 1200 - crop_size
        offset_y = int(center_off * (random.random() - 0.5))
        offset_x = int(center_off * (random.random() - 0.5))
    img_center = [img_center[0] + offset_y, img_center[1] + offset_x]
    # one crop
    img_crop, label_crop = get_subwindow(img, label, img_center, crop_size, avg_chans)

    return img_crop, label_crop


def get_subwindow(im, label, center_pos, original_sz, avg_chans):
    """
     img
     pos: center
     original_sz: crop patch size = 320
    """
    if isinstance(center_pos, float):
        center_pos = [center_pos, center_pos]
    sz = original_sz
    im_sz = im.shape  ## H,W
    c = (original_sz + 1) / 2  # 320/2 = 160

    context_xmin = round(center_pos[0] - c)  # floor(pos(2) - sz(2) / 2);
    context_xmax = context_xmin + sz - 1
    context_ymin = round(center_pos[1] - c)  # floor(pos(1) - sz(1) / 2);
    context_ymax = context_ymin + sz - 1

    left_pad = int(max(0., -context_xmin))
    top_pad = int(max(0., -context_ymin))
    right_pad = int(max(0., context_xmax - im_sz[1] + 1))
    bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

    ## for example, if context_ymin<0, now context_ymin = 0
    context_xmin = context_xmin + left_pad
    context_xmax = context_xmax + left_pad
    context_ymin = context_ymin + top_pad
    context_ymax = context_ymax + top_pad

    # zzp: a more easy speed version
    r, c, k = im.shape
    # avg_chans = np.array(avg_chans).reshape(3,)
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        te_im = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k),
                         np.uint8)  # 0 is better than 1 initialization
        te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im

        if top_pad:
            te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_im[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_im[:, c + left_pad:, :] = avg_chans
        im_patch_original = te_im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]
    else:
        im_patch_original = im[int(context_ymin):int(context_ymax + 1), int(context_xmin):int(context_xmax + 1), :]

    r, c, k = label.shape
    avg_chans = np.array([0, 0]).reshape(2, )
    if any([top_pad, bottom_pad, left_pad, right_pad]):
        te_label = np.zeros((r + top_pad + bottom_pad, c + left_pad + right_pad, k),
                            np.float32)  # 0 is better than 1 initialization
        te_label[top_pad:top_pad + r, left_pad:left_pad + c, :] = label

        if top_pad:
            te_label[0:top_pad, left_pad:left_pad + c, :] = avg_chans
        if bottom_pad:
            te_label[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
        if left_pad:
            te_label[:, 0:left_pad, :] = avg_chans
        if right_pad:
            te_label[:, c + left_pad:, :] = avg_chans
        label_patch_original = te_label[int(context_ymin):int(context_ymax + 1),
                               int(context_xmin):int(context_xmax + 1), :]
    else:
        label_patch_original = label[int(context_ymin):int(context_ymax + 1),
                               int(context_xmin):int(context_xmax + 1), :]

    return im_patch_original, label_patch_original


# transform in mtorl
class Normalize(object):
    def __init__(self, mean, std):
        assert isinstance(mean, list)
        assert isinstance(std, list)
        self.mean = mean
        self.std = std
        self.normalize = tv_transforms.Normalize(mean, std)

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']

        if type(image) == list:
            for i in range(len(image)):
                image[i] = self.normalize(image[i])
        else:
            image = self.normalize(image)

        return {'image': image, 'labels': labels}


class RandomCrop(tv_transforms.RandomCrop):

    def forward(self, sample):
        image, labels = sample['image'], sample['labels']

        _, height, width = image.size()
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            image = F.pad(image, padding, self.fill, self.padding_mode)
            labels = F.pad(labels, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            image = F.pad(image, padding, self.fill, self.padding_mode)
            labels = F.pad(labels, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(image, self.size)
        image = F.crop(image, i, j, h, w)
        labels = F.crop(labels, i, j, h, w)
        return {'image': image, 'labels': labels}


class RandomRotation(tv_transforms.RandomRotation):

    def forward(self, sample):
        image, labels = sample['image'], sample['labels']

        image_fill = self.fill
        if isinstance(image, Tensor):
            if isinstance(image_fill, (int, float)):
                image_fill = [float(image_fill)] * F._get_image_num_channels(image)
            else:
                image_fill = [float(f) for f in image_fill]

        labels_fill = self.fill
        if isinstance(image, Tensor):
            if isinstance(labels_fill, (int, float)):
                labels_fill = [float(labels_fill)] * F._get_image_num_channels(labels)
            else:
                labels_fill = [float(f) for f in labels_fill]

        angle = self.get_params(self.degrees)

        image = F.rotate(image, angle, self.resample, self.expand, self.center, image_fill)
        labels = F.rotate(labels, angle, self.resample, self.expand, self.center, labels_fill)

        b_label, o_label = labels[0], labels[1]
        mask = b_label == 1
        o_label[mask] += (angle * math.pi / 180.0)
        o_label[mask] %= (2 * math.pi)
        labels[0], labels[1] = b_label, o_label

        return {'image': image, 'labels': labels}
