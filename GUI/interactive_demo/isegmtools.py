from functools import lru_cache

import cv2
import numpy as np
from copy import deepcopy


class Clicker(object):
    def __init__(self, gt_mask=None, init_clicks=None, ignore_label=-1, click_indx_offset=0):
        self.click_indx_offset = click_indx_offset
        if gt_mask is not None:
            self.gt_mask = gt_mask == 1
            self.not_ignore_mask = gt_mask != ignore_label
        else:
            self.gt_mask = None

        self.reset_clicks()

        if init_clicks is not None:
            for click in init_clicks:
                self.add_click(click)

    def make_next_click(self, pred_mask):
        assert self.gt_mask is not None
        click = self._get_next_click(pred_mask)
        self.add_click(click)

    def get_clicks(self, clicks_limit=None):
        return self.clicks_list[:clicks_limit]

    def _get_next_click(self, pred_mask, padding=True):
        fn_mask = np.logical_and(np.logical_and(self.gt_mask, np.logical_not(pred_mask)), self.not_ignore_mask)
        fp_mask = np.logical_and(np.logical_and(np.logical_not(self.gt_mask), pred_mask), self.not_ignore_mask)

        if padding:
            fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), 'constant')
            fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), 'constant')

        fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
        fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)

        if padding:
            fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
            fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

        fn_mask_dt = fn_mask_dt * self.not_clicked_map
        fp_mask_dt = fp_mask_dt * self.not_clicked_map

        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)

        is_positive = fn_max_dist > fp_max_dist
        if is_positive:
            coords_y, coords_x = np.where(fn_mask_dt == fn_max_dist)  # coords is [y, x]
        else:
            coords_y, coords_x = np.where(fp_mask_dt == fp_max_dist)  # coords is [y, x]

        return Click(is_positive=is_positive, coords=(coords_y[0], coords_x[0]))

    def add_click(self, click):
        coords = click.coords

        click.indx = self.click_indx_offset + self.num_pos_clicks + self.num_neg_clicks
        if click.is_positive:
            self.num_pos_clicks += 1
        else:
            self.num_neg_clicks += 1

        self.clicks_list.append(click)
        if self.gt_mask is not None:
            self.not_clicked_map[coords[0], coords[1]] = False

    def _remove_last_click(self):
        click = self.clicks_list.pop()
        coords = click.coords

        if click.is_positive:
            self.num_pos_clicks -= 1
        else:
            self.num_neg_clicks -= 1

        if self.gt_mask is not None:
            self.not_clicked_map[coords[0], coords[1]] = True

    def reset_clicks(self):
        if self.gt_mask is not None:
            self.not_clicked_map = np.ones_like(self.gt_mask, dtype=np.bool)

        self.num_pos_clicks = 0
        self.num_neg_clicks = 0

        self.clicks_list = []

    def get_state(self):
        return deepcopy(self.clicks_list)

    def set_state(self, state):
        self.reset_clicks()
        for click in state:
            self.add_click(click)

    def __len__(self):
        return len(self.clicks_list)


class Click:
    def __init__(self, is_positive, coords, indx=None):
        self.is_positive = is_positive
        self.coords = coords
        self.indx = indx

    @property
    def coords_and_indx(self):
        return (*self.coords, self.indx)

    def copy(self, **kwargs):
        self_copy = deepcopy(self)
        for k, v in kwargs.items():
            setattr(self_copy, k, v)
        return self_copy



def visualize_instances(imask, bg_color=255,
                        boundaries_color=None, boundaries_width=1, boundaries_alpha=0.8):
    num_objects = imask.max() + 1
    palette = get_palette(num_objects)
    if bg_color is not None:
        palette[0] = bg_color

    result = palette[imask].astype(np.uint8)
    if boundaries_color is not None:
        boundaries_mask = get_boundaries(imask, boundaries_width=boundaries_width)
        tresult = result.astype(np.float32)
        tresult[boundaries_mask] = boundaries_color
        tresult = tresult * boundaries_alpha + (1 - boundaries_alpha) * result
        result = tresult.astype(np.uint8)

    return result


@lru_cache(maxsize=16)
def get_palette(num_cls):
    palette = np.zeros(3 * num_cls, dtype=np.int32)

    for j in range(0, num_cls):
        lab = j
        i = 0

        while lab > 0:
            palette[j*3 + 0] |= (((lab >> 0) & 1) << (7-i))
            palette[j*3 + 1] |= (((lab >> 1) & 1) << (7-i))
            palette[j*3 + 2] |= (((lab >> 2) & 1) << (7-i))
            i = i + 1
            lab >>= 3

    return palette.reshape((-1, 3))


def visualize_mask(mask, num_cls):
    palette = get_palette(num_cls)
    mask[mask == -1] = 0

    return palette[mask].astype(np.uint8)


def visualize_proposals(proposals_info, point_color=(255, 0, 0), point_radius=1):
    proposal_map, colors, candidates = proposals_info

    proposal_map = draw_probmap(proposal_map)
    for x, y in candidates:
        proposal_map = cv2.circle(proposal_map, (y, x), point_radius, point_color, -1)

    return proposal_map


def draw_probmap(x):
    return cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_HOT)


def draw_points(image, points, color, radius=3):
    image = image.copy()
    for p in points:
        if p[0] < 0:
            continue
        if len(p) == 3:
            pradius = {0: 8, 1: 6, 2: 4}[p[2]] if p[2] < 3 else 2
        else:
            pradius = radius
        image = cv2.circle(image, (int(p[1]), int(p[0])), pradius, color, -1)

    return image


def draw_instance_map(x, palette=None):
    num_colors = x.max() + 1
    if palette is None:
        palette = get_palette(num_colors)

    return palette[x].astype(np.uint8)


def blend_mask(image, mask, alpha=0.6):
    if mask.min() == -1:
        mask = mask.copy() + 1

    imap = draw_instance_map(mask)
    result = (image * (1 - alpha) + alpha * imap).astype(np.uint8)
    return result


def get_boundaries(instances_masks, boundaries_width=1):
    boundaries = np.zeros((instances_masks.shape[0], instances_masks.shape[1]), dtype=np.bool)

    for obj_id in np.unique(instances_masks.flatten()):
        if obj_id == 0:
            continue

        obj_mask = instances_masks == obj_id
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        inner_mask = cv2.erode(obj_mask.astype(np.uint8), kernel, iterations=boundaries_width).astype(np.bool)

        obj_boundary = np.logical_xor(obj_mask, np.logical_and(inner_mask, obj_mask))
        boundaries = np.logical_or(boundaries, obj_boundary)
    return boundaries
    
 
def draw_with_blend_and_clicks(img, mask=None, alpha=0.6, clicks_list=None, pos_color=(0, 255, 0),
                               neg_color=(255, 0, 0), radius=1):
    result = img.copy()

    if mask is not None:
        palette = get_palette(np.max(mask) + 1)
        rgb_mask = palette[mask.astype(np.uint8)]

        mask_region = (mask > 0).astype(np.uint8)
        result = result * (1 - mask_region[:, :, np.newaxis]) + \
            (1 - alpha) * mask_region[:, :, np.newaxis] * result + \
            alpha * rgb_mask
        result = result.astype(np.uint8)

        # result = (result * (1 - alpha) + alpha * rgb_mask).astype(np.uint8)

    if clicks_list is not None and len(clicks_list) > 0:
        pos_points = [click.coords for click in clicks_list if click.is_positive]
        neg_points = [click.coords for click in clicks_list if not click.is_positive]

        result = draw_points(result, pos_points, pos_color, radius=radius)
        result = draw_points(result, neg_points, neg_color, radius=radius)

    return result

