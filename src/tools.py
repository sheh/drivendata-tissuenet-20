import math
from typing import List

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import pandas as pd
from torchvision.transforms import transforms

ERROR_TABLE = np.array([
    [0.0, 0.1, 0.7, 1.0],
    [0.1, 0.0, 0.3, 0.7],
    [0.7, 0.3, 0.0, 0.3],
    [1.0, 0.7, 0.3, 0.0]
])


def tissue_loc(original_img, max_locs=6, pad=0, white_pad=0.1, show=False):
    if show:
        fig, ax = plt.subplots(2, 2, figsize=(40, 40))

    img = original_img.copy()
    height, width = img.shape[:2]

    gray_s1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    y1_w, x1_w = np.argwhere(gray_s1 != 255).min(axis=0)
    y2_w, x2_w = np.argwhere(gray_s1 != 255).max(axis=0)
    gray_s1 = gray_s1[y1_w:y2_w, x1_w:x2_w]

    x_margin = int(white_pad*gray_s1.shape[1])
    y_margin = int(white_pad*gray_s1.shape[0])
    gray_s1[:y_margin, :] = 255
    gray_s1[-y_margin:, :] = 255
    gray_s1[:, :x_margin] = 255
    gray_s1[:, -x_margin:] = 255

    if show:
        ax[0, 0].imshow(gray_s1, cmap='gray')
        ax[0, 0].set_title('gray_s1')

    contours_s1 = find_contours(gray_s1)

    gray_s2 = gray_s1  # fill_black_lines_whit(contours_s1, gray_s1)
    if show:
        draw = gray_s2.copy()
        cv2.drawContours(draw, contours_s1, -1, (255, 0, 0), 5)
        ax[0, 1].imshow(draw, cmap='gray')
        ax[0, 1].set_title('without black lines')

    contours_s2 = find_contours(gray_s2)
    rects_s2 = contours2rects(contours_s2, pad, gray_s2.shape[1], gray_s2.shape[0])
    if show:
        draw = gray_s2.copy()
        cv2.drawContours(draw, contours_s2, -1, (255, 0, 0), 5)
        for r in rects_s2:
            cv2.rectangle(draw, (r[0], r[1]), (r[2], r[3]), (0, 0, 255), 3)
        ax[1, 0].imshow(draw, cmap='gray')
        ax[1, 0].set_title('step2')

    ret = []
    for rect in rects_s2:
        x1, y1, x2, y2 = rect
        biggest_rect = gray_s2[y1:y2, x1:x2]
        contours_s3 = find_contours(biggest_rect)
        rects_s3 = contours2rects(contours_s3, pad, x2-x1, y2-y1)

        if show and rects_s3.shape[0]:
            draw = biggest_rect.copy()
            cv2.drawContours(draw, contours_s3, -1, (255, 0, 0), 5)
            for r in rects_s3:
                cv2.rectangle(draw, (r[0], r[1]), (r[2], r[3]), (0, 0, 255), 3)
            ax[1, 1].imshow(draw, cmap='gray')
            ax[1, 1].set_title('step3')
            plt.show()
        for r in rects_s3:
            ret.append(r + np.array([x1 + x1_w, y1 + y1_w, x1 + x1_w, y1 + y1_w]))
    return ret[:max_locs]


def contours2rects(contours, pad, width, height, ratio_limit=7.0, min_area=0.1):
    # calc bounding boxes
    rects = sorted(map(cv2.boundingRect, contours), key=lambda x: x[2] * x[3], reverse=True)
    # filter by ration
    rects = list(filter(lambda x: (1 / ratio_limit) < x[2] / x[3] < (ratio_limit / 1), rects))
    # filter by area
    rects = list(filter(lambda x: x[2] * x[3] / (rects[0][2] * rects[0][3]) > min_area, rects))
    # add padding and convert to x1, y1, x2, y2
    rects = list(map(lambda x: [
        max(int(x[0] - pad * x[2]), 0),
        max(int(x[1] - pad * x[3]), 0),
        min(int(x[0] + pad * x[2] + x[2]), width),
        min(int(x[1] + pad * x[3] + x[3]), height),
    ], rects))
    return np.array(rects)


def find_contours(img_gray):
    blur = cv2.GaussianBlur(img_gray, (25, 25), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    noise_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, noise_kernel, iterations=1)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (14, 14))
    close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    contours, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # fig, ax = plt.subplots(3, 1)
    # ax[0].imshow(thresh, cmap='gray')
    # ax[1].imshow(opening, cmap='gray')
    # ax[2].imshow(close, cmap='gray')
    # plt.show()
    return contours


def fill_black_lines_whit(contours, gray):
    ret = gray.copy()
    for i in range(len(contours)):
        cimg = np.zeros_like(gray)
        cv2.drawContours(cimg, contours, i, color=255, thickness=-1)

        # Access the image pixels and create a 1D numpy array then add to list
        pts = np.where(cimg == 255)
        if ret[pts[0], pts[1]].mean() < 50:
            ret[pts[0], pts[1]] = 255
    return ret

# model_path = "../checkpoint.pth"
# region_model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
#                               in_channels=3, out_channels=1, init_features=32, pretrained=True)
# region_model.load_state_dict(torch.load(model_path))
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# region_model = region_model.to(device)
# region_trans = transforms.Compose([
#   transforms.ToTensor(),
#   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
# ])
# region_img_size = 256
#
#
# def get_predicted_regions(tissue_img, image_resolution_mk, max_regions=36):
#     org_size = tissue_img.shape
#     tissue_img = cv2.resize(tissue_img, (region_img_size, region_img_size))
#
#     input_tensor = region_trans(tissue_img).unsqueeze(0)
#     with torch.no_grad():
#         pred = region_model(input_tensor.to(device))
#         pred = pred.detach().cpu().numpy().squeeze()
#     pred = cv2.resize(pred, (org_size[1], org_size[0]))
#     pred = (pred / pred.max() * 255).astype(np.uint8)
#
#     sz = int(300 / image_resolution_mk)
#     sum = cv2.resize(pred, (pred.shape[1] // sz, pred.shape[0] // sz))
#
#     masked = np.argwhere(sum > 230)
#     some_id = np.arange(masked.shape[0])
#     np.random.shuffle(some_id)
#     region_ids = masked[some_id[:max_regions]]
#     region_ids[:, [0, 1]] = region_ids[:, [1, 0]]
#     return np.hstack([
#         region_ids*sz,
#         region_ids*sz+sz
#     ])


def get_regions(tissue_img, image_resolution_mk, max_regions=36):
    height, width = tissue_img.shape[:2]
    gray = cv2.cvtColor(tissue_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (25, 25), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    noise_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, noise_kernel, iterations=2)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (14, 14))
    close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, close_kernel, iterations=3)

    contours, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sz = int(300 / image_resolution_mk)

    area_contours = sorted([(cv2.contourArea(cnt), cnt) for cnt in contours],
                           key=lambda x: x[0], reverse=True)

    prev_rect = None
    bbs = []
    for pt in area_contours[0][1]:
        pt = pt[0]
        max_sum = -1
        for j in range(4):

            if j == 0:
                x1 = pt[0]
                y1 = pt[1]
                x2 = x1 + sz
                y2 = y1 + sz
            elif j == 1:
                x2 = pt[0]
                y1 = pt[1]
                x1 = x2 - sz
                y2 = y1 + sz
            elif j == 2:
                x2 = pt[0]
                y2 = pt[1]
                x1 = x2 - sz
                y1 = y2 - sz
            elif j == 3:
                x1 = pt[0]
                y2 = pt[1]
                x2 = x1 + sz
                y1 = y2 - sz

            s = close[y1:y2, x1:x2].sum()
            if s > max_sum:
                max_sum = s
                rect = (x1, y1, x2, y2)
        if prev_rect is None or calc_iou(prev_rect, rect) < 0.1:
            prev_rect = rect
            bbs.append(rect)
            if len(bbs) >= max_regions:
                break
    return bbs


def make_gaussian(size, fwhm=3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)


def calc_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def clr_by_cls(cls):
    if cls == 0:
        return 35, 222, 110
    elif cls == 1:
        return 35, 135, 222
    elif cls == 2:
        return 222, 210, 35
    elif cls == 3:
        return 252, 40, 34
    elif cls is None:
        return 255, 255, 255


def draw_annotations(img, annotations, thickness=3, line_type=cv2.FILLED, class_n=4):
    for a in annotations:
        # lbl = float(a.label) if class_n is None else a.label/class_n
        # clr = (np.array(cm.jet(lbl))[:3]*255)
        # clr = (int(clr[0]), int(clr[1]), int(clr[2]))
        clr = clr_by_cls(a.label)
        cv2.rectangle(img, a.bb.pt1, a.bb.pt2, clr, thickness, lineType=line_type)
        cv2.putText(img, f"{a.label}", a.bb.center,
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0))


def draw_gt_annotations(img, annotations, class_n=4):
    draw_annotations(img, annotations, thickness=5, class_n=class_n)


def draw_pred_annotations(img, annotations, class_n=4):
    draw_annotations(img, annotations, thickness=1, line_type=cv2.LINE_4, class_n=class_n)


class BBox:

    def __init__(self, x1, y1, x2, y2, page=None):
        self.x1 = int(round(x1))
        self.y1 = int(round(y1))
        self.x2 = int(round(x2))
        self.y2 = int(round(y2))
        self.page = page

    @property
    def w(self):
        return self.x2 - self.x1

    @property
    def h(self):
        return self.y2 - self.y1

    @property
    def pt1(self):
        return self.x1, self.y1

    @property
    def pt2(self):
        return self.x2, self.y2

    @property
    def center(self):
        return self.x1 + self.w // 2, self.y1 + self.h // 2

    @property
    def xywh(self):
        return self.x1, self.y1, self.w, self.h

    @property
    def s_img(self):
        return np.s_[self.y1:self.y2, self.x1:self.x2]

    def with_offset(self, x_offset, y_offset):
        return BBox(self.x1+x_offset, self.y1+y_offset, self.x2 + x_offset, self.y2 + y_offset, self.page)

    # def with_scale(self, scale):
    #     return BBox(self.x1 * scale, self.y1 * scale, self.x2 * scale, self.y2 * scale)
    #
    # def with_new_coordinates(self, x_offset, y_offset, scale):
    #     return self.with_offset(x_offset, y_offset).with_scale(scale)

    def with_pad(self, pad):
        return BBox(self.x1 - int(pad*self.w), self.y1 - int(pad*self.h),
                    self.x2 + int(pad*self.w), self.y2 + int(pad*self.h), self.page)

    def with_fit_image(self, width, height):
        return BBox(max(self.x1, 0), max(self.y1, 0), min(width, self.x2), min(height, self.y2), self.page)

    def calc_iou(self, bb):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(self.x1, bb.x1)
        yA = max(self.y1, bb.y1)
        xB = min(self.x2, bb.x2)
        yB = min(self.y2, bb.y2)
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (self.w + 1) * (self.h + 1)
        boxBArea = (bb.w + 1) * (bb.h + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou

    def contains(self, bb):
        if bb.page != self.page:
            bb = bb.with_page(self.page)
        return self.x1 <= bb.x1 and self.y1 <= bb.y1 and \
               self.x2 >= bb.x2 and self.y2 >= bb.y2

    def contains_point(self, x, y):
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def filter_contains(self, bbs):
        return list(filter(self.contains, bbs))

    def with_page(self, page):
        assert self.page is not None, "Page operation requites 'page' attr"
        curr_scale = 2**self.page
        new_scale = 2**page
        scale = curr_scale / new_scale
        return BBox(self.x1 * scale, self.y1 * scale, self.x2 * scale, self.y2 * scale, page)

    def set_page(self, page):
        assert self.page is None, "Page is not set yet"
        self.page = page

    def __repr__(self):
        return f"[{self.x1} {self.y1} {self.x2} {self.y2}]" + (f" p={self.page}" if self.page else "")


def create_bb_grid(w, h, bb_sz) -> List[BBox]:
    if w < bb_sz or h < bb_sz:
        return [BBox(0, 0, bb_sz, bb_sz)]
    a = np.mgrid[0:math.ceil(h/bb_sz), 0:math.ceil(w/bb_sz)] * bb_sz
    a = np.dstack((a[1], a[0]))
    return list(map(lambda x: BBox(x[0], x[1], x[0]+bb_sz, x[1]+bb_sz), a.reshape(-1, 2)))


def filter_annotations(bb, annotations):
    return list(filter(lambda a: bb.contains(a.bb), annotations))


class Annotation:

    def __init__(self, label, bb: BBox):
        self.label = label
        self.bb = bb

    def with_page(self, page_n):
        return Annotation(self.label, self.bb.with_page(page_n))

    def with_offset(self, x, y):
        return Annotation(self.label, self.bb.with_offset(x, y))

    def __repr__(self):
        return f"{self.bb} {self.label}"


class DbgInfo:

    def __init__(self, is_on):
        self.is_on = is_on
        self._idx = -1
        self._tissues_img = []
        self._tissues = []
        self._roi_mask = []
        self._others = []
        self._roi_scaled_mask = []
        self._predicts_bb = []
        self._predicts_cls = []
        self._predicts_prob =[]

    def add_tissue(self, tissue, tissue_img, roi_mask=None, roi_scaled_mask=None):
        if not self.is_on:
            return
        self._idx += 1
        self._tissues_img.append(tissue_img)
        self._tissues.append(tissue)
        if roi_mask:
            self._roi_mask.append(roi_mask)
        if roi_scaled_mask:
            self._roi_scaled_mask.append(roi_scaled_mask)
        self._others.append([])
        self._predicts_bb.append([])
        self._predicts_cls.append([])
        self._predicts_prob.append([])

    def add_predicts(self, bbs, cls, pred_probs):
        if not self.is_on:
            return
        self._predicts_bb[self._idx].extend(bbs)
        self._predicts_cls[self._idx].extend(cls)
        self._predicts_prob[self._idx].extend(pred_probs)

    def add_others(self, images):
        if not self.is_on:
            return
        assert self._idx >= 0
        self._others[self._idx].extend(images)

    def show(self, name):
        if not self.is_on:
            return
        for i in range(len(self._tissues_img)):
            tissue_img = self._tissues_img[i]
            tissue = self._tissues[i]
            for bb, cls in zip(self._predicts_bb[i], self._predicts_cls[i]):
                bb = bb.with_page(tissue.bb.page).with_offset(-tissue.bb.x1, -tissue.bb.y1)
                cv2.rectangle(tissue_img, bb.pt1, bb.pt2, list(clr_by_cls(cls))[::-1], 1)
            for a in tissue.annotations:
                bb = a.bb.with_page(tissue.bb.page).with_offset(-tissue.bb.x1, -tissue.bb.y1)
                cv2.rectangle(tissue_img, bb.pt1, bb.pt2, list(clr_by_cls(a.label))[::-1], 4)
            cv2.imshow("", tissue_img)
            cv2.waitKey()

    def outdated_show(self, name):
        if not self.is_on:
            return
        for i in range(len(self._tissues_img)):
            tissue_img = self._tissues_img[i]
            roi_mask = self._roi_mask[i]
            roi_scaled_mask = self._roi_scaled_mask[i]
            roi_mask = (roi_mask / roi_mask.max() * 255).astype(np.uint8)
            roi_mask = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
            roi_mask = cv2.resize(roi_mask, (tissue_img.shape[1], tissue_img.shape[0]))
            tissue_img_with_mask = cv2.addWeighted(tissue_img, 0.7, roi_mask, 0.3, 0)

            for bb, cls in zip(self._predicts_bb[i], self._predicts_cls[i]):
                bb = bb.with_page(5)
                cv2.rectangle(tissue_img_with_mask, bb.pt1, bb.pt2, clr_by_cls(cls), 2)

            others = self._others[i]
            others_n = min(len(others), 6)
            fig, ax = plt.subplots(2, 1+max(math.ceil(others_n / 2.), 2))
            ax[0, 0].imshow(tissue_img_with_mask)
            ax[0, 0].set_title(name)

            # roi_scaled_mask = cv2.cvtColor((roi_scaled_mask*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            # roi_scaled_mask = cv2.resize(roi_scaled_mask, (tissue_img.shape[1], tissue_img.shape[0]))
            # tissue_img_with_mask = cv2.addWeighted(tissue_img, 0.7, roi_scaled_mask, 0.3, 0)
            ax[1, 0].imshow(roi_scaled_mask)

            for j in range(others_n):
                ax[j % 2, j//2+1].imshow(others[j])
                cls = self._predicts_cls[i][j]
                prob = self._predicts_prob[i][j][cls]
                ax[j % 2, j//2+1].set_title(f"{cls} {prob:.2f}")

            # others = self._others[i]
            # if others:
            #     fig, axes = plt.subplots(max(2, len(others)), 2)
            #     for j in range(len(others)):
            #         region_img, pred_mask, region_scale_mask = others[j]
            #         pred_mask_draw = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)
            #         region_img = cv2.addWeighted(region_img, 0.7, pred_mask_draw, 0.3, 0)
            #         axes[j, 0].imshow(region_img)
            #         axes[j, 1].imshow(region_scale_mask)
            plt.show()


def best_page_for_sz(sz_p0, sz):
    min_page = 8
    for page_n in range(min_page):
        if 2*sz > (sz_p0 / 2 ** page_n):
            return page_n
    return min_page
