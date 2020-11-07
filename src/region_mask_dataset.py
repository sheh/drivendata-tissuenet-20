import random
from collections import defaultdict

from time import time
from typing import List

import cv2
import numpy as np
import pyvips
from torch.utils.data import Dataset
from tqdm import tqdm

from src.slide import Slide, SlidesListLocalTrain
from src.tools import BBox, Annotation, filter_annotations, best_page_for_sz, clr_by_cls


class RegionMaskDataset(Dataset):
    def __init__(self, slides: List[Slide], image_size, pad=0.3, region_size_annotations=10,
                 use_each_slide_times=1, dbg=False, transforms=None):
        self._slides = slides
        self._transforms = transforms
        self._use_each_slide_times = use_each_slide_times
        self._image_size = image_size
        self._pad = pad
        self._region_size_annotations = region_size_annotations
        self._tissue_cache = defaultdict(list)
        self._dbg = dbg

    def __len__(self):
        return len(self._slides) * self._use_each_slide_times

    def __getitem__(self, index):
        index = index % len(self._slides)
        slide = self._slides[index]

        region_size_p0 = int(self._region_size_annotations * slide.annotation_size(page=0))
        page_n = best_page_for_sz(region_size_p0, self._image_size)

        if index not in self._tissue_cache:
            tissues = slide.get_tissues(max_locs=6)
            self._tissue_cache[index].extend(tissues)
        tissues = self._tissue_cache[index]

        tissue_with_annotations = list(filter(lambda x: x.has_annotations, tissues))
        cut_bb = None
        if len(tissue_with_annotations):
            tissue = random.choice(tissue_with_annotations)

            random_annotation = random.choice(tissue.annotations)

            cut_bb_rel = self.get_region_around(random_annotation, region_size_p0, slide.width, slide.height)
            cut_bb_rel.set_page(page=0)

            cut_bb = cut_bb_rel.with_page(page_n)

            try:
                cut_img, page_n_read = slide.get_page_region(crop_bb=cut_bb, pad=True)
            except pyvips.Error:
                cut_img, mask = self.blank_img_mask(cut_bb.w)

            annotations_for_mask = map(lambda a: a.with_page(page_n).with_offset(-cut_bb.x1, -cut_bb.y1),
                                       filter_annotations(cut_bb_rel, slide.annotations))
            mask = self.get_mask(cut_img, list(annotations_for_mask), pad=self._pad)
            # import matplotlib.pyplot as plt
            # plt.imshow(mask)
            # plt.show()
        else:
            # if somehow cannot find `tissue_annotations`
            cut_img, mask = self.blank_img_mask(region_size_p0)
        if self._transforms:
            cut_img, mask = self._transforms(cut_img, mask)
            mask.unsqueeze_(0)
        return (cut_img, mask) if not self._dbg else (cut_img, mask, cut_bb)

    @staticmethod
    def get_region_around(annotation, region_size, max_width, max_height):
        additive_x = random.randint(annotation.bb.w, region_size)
        x2 = min(annotation.bb.x1 + additive_x, max_width)
        x1 = max(x2 - region_size, 0)
        x2 = x1 + region_size
        additive_y = random.randint(annotation.bb.h, region_size)
        y2 = min(annotation.bb.y1 + additive_y, max_height)
        y1 = max(y2 - region_size, 0)
        y2 = y1 + region_size
        cut_bb_loc = BBox(x1, y1, x2, y2)
        return cut_bb_loc

    @staticmethod
    def blank_img_mask(size):
        cut_img = np.zeros((size, size, 3), dtype=np.uint8)
        cut_img.fill(255)
        mask = np.zeros((size, size), dtype=np.float32)
        return cut_img, mask

    def get_mask(self, img, annotations, pad=0.):
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
        for a in annotations:
            bb = a.bb.with_pad(pad)
            g_sz = max(mask.shape[0], mask.shape[1])
            g = self.make_gaussian(g_sz, bb.center, fwhm=a.bb.h)
            try:
                mask += g[:mask.shape[0], :mask.shape[1]]
            except Exception as ex:
                raise ex

        mask = np.clip(mask, 0., 1.)
        return mask

    @staticmethod
    def make_gaussian(size, center, fwhm=3):
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = center
        return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    image_size = 512*20
    # transforms = A.Compose([
    #     A.Resize(image_size, image_size),
    #     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    #     ToTensorV2(),
    # ])
    slides = SlidesListLocalTrain()
    dataset = RegionMaskDataset([slides[i] for i in range(len(slides))],
                                image_size, pad=0.3, dbg=True, region_size_annotations=50, transforms=None)

    #fig, axes = plt.subplots(4, 2)
    idx = np.arange(0, len(slides))
    np.random.shuffle(idx)
    for i in tqdm(idx):
        slide = slides[i]
        img, mask, cut_bb = dataset[i]
        if cut_bb:
            for a in slide.annotations:
                if cut_bb.contains(a.bb):
                    bb = a.bb.with_page(cut_bb.page).with_offset(-cut_bb.x1, -cut_bb.y1)
                    cv2.rectangle(img, bb.pt1, bb.pt2, clr_by_cls(a.label), 10)

        mask = (mask * 255).astype(np.uint8)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        #img = cv2.resize(img, (image_size, image_size))
        mask = cv2.resize(mask, (image_size, image_size))
        plt.imshow(img)
        plt.show()
        # axes[(i % 8) % 4, (i % 8) // 4].imshow(cv2.addWeighted(img, 0.6, mask, 0.4, 0))
        # axes[(i % 8) % 4, (i % 8) // 4].set_title(f"{slide.name} {i}")
        # if i % 8 == 0 and i:
        #     plt.show()
        #     fig, axes = plt.subplots(4, 2)
