import random
from typing import List, Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from src.slide import Slide, SlidesList, SlidesListLocalTrain
from src.tools import best_page_for_sz, BBox, Annotation


class AnnotationDataset(Dataset):

    def __init__(self, slides, df, image_size, pad, transform=None):
        self._slides = slides
        self._df = df
        self._transform = transform
        self._image_size = image_size
        self._pad = pad

    def __getitem__(self, index: int):
        filename = self._df.iloc[index].at["filename"]
        annotation_class = self._df.iloc[index].at["annotation_class"]
        geometry = self._df.iloc[index].at["geometry"]

        slide = self._slides.get_by_name(filename)

        region_sz = slide.annotation_size(page=0) + 2 * self._pad * slide.annotation_size(page=0)
        page_n = best_page_for_sz(region_sz, self._image_size)

        bb_coors = SlidesList.parse_geometry(geometry, slide.height)
        bb = BBox(*bb_coors, page=0).with_page(page_n).with_pad(self._pad)

        img, _ = slide.get_page_region(crop_bb=bb)

        if self._transform is not None:
            img = self._transform(img)

        label = [0, 0, 0, 0]
        label[annotation_class] = 1
        return img, torch.tensor(label).float()

    def __len__(self) -> int:
        return len(self._df)


class PseudoAnnotationDataset(Dataset):

    def __init__(self, slides, df, image_size, pad, transform=None):
        self._slides = slides
        self._df = df
        self._transform = transform
        self._image_size = image_size
        self._pad = pad

    def __getitem__(self, index: int):
        filename = self._df.iloc[index].at["filename"]
        annotation_class = self._df.iloc[index].at["pseudo_class"]

        slide = self._slides.get_by_name(filename)

        region_sz = slide.annotation_size(page=0) + 2 * self._pad * slide.annotation_size(page=0)
        page_n = best_page_for_sz(region_sz, self._image_size)
        bb = BBox(self._df.iloc[index].at["x1"],
                  self._df.iloc[index].at["y1"],
                  self._df.iloc[index].at["x2"],
                  self._df.iloc[index].at["y2"],
                  page=0).with_page(page_n).with_pad(self._pad)

        img, _ = slide.get_page_region(crop_bb=bb, pad=True)

        if self._transform is not None:
            img = self._transform(img)

        label = [0, 0, 0, 0]
        label[annotation_class] = 1
        return img, torch.tensor(label).float()

    def __len__(self) -> int:
        return len(self._df)


class AnnotationSlideDataset(Dataset):
    TOTAL_CLASSES = 4

    def __init__(self, slides: SlidesList, image_size, pad, use_each_slide_times=1,
                 target_class=None, transform=None):
        self._slides = slides
        self._transform = transform
        self._target_class = target_class
        self._use_each_slide_times = use_each_slide_times
        self._image_size = image_size
        self._pad = pad

    def __getitem__(self, index: int):
        index = index % len(self._slides)
        slide = self._slides[index]

        annotations = slide.annotations if self._target_class is None or slide.label != self._target_class else \
            list(filter(lambda x: x.label == self._target_class, slide.annotations))

        annotation_idx = random.randint(0, len(annotations)-1)
        annotation = annotations[annotation_idx]

        img, _ = self._region_reader(slide, annotation.bb)

        if self._transform is not None:
            img = self._transform(img)

        if self._target_class is None:
            label = np.zeros((self.TOTAL_CLASSES, ))
            label[annotation.label] = 1
        else:
            label = np.array([0, 1]) if annotation.label == self._target_class else np.array([1, 0])
        label = label.astype(np.float)
        return img, torch.from_numpy(label).float()

    def __len__(self) -> int:
        return len(self._slides)*self._use_each_slide_times

    def _region_reader(self, s, rbb):
        page_n = best_page_for_sz(s.annotation_size(page=0), self._image_size)
        scale = 2**page_n
        rbb = rbb.with_page(page_n)\
            .with_pad(self._pad)\
            .with_fit_image(s.width // scale, s.height // scale)
        return s.get_page_region(crop_bb=rbb)


class Annotation3ClassDataset(Dataset):

    def __init__(self, slides, df, image_size, pad, transform=None):
        self._slides = slides
        self._df = df
        self._transform = transform
        self._image_size = image_size
        self._pad = pad

    def __getitem__(self, index: int):
        filename = self._df.iloc[index].at["filename"]
        annotation_class = self._df.iloc[index].at["annotation_class"]
        geometry = self._df.iloc[index].at["geometry"]

        slide = self._slides.get_by_name(filename)

        region_sz = slide.annotation_size(page=0) + 2 * self._pad * slide.annotation_size(page=0)
        page_n = best_page_for_sz(region_sz, self._image_size)

        bb_coors = SlidesList.parse_geometry(geometry, slide.height)
        annotation = Annotation(annotation_class, BBox(*bb_coors, page=0))
        annotation.bb = annotation.bb.with_page(page_n)

        img, _ = slide.get_page_region(crop_bb=annotation.bb)

        if self._transform is not None:
            img = self._transform(img)

        label = [0, 0]
        label[annotation.label != 3] = 1
        return img, torch.tensor(label).float()

    def __len__(self) -> int:
        return len(self._df)


class EpitheliumDataset(Dataset):

    def __init__(self, slides, df, image_size, pad, transform=None):
        self._slides = slides
        self._df = df
        self._transform = transform
        self._image_size = image_size
        self._pad = pad

    def __getitem__(self, index: int):
        filename = self._df.iloc[index].at["filename"]
        annotation_class = self._df.iloc[index].at["annotation_class"]
        bb = BBox(self._df.iloc[index].at["x1"],
                  self._df.iloc[index].at["y1"],
                  self._df.iloc[index].at["x2"],
                  self._df.iloc[index].at["y2"],
                  page=0)

        slide = self._slides.get_by_name(filename)

        region_sz = slide.annotation_size(page=0) + 2 * self._pad * slide.annotation_size(page=0)
        page_n = best_page_for_sz(region_sz, self._image_size)

        bb = bb.with_page(self._pad).with_page(page_n)
        img, _ = slide.get_page_region(crop_bb=bb)

        if self._transform is not None:
            img = self._transform(img)

        label = [0, 0]
        label[annotation_class] = 1
        return img, torch.tensor(label).float()

    def __len__(self) -> int:
        return len(self._df)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd
    pad = 0
    image_size = 256
    slides = SlidesListLocalTrain()

    annotations = []
    for s in slides:
        for a in s.annotations:
            annotations.append(dict(
                filename=s.name,
                x1=a.bb.x1,
                y1=a.bb.y1,
                x2=a.bb.x2,
                y2=a.bb.y2,
                annotation_class=a.label,
            ))

    df = pd.read_csv("../manual_annotations.csv")
    df = df.append(annotations, ignore_index=True)

    dataset = EpitheliumDataset(slides, df, image_size, pad)

    fig, axes = plt.subplots(2, 4)
    for i in range(len(dataset)):
        img, labels = dataset[i]

        ax_idx_x = (i % 8) // 4
        ax_idx_y = (i % 8) % 4
        axes[ax_idx_x, ax_idx_y].imshow(img)
        axes[ax_idx_x, ax_idx_y].set_title(labels)
        if i % 8 == 0 and i:
            plt.show()
            fig, axes = plt.subplots(2, 4)
