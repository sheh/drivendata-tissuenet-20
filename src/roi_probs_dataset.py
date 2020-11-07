import random
from typing import List, Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from src.slide import Slide, SlidesList, SlidesListLocalTrain
from src.tools import BBox, best_page_for_sz

import pandas as pd


class RoiProbsDataset(Dataset):

    def __init__(self, df, slides, image_size, transform=None):
        self._transform = transform
        self._df = df
        self._slides = slides
        self._image_size = image_size

    def __getitem__(self, index: int):
        row = self._df.loc[index]
        c = self._df.loc[index, "x1":"y2"]
        bb = BBox(*c, page=0)
        probs = np.asarray(self._df.loc[index, "prob0":"prob3"])
        slide = self._slides.get_by_name(row.filename)
        page_n = best_page_for_sz(bb.w, self._image_size)
        bb = bb.with_page(page_n)
        roi_image, _ = slide.get_page_region(crop_bb=bb)

        if self._transform is not None:
            roi_image = self._transform(roi_image)

        cls = probs.argmax()
        labels = np.zeros((4,), dtype=np.float)
        labels[cls] = 1
        return roi_image, labels

    def __len__(self) -> int:
        return len(self._df)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import cv2

    def _filter_df(df, prob):
        df["pred"] = df.iloc[:, 5:9].values.argmax(axis=1)
        df["pred_prob"] = df.iloc[:, 5:9].values[np.arange(len(df["pred"].values)), df["pred"].values]
        ret = df[(df["label"] == df["pred"]) & (df["pred_prob"] >= prob)]
        return ret.reset_index()


    image_size = 256

    train_df = pd.read_csv("./../train_roi_probs.csv")
    train_df = _filter_df(train_df, 0.8)

    slides = SlidesListLocalTrain()
    train_dataset = RoiProbsDataset(train_df, slides, image_size, transform=None)

    fig, axes = plt.subplots(4, 2)
    for i in range(len(train_dataset)):
        # row = train_df.loc[i]
        # slide = slides.get_by_name(row.filename)
        # bb = BBox(row.x1, row.y1, row.x2, row.y2, page=0)
        # bb = bb.with_page(5)
        # slide_img, _ = slide.get_page_region(page_n=5)
        # cv2.rectangle(slide_img, bb.pt1, bb.pt2, (255, 0, 0), 5)
        # fig2, ax = plt.subplots(1, 1)
        # ax.imshow(slide_img)
        # plt.show()
        img, label = train_dataset[i]

        #train_df.

        # if cut_bb:
        #     for a in slide.annotations:
        #         if cut_bb.contains(a.bb):
        #             bb = a.bb.with_page(cut_bb.page).with_offset(-cut_bb.x1, -cut_bb.y1)
        #             cv2.rectangle(img, bb.pt1, bb.pt2, (255, 0, 0), 2)

        axes[(i % 8) % 4, (i % 8) // 4].imshow(img)
        axes[(i % 8) % 4, (i % 8) // 4].set_title(label)
        if i % 8 == 0 and i:
            plt.show()
            fig, axes = plt.subplots(4, 2)
