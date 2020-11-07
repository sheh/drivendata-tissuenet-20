import os
from functools import lru_cache
from pathlib import Path
from time import time
from typing import Optional, List, Tuple

import cv2
import pandas as pd
import numpy as np
import pyvips

from src.tissue import Tissue
from src.tools import BBox, Annotation, filter_annotations


class SlidesList:

    def __init__(self, tif_path, df_metadata, df_annotations=None, df_label=None):
        self.tif_path = Path(tif_path)
        self.df_meta = df_metadata
        self.df_annotations = df_annotations
        self.df_label = df_label
        self._slide_cache = {}

    @classmethod
    def from_csv(cls, tif_path, metadata_csv, annotations_csv=None, label_csv=None):
        df_meta = pd.read_csv(metadata_csv)
        df_annotations = pd.read_csv(annotations_csv) if annotations_csv else None
        df_label = pd.read_csv(label_csv) if label_csv else None
        return cls(tif_path, df_meta, df_annotations, df_label)

    def split_train_test(self, filenames_test):
        df_meta_train = self.df_meta[~self.df_meta.filename.isin(filenames_test)].reset_index(drop=True)
        df_annotations_train = self.df_annotations[~self.df_annotations.filename.isin(filenames_test)].reset_index(drop=True) \
            if self.df_annotations is not None else None
        df_label_train = self.df_label[~self.df_label.filename.isin(filenames_test)].reset_index(drop=True) if self.df_label is not None else None
        train = SlidesList(self.tif_path, df_meta_train, df_annotations_train, df_label_train)

        df_meta_test = self.df_meta[self.df_meta.filename.isin(filenames_test)].reset_index(drop=True)
        df_annotations_test = self.df_annotations[self.df_annotations.filename.isin(filenames_test)].reset_index(drop=True) \
            if self.df_annotations is not None else None
        df_label_test = self.df_label[self.df_label.filename.isin(filenames_test)].reset_index(drop=True) if self.df_label is not None else None
        test = SlidesList(self.tif_path, df_meta_test, df_annotations_test, df_label_test)

        return train, test

    def __getitem__(self, item):
        row = self.df_meta.loc[item, :]
        return self.get_by_name(row.filename)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_by_name(self, slide_name):
        if slide_name not in self._slide_cache:
            row = self.df_meta[self.df_meta.filename == slide_name].iloc[0]
            if self.df_annotations is not None:
                annotations = []
                for _, row_annot in self.df_annotations[self.df_annotations.filename == slide_name].iterrows():
                    bb = self.parse_geometry(row_annot.geometry, row.height)
                    annotations.append(Annotation(row_annot.annotation_class, BBox(*bb, page=0)))
            else:
                annotations = None

            if self.df_label is not None:
                on_hot = self.df_label[self.df_label.filename == slide_name].iloc[:, 1:].to_numpy()
                label = np.argmax(on_hot, axis=1)[0]
            else:
                label = None
            s = Slide(row.filename, self.tif_path / row.filename,
                      row.width, row.height, row.resolution,
                      annotations, label
                      )
            self._slide_cache.update({s.name: s})
        return self._slide_cache[slide_name]

    def __len__(self):
        return len(self.df_meta)

    @staticmethod
    def parse_geometry(geo, img_height):
        coord = []
        for point in geo.split("((", 1)[1][:-3].split(","):
            x, y = point.strip().split(' ')
            x = float(x)
            y = float(y)
            coord.append((x, y))
        coord = np.array(coord)
        coord[:, 1] = img_height - coord[:, 1]
        coord = coord.astype(dtype=np.int32)
        coord = np.unique(coord, axis=0)
        coord = coord[coord[:, 1].argsort()]
        return coord[[0, -1], :].ravel()

    def dump_dfs_for_stage(self, output_dir, filenames, prefix):
        output_dir = Path(output_dir)
        self.df_annotations[self.df_annotations["filename"].isin(filenames)]\
            .to_csv(output_dir / f"{prefix}_annotations.csv", index=False)
        self.df_meta[self.df_meta["filename"]
            .isin(filenames)].to_csv(output_dir / f"{prefix}_metadata.csv", index=False)
        self.df_label[self.df_label["filename"].isin(filenames)]\
            .to_csv(output_dir / f"{prefix}_labels.csv", index=False)

        submission_format = self.df_label[self.df_label["filename"].isin(filenames)].copy()
        submission_format.iloc[:, 1:] = 0
        if prefix == "test":
            submission_format.to_csv(output_dir / f"submission_format.csv", index=False)
        else:
            submission_format.to_csv(output_dir / f"{prefix}_submission_format.csv", index=False)


class SlidesListLocalTrain(SlidesList):

    def __init__(self):
        root = Path(os.getenv("TISSIENET_CSV"))
        super().__init__(
            Path(os.getenv("TISSIENET_TIF")),
            pd.read_csv(root / "train_metadata.csv"),
            pd.read_csv(root / "train_annotations.csv"),
            pd.read_csv(root / "train_labels.csv"),
        )


class UnlabeledSlidesListLocalTrain(SlidesList):

    def __init__(self):
        root = Path(os.getenv("TISSIENET_CSV"))
        tif_path = Path(os.getenv("TISSIENET_TIF_UNLABELED"))
        tif_files = list([p.name for p in tif_path.glob("*.tif")])
        df_meta = pd.read_csv(root / "unlabeled_metadata.csv")
        df_meta = df_meta[df_meta.filename.isin(tif_files)].reset_index()
        super().__init__(
            tif_path,
            df_meta,
        )


class Slide:

    SEARCH_TISSUE_PATH = 8
    SEARCH_TISSUE_SCALE = 2**SEARCH_TISSUE_PATH
    REGION_SIZE_MK = 300

    def __init__(self, name, tif_path, width, height, resolution, annotations=None, label=None):
        self.name = name
        self.width = width
        self.height = height
        self.resolution = resolution
        self.annotations = annotations
        self.label = label
        self.tif_path = tif_path

    def get_page_region(self, page_n=None, crop_bb: Optional[BBox] = None, pad=False):
        assert page_n is not None or crop_bb.page is not None
        page_n = page_n or crop_bb.page
        slide, page_n = self._open_slide(page_n)

        if crop_bb is None:
            crop_bb = BBox(0, 0, slide.width, slide.height)

        pad_x = pad_y = 0
        if pad:
            pad_x = max(crop_bb.x2 - slide.width, 0)
            pad_y = max(crop_bb.y2 - slide.height, 0)
            crop_bb = crop_bb.with_fit_image(slide.width, slide.height)

        region = pyvips.Region.new(slide).fetch(*crop_bb.xywh)

        img = np.ndarray(
            buffer=region,
            dtype=np.uint8,
            shape=(crop_bb.h, crop_bb.w, 3)
        )
        if pad_x > 0 or pad_y > 0:
            img = np.pad(img, [(0, pad_x), (0, pad_y), (0, 0)], mode='constant', constant_values=255)

        del region
        return img, page_n

    def _open_slide(self, page_n):
        assert page_n > 0, "At least page=0 can be open"
        try:
            slide = pyvips.Image.new_from_file(str(self.tif_path), page=page_n)
        except pyvips.Error as ex:
            return self._open_slide(page_n-1)
        else:
            return slide, page_n

    def get_tissues(self, max_locs=6, pad=0, white_column_lvl=250, white_column_min_width=0.05) -> List[Tissue]:
        # TODO: use search on different scales in case unsuccess
        img, page_n = self.get_page_region(self.SEARCH_TISSUE_PATH)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # remove white pads
        gray_dilate = cv2.dilate(gray, np.ones((15, 15), np.uint8), iterations=1)
        y1_w, x1_w = np.argwhere(gray_dilate != 255).min(axis=0)
        y2_w, x2_w = np.argwhere(gray_dilate != 255).max(axis=0)
        gray = gray[y1_w:y2_w, x1_w:x2_w]

        # remove white columns
        gray, offsets = self.remove_vertical_column(gray, white_column_lvl, white_column_min_width)

        # remove black lines
        thresh_black = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)[1]
        gray[thresh_black == 255] = np.bincount(gray.ravel()).argmax()

        # find contours and rectangles
        contours = self._find_contours(gray)
        possible_tissue_bbs = self._contours2rects(contours, pad, gray.shape[1], gray.shape[0])

        # the same for each found rectangle from the previous step
        step_n = 1
        ret = []
        if step_n == 2:
            for bb in possible_tissue_bbs:
                biggest_rect = gray[bb.s_img]
                contours_s3 = self._find_contours(biggest_rect)
                sub_bbs = self._contours2rects(contours_s3, pad, bb.w, bb.h)
                for sub_bb in sub_bbs:
                    tbb = sub_bb.with_offset(bb.x1 + x1_w, bb.y1 + y1_w)
                    tbb.set_page(page_n)
                    ret.append(Tissue(tbb, filter_annotations(tbb, self.annotations or [])))
        else:
            for bb in possible_tissue_bbs:
                tbb = bb.with_offset(x1_w + offsets[bb.x1], y1_w)
                tbb.set_page(page_n)
                ret.append(Tissue(tbb, filter_annotations(tbb, self.annotations or [])))
        return ret[:max_locs]

    @staticmethod
    def remove_vertical_column(gray, white_column_lvl, column_min_width):
        curr_seq_start = None
        gray_mean = gray.mean(axis=0)
        cols_to_remove = []
        offsets = np.zeros((gray_mean.shape[0],))
        deleted_columns_n = 0
        for i in range(gray_mean.shape[0]):
            if gray_mean[i] >= white_column_lvl and curr_seq_start is None:
                curr_seq_start = i
            elif gray_mean[i] < white_column_lvl and curr_seq_start is not None:
                if (i - curr_seq_start) >= column_min_width * gray_mean.shape[0]:
                    cols_to_remove.extend(list(range(curr_seq_start, i)))
                    offsets[curr_seq_start - deleted_columns_n:] += i - curr_seq_start
                    deleted_columns_n += i - curr_seq_start
                curr_seq_start = None
        if gray.shape[1] - len(cols_to_remove) > .01 * gray.shape[1]:
            gray = np.delete(gray, cols_to_remove, axis=1)
        else:
            offsets = np.zeros((gray_mean.shape[0],))
        return gray, offsets

    def annotation_size(self, page):
        return int((self.REGION_SIZE_MK / self.resolution) // 2**page)

    def page_width(self, page):
        return self.width // 2 ** page

    def page_height(self, page):
        return self.height // 2 ** page

    @staticmethod
    def _contours2rects(contours, pad, width, height, ratio_limit=7.0, min_area=0.1):
        # calc bounding boxes
        rects = sorted(map(cv2.boundingRect, contours), key=lambda x: x[2] * x[3], reverse=True)
        # filter by ration
        rects = list(filter(lambda x: (1 / ratio_limit) < x[2] / x[3] < (ratio_limit / 1), rects))
        # filter by area
        rects = list(filter(lambda x: x[2] * x[3] / (rects[0][2] * rects[0][3]) > min_area, rects))
        # add padding and convert to x1, y1, x2, y2
        return list(map(lambda x: BBox(
            max(int(x[0] - pad * x[2]), 0),
            max(int(x[1] - pad * x[3]), 0),
            min(int(x[0] + pad * x[2] + x[2]), width),
            min(int(x[1] + pad * x[3] + x[3]), height),
        ), rects))

    @staticmethod
    def _find_contours(img_gray):
        blur = cv2.GaussianBlur(img_gray, (25, 25), 0)
        blur = (255 * (blur / blur.max())).astype(np.uint8)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        noise_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, noise_kernel, iterations=2)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, close_kernel, iterations=2)

        contours, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
