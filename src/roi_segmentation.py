import logging

import math
from pathlib import Path

import cv2
import torch

import numpy as np

from src.tools import create_bb_grid
from src.unet import UNet


class ROISegmentation:

    def __init__(self, model_path, region_size_annotations, image_size, batch_size, transforms):
        self.region_size_annotations = region_size_annotations
        self.image_size = image_size
        self.model_path = model_path
        self.batch_size = batch_size
        self.transforms = transforms
        self.device, self.model = self._load_model()

    def predict_roi_mask(self, slide, tissue):
        page_n = self._best_page(slide)

        region_size = int(self.region_size_annotations * slide.annotation_size(page=tissue.bb.page))
        grid_bb = create_bb_grid(tissue.bb.w, tissue.bb.h, region_size)

        regions = []
        regions_bb = []
        for i, bb in enumerate(grid_bb):
            bb.set_page(tissue.bb.page)
            bb_glob = bb.with_offset(tissue.bb.x1, tissue.bb.y1).with_page(page_n)
            region_img, _ = slide.get_page_region(crop_bb=bb_glob, pad=True)
            inp = self.transforms(region_img)
            regions.append(inp)
            regions_bb.append(bb)

        batch_n = math.ceil(len(regions) / self.batch_size)
        batches = [regions[self.batch_size*i:self.batch_size*(i+1)] for i in range(batch_n)]
        regions_bb_batched = [regions_bb[self.batch_size * i:self.batch_size * (i + 1)] for i in range(batch_n)]

        roi_mask = np.zeros((tissue.bb.h+region_size, tissue.bb.w+region_size))

        for batch, regions_bb in zip(batches, regions_bb_batched):
            predicts = self.model(torch.stack(batch).to(self.device))
            predicts = predicts.detach().cpu().numpy()
            for i in range(predicts.shape[0]):
                p = cv2.resize(predicts[i].transpose(1, 2, 0), (regions_bb[i].w, regions_bb[i].h))
                roi_mask[regions_bb[i].s_img] = p
            # pred_mask = cv2.resize(pred_mask, (region_img.shape[1], region_img.shape[0]))
            # region_scale_mask = cv2.resize(pred_mask,
            #                                (int(pred_mask.shape[1] // slide.annotation_size(page_n)),
            #                                 int(pred_mask.shape[0] // slide.annotation_size(page_n))
            #                                 ), interpolation=cv2.INTER_CUBIC)
            # am = np.unravel_index(region_scale_mask.argmax(), region_scale_mask.shape)
        return roi_mask[:tissue.bb.h, :tissue.bb.w]

    def _load_model(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info('device: %s', device)
        model = UNet(in_channels=3, out_channels=1)
        state_dict = torch.load(self.model_path)
        logging.info("ROI segm val metrics: %s", state_dict["valid_metrics"])
        model.load_state_dict(state_dict["model_state_dict"])
        model.eval()
        model = model.to(device)
        return device, model

    def _best_page(self, slide):
        region_size_p0 = int(self.region_size_annotations * slide.annotation_size(page=0))
        page_n = 5
        for page_n in range(8):
            if (region_size_p0 / 2 ** page_n) < self.image_size:
                page_n = page_n - 1
                break
        return page_n
