import logging

import math

import torch

from torchvision.transforms import Compose, Resize, Normalize, ToTensor, ToPILImage
import numpy as np

from src.efficientnet import EfficientNet
from src.tools import best_page_for_sz, create_bb_grid


class EpthClassifier:

    def __init__(self, model_path, batch_size):
        state_dict = torch.load(model_path)
        self._pad = state_dict["checkpoint_data"]["pad"]
        self._image_size = state_dict["checkpoint_data"]["image_size"]
        self._batch_size = batch_size
        self._device = torch.device("cuda")
        self._transforms = Compose([
            ToPILImage(),
            Resize((self._image_size, self._image_size)),
            ToTensor(),
            Normalize(mean=(0.73108001, 0.54549926, 0.67233236), std=(0.11515842, 0.13832434, 0.11472176)),
        ])
        self.model = EfficientNet.from_name(model_name=state_dict["checkpoint_data"]["network_type"], num_classes=2)
        ret = self.model.load_state_dict(state_dict["model_state_dict"])
        print(f"Epth clf val metrics: {state_dict['valid_metrics']}")
        assert not set(ret.missing_keys), 'Missing keys when loading pretrained weights: {}'.format(ret.missing_keys)
        self.model.eval()
        self.model.to(self._device)

    def predict_epth(self, slide, tissue):
        grid_sz_p0 = slide.annotation_size(0) + 2 * self._pad * slide.annotation_size(0)
        region_page_n = best_page_for_sz(grid_sz_p0, self._image_size)

        tbb = tissue.bb.with_page(region_page_n)
        grid_sz = slide.annotation_size(region_page_n) + 2 * self._pad * slide.annotation_size(region_page_n)
        region_images = []
        region_bb = []
        for j, gbb in enumerate(create_bb_grid(tbb.w, tbb.h, grid_sz)):
            gbb.set_page(region_page_n)
            region_bb.append(gbb.with_offset(tbb.x1, tbb.y1).with_page(0))
            gbb = gbb.with_offset(tbb.x1, tbb.y1).with_page(region_page_n)
            region_img, _ = slide.get_page_region(crop_bb=gbb, pad=True)
            region_images.append(self._transforms(region_img))

        batch_n = math.ceil(len(region_images) / self._batch_size)
        batches = [region_images[self._batch_size * j:self._batch_size * (j + 1)] for j in range(batch_n)]

        probs = []
        for batch in batches:
            with torch.no_grad():
                logits = self.model(torch.stack(batch, dim=0).to(self._device))
                batch_probs = logits.softmax(1).detach().cpu().numpy()
            probs.extend(batch_probs[:, 1].tolist())
        return np.array(region_bb), np.array(probs)
