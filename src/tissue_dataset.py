import numpy as np
import torch
from torch.utils.data import Dataset

from src.slide import SlidesListLocalTrain
from src.tools import best_page_for_sz


class Tissue3ClsDataset(Dataset):

    def __init__(self, slides, image_size, pad, use_each_slide_times=1, transform=None):
        self._slides = slides
        self._transform = transform
        self._image_size = image_size
        self._pad = pad
        self._use_each_slide_times = use_each_slide_times

    def __getitem__(self, index: int):
        index = index % len(self._slides)
        slide = self._slides[index]

        found_tissues = slide.get_tissues(max_locs=10, pad=self._pad)

        if slide.label == 3:
            tissues = list(filter(lambda t: any([a.label == 3 for a in t.annotations]), found_tissues))
            if not tissues:
                tissues = found_tissues
        else:
            tissues = found_tissues

        if tissues:
            selected_tissue = np.random.choice(tissues)

            tbb_p0 = selected_tissue.bb.with_page(0)
            tissue_sz_p0 = min(tbb_p0.h, tbb_p0.w)
            page_n = best_page_for_sz(tissue_sz_p0, self._image_size)

            rnd_offset_x = np.random.randint(0, int(0.1 * tbb_p0.w))
            rnd_offset_y = np.random.randint(0, int(0.1 * tbb_p0.h))
            bb = tbb_p0.with_offset(rnd_offset_x, rnd_offset_y).with_page(page_n)
            img, _ = slide.get_page_region(crop_bb=bb, pad=True)

            label = [0, 0]
            label[int(slide.label == 3)] = 1
        else:
            img = np.zeros((256, 256, 3), dtype=np.uint8)
            label = [1, 0]

        if self._transform is not None:
            img = self._transform(img)

        return img, torch.tensor(label).float()

    def __len__(self) -> int:
        return len(self._slides)*self._use_each_slide_times


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd
    pad = 0.1
    image_size = 256
    use_each_slide_times = 10
    slides = SlidesListLocalTrain()

    dataset = Tissue3ClsDataset(slides, image_size, pad, use_each_slide_times)
    fig, axes = plt.subplots(2, 4)
    for i in range(len(dataset)):
        img, label = dataset[i // 4]

        # ax_idx_x = (i % 8) // 4
        # ax_idx_y = (i % 8) % 4
        # axes[ax_idx_x, ax_idx_y].imshow(img)
        # axes[ax_idx_x, ax_idx_y].set_title(f"{label}")
        # if i % 8 == 0 and i:
        #     plt.show()
        #     fig, axes = plt.subplots(2, 4)
