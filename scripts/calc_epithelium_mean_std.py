from torch.utils.data import ConcatDataset
from torchvision.transforms import Compose, ToPILImage, Resize
from tqdm import tqdm

from src.annotation_dataset import EpitheliumDataset
from src.slide import SlidesListLocalTrain, UnlabeledSlidesListLocalTrain

import pandas as pd
import numpy as np


def get_dateset(slides, manual_annotation_path, pad, image_size):
    annotations = []
    for s in slides:
        for a in s.annotations or []:
            annotations.append(dict(
                filename=s.name,
                x1=a.bb.x1,
                y1=a.bb.y1,
                x2=a.bb.x2,
                y2=a.bb.y2,
                annotation_class=1,
            ))
    df = pd.read_csv(manual_annotation_path)
    df = df.append(annotations, ignore_index=True)
    return EpitheliumDataset(slides, df, image_size, pad)


if __name__ == '__main__':
    pad = 0.25
    image_size = 224

    ds1 = get_dateset(SlidesListLocalTrain(), "../manual_annotations.csv", pad, image_size)
    ds2 = get_dateset(UnlabeledSlidesListLocalTrain(), "../manual_annotations_unlabeled.csv", pad, image_size)
    dataset = ConcatDataset([ds1, ds2])

    mean = np.zeros((3, ))
    std = np.zeros((3,))
    for i in tqdm(range(len(dataset))):
        img, _ = dataset[i]
        img = (img / 255).reshape(-1, 3)
        mean += img.mean(axis=0)
        std += img.std(axis=0)
    mean /= len(dataset)
    std /= len(dataset)
    print(f"mean: {mean}")
    print(f"std: {std}")
