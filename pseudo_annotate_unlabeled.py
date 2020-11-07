from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from src.epth_classifier import EpthClassifier
from src.region_classifier import RegionClassifier
from src.slide import UnlabeledSlidesListLocalTrain, SlidesListLocalTrain
from src.tools import BBox, best_page_for_sz
import pandas as pd


def main(args):
    epth_clf = EpthClassifier(args.model_path, 256)
    slides = UnlabeledSlidesListLocalTrain()

    high_scored_roi = []
    try:
        for slide in tqdm(slides):
            for tissue_n, tissue in enumerate(slide.get_tissues(max_locs=12, pad=0.1)):
                roi_bbs, probs = epth_clf.predict_epth(slide, tissue)
                roi_bbs = roi_bbs[probs > args.min_prob]
                probs = probs[probs > args.min_prob]
                for i in range(len(roi_bbs)):
                    high_scored_roi.append({
                        "filename": slide.name,
                        "x1": roi_bbs[i].x1,
                        "y1": roi_bbs[i].y1,
                        "x2": roi_bbs[i].x2,
                        "y2": roi_bbs[i].y2,
                        "prob": probs[i],
                    })
    except KeyboardInterrupt:
        pass
    pd.DataFrame(high_scored_roi).to_csv("high_score_roi_unlabeled.csv", index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model-path", default="./logdir-clf-epithelium/checkpoints/best_full.pth")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--min-prob", default=0.7, type=float)

    main(parser.parse_args())
