import csv
from collections import defaultdict
import shutil

import sys

import math
from argparse import ArgumentParser

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

from src.efficientnet import EfficientNet
from src.epth_classifier import EpthClassifier
from src.slide import SlidesListLocalTrain, UnlabeledSlidesListLocalTrain
from src.tools import create_bb_grid, best_page_for_sz, clr_by_cls, Annotation, BBox
from train_clf_epithelium import get_transforms


def load_model(device, model_name):
    model_path = "../logdir-clf-epithelium/checkpoints/best.pth"
    model = EfficientNet.from_name(model_name=model_name, num_classes=2)
    state_dict = torch.load(model_path)
    ret = model.load_state_dict(state_dict["model_state_dict"])
    print(state_dict['valid_metrics'])
    assert not set(ret.missing_keys), 'Missing keys when loading pretrained weights: {}'.format(ret.missing_keys)
    model.eval()
    model.to(device)
    return model


class ManualAnnotationFile:

    def __init__(self, path):
        self._path = path
        self._annotations = defaultdict(list)
        for r in csv.DictReader(open(self._path, "r")):
            ri = {f: int(r[f]) for f in ("x1", "y1", "x2", "y2", "annotation_class")}
            a = Annotation(ri["annotation_class"], BBox(ri["x1"], ri["y1"], ri["x2"], ri["y2"], page=0))
            self._annotations[r["filename"]].append(a)
        back_file = self._path + ".back"
        shutil.rmtree(back_file, ignore_errors=True)
        shutil.copyfile(self._path, back_file)

    def get(self, filename):
        return self._annotations[filename]

    def set(self, filename, annotations):
        self._annotations[filename] = annotations
        self._sync()

    def _sync(self):
        with open(self._path, "w") as fd:
            writer = csv.DictWriter(fd, fieldnames=["filename", "x1", "x2", "y1", "y2", "annotation_class"])
            writer.writeheader()
            for filename, annotations in self._annotations.items():
                for a in annotations:
                    bb = a.bb.with_page(0)
                    writer.writerow({
                        "filename": filename,
                        "x1": bb.x1, "y1": bb.y1, "x2": bb.x2, "y2": bb.y2,
                        "annotation_class": a.label
                    })

    def get_labels_n(self, label):
        return sum([1 for annotations in self._annotations.values() for a in annotations if a.label == label])


MAX_TISSUES = 4
WORKING_PAGE = 3
PAD = 0.5
BATCH_SIZE = 128
IMAGE_SIZE = 128
MODEL_NAME = "efficientnet-b0"


def main(args):
    _, transforms = get_transforms(IMAGE_SIZE)

    cv2.namedWindow("tissue", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("tissue", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.namedWindow("tissue", cv2.WINDOW_GUI_NORMAL)

    exit_ = False

    if args.unlabeled:
        slides = UnlabeledSlidesListLocalTrain()
        annotation_file = ManualAnnotationFile("../manual_annotations_unlabeled.csv")
    else:
        slides = SlidesListLocalTrain()
        annotation_file = ManualAnnotationFile("../manual_annotations.csv")
    total_slides = len(slides)

    filenames_test = set()
    if args.check_test:
        filenames_test = set(pd.read_csv("inference-data/test_metadata.csv").filename.values.tolist())

    if args.use_model:
        epth_clf = EpthClassifier("../logdir-clf-epithelium/checkpoints/best.pth", 128)

    for i, slide in enumerate(slides):
        if args.check_test and slide.name not in filenames_test:
            continue
        if args.skip_annotated and annotation_file.get(slide.name):
            continue
        tissues = slide.get_tissues(max_locs=MAX_TISSUES, pad=0.1)
        tissues = sorted(tissues, key=lambda x: x.has_annotations, reverse=True)

        slide_placed_annotation = annotation_file.get(slide.name)
        for t in tissues[:MAX_TISSUES]:
            tbb = t.bb.with_page(WORKING_PAGE)
            # get placed annotations
            tissue_placed_annotations = [a for a in slide_placed_annotation if tbb.contains(a.bb)]
            for a in tissue_placed_annotations:
                slide_placed_annotation.remove(a)
            placed_annotations = [a.with_page(tbb.page).with_offset(-tbb.x1, -tbb.y1) for a in
                                  tissue_placed_annotations]

            tissue_img, _ = slide.get_page_region(crop_bb=tbb)
            tissue_img = cv2.cvtColor(tissue_img, cv2.COLOR_BGR2RGB)
            cv2.putText(tissue_img, f"{i+1}/{total_slides}", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 6, 0, 6)
            cv2.putText(tissue_img, str(annotation_file.get_labels_n(0)), (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 6, 0, 6)
            cv2.putText(tissue_img, str(annotation_file.get_labels_n(1)), (100, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 0, 0), 6)
            cv2.putText(tissue_img, str(slide.label), (100, 600),
                        cv2.FONT_HERSHEY_SIMPLEX, 6, clr_by_cls(slide.label), 6)

            if args.use_model:
                roi_bbs, probs = epth_clf.predict_epth(slide, t)

                for p, bb in zip(probs, roi_bbs):
                    clr = cm.viridis(p)
                    clr = (int(clr[2]*255), int(clr[1]*255), int(clr[0]*255), int(clr[3]*255))
                    bb = bb.with_page(tbb.page).with_offset(-tbb.x1, -tbb.y1).with_pad(-0.05)
                    cv2.rectangle(tissue_img, bb.pt1, bb.pt2, clr, 10)

            for annotation in t.annotations:
                abb = annotation.bb.with_page(tbb.page).with_offset(-tbb.x1, -tbb.y1)
                clr = clr_by_cls(annotation.label)
                clr = (clr[2], clr[1], clr[0])
                cv2.rectangle(tissue_img, abb.pt1, abb.pt2, clr, 10)

            current_annotation_class = 0

            draw = tissue_img.copy()
            for a in placed_annotations:
                cv2.rectangle(draw, a.bb.pt1, a.bb.pt2, a.label * 255, 12)

            def on_click(event, x, y, flags, param):
                nonlocal placed_annotations
                if event not in [cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN]:
                    return None
                draw = tissue_img.copy()
                x1 = x - slide.annotation_size(tbb.page) // 2
                x2 = x + slide.annotation_size(tbb.page) // 2
                y1 = y - slide.annotation_size(tbb.page) // 2
                y2 = y + slide.annotation_size(tbb.page) // 2

                if not any([a.bb.contains_point(x, y) for a in placed_annotations]):
                    new_annotation = Annotation(current_annotation_class, BBox(x1, y1, x2, y2, tbb.page))
                    placed_annotations.append(new_annotation)
                else:
                    placed_annotations = list(filter(lambda a: not a.bb.contains_point(x, y), placed_annotations))

                for a in placed_annotations:
                    cv2.rectangle(draw, a.bb.pt1, a.bb.pt2, a.label*255, 12)

                cv2.imshow("tissue", draw)

            cv2.setMouseCallback("tissue", on_click)
            cv2.imshow("tissue", draw)
            while True:
                k = cv2.waitKey()
                if k == 27:
                    exit_ = True
                    break
                elif k == 32:
                    break
                elif k == ord('q'):
                    current_annotation_class = int(not current_annotation_class)

            placed_annotations = [a.with_offset(tbb.x1, tbb.y1) for a in placed_annotations]
            slide_placed_annotation.extend(placed_annotations)
            annotation_file.set(slide.name, slide_placed_annotation)
            if exit_:
                break
        if exit_:
            break


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--skip-annotated", action="store_true")
    parser.add_argument("--check-test", action="store_true")
    parser.add_argument("--use-model", action="store_true")
    parser.add_argument("--unlabeled", action="store_true")
    main(parser.parse_args())
