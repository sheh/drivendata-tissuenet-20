#!/usr/bin/env python
import logging
import logging.handlers
import pickle
from argparse import ArgumentParser
from pathlib import Path

from src.epth_classifier import EpthClassifier
from src.region_classifier import RegionClassifier
from src.rf_clf import train_random_forest, predict_random_forest

from src.roi_segmentation import ROISegmentation
from src.slide import SlidesListLocalTrain, SlidesList
from src.tools import *

import cv2


def setup_logging(file_path):
    formatter = logging.Formatter("%(asctime)s %(message)s")

    logger = logging.getLogger()
    if file_path:
        handler = logging.FileHandler(file_path, mode='w')
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.setLevel(logging.INFO)
    return logger


def predict(args, filenames):
    logger = setup_logging(args.log_path)
    logger.info("Start main.py")

    max_roi_per_tissue = 32

    epth_clf = EpthClassifier(args.epth_model_path, args.epth_batch_size)

    clf = RegionClassifier(model_path=args.clf_model_path)

    errors = []

    roi_predicts = []

    all_slides = SlidesList.from_csv(args.tif_path, args.metadata_path,
                                     annotations_csv=args.annotations_path, label_csv=args.labels_path)
    for i, filename in enumerate(filenames):
        slide = all_slides.get_by_name(filename)
        slide_predicts = []
        slide_predicts_probs = []
        annotation_size_p0 = slide.annotation_size(0)
        clf_page_n = best_page_for_sz(annotation_size_p0, clf.image_size)
        dbg_info = DbgInfo(is_on=args.dbg)
        at_least_one_roi = False
        for tissue_n, tissue in enumerate(slide.get_tissues(max_locs=12, pad=0.1)):
            tbb_p0 = tissue.bb.with_page(0)
            tissue.bb = tissue.bb.with_page(5)
            tissue_img, _ = slide.get_page_region(crop_bb=tissue.bb.with_page(5))

            if args.test_clf:
                roi_bbs = [a.bb for a in tissue.annotations]
            else:
                roi_bbs, roi_probs = epth_clf.predict_epth(slide, tissue)
                roi_bbs = roi_bbs[(-roi_probs).argsort()]
                roi_probs = roi_probs[(-roi_probs).argsort()]
                roi_bbs = roi_bbs[roi_probs > args.epth_min_prob]
            roi_images = []
            roi_bbs_glob = []
            for roi_bb in roi_bbs[:max_roi_per_tissue]:
                r_sz = slide.annotation_size(roi_bb.page)
                # remove padding
                x_center, y_center = roi_bb.center
                roi_bb = BBox(x_center - r_sz // 2, y_center - r_sz // 2,
                              x_center + r_sz // 2, y_center + r_sz // 2, page=0)

                roi_bbs_glob.append(roi_bb)
                roi_bb_glob = roi_bb.with_page(clf_page_n)

                roi_img, _ = slide.get_page_region(crop_bb=roi_bb_glob, pad=True)
                roi_images.append(roi_img)
            if args.dbg:
                dbg_info.add_tissue(tissue, tissue_img, None, None)
                dbg_info.add_others(roi_images)
            if roi_images:
                at_least_one_roi = True
                pred_probs = clf.predict_softmax(roi_images)
                for bb, probs, img in zip(roi_bbs_glob, pred_probs, roi_images):
                    roi_predicts.append({
                        "filename": filename,
                        "x1": bb.x1, "y1": bb.y1, "x2": bb.x2, "y2": bb.y2,
                        "prob0": probs[0], "prob1": probs[1], "prob2": probs[2], "prob3": probs[3],
                        "t_i": tissue_n+1,
                        "t_w": tbb_p0.w // annotation_size_p0, "t_h": tbb_p0.h // annotation_size_p0,
                        "t_r": tissue_img[:, :, 0].mean(),
                        "t_g": tissue_img[:, :, 1].mean(),
                        "t_b": tissue_img[:, :, 2].mean(),
                        "label": slide.label,
                    })
                pred_classes = pred_probs.argmax(axis=1)
                dbg_info.add_predicts(roi_bbs_glob, pred_classes.tolist(), pred_probs.tolist())
                slide_predicts.extend(pred_classes.tolist())
                slide_predicts_probs.extend(pred_probs)

        if not at_least_one_roi:
            roi_predicts.append({
                "filename": filename,
                "x1": 0, "y1": 0, "x2": 0, "y2": 0,
                "prob0": 0, "prob1": 1, "prob2": 0, "prob3": 0,
                "t_i": 0, "t_w": 0, "t_h": 0,
                "label": slide.label,
            })

        # final_predict = np.bincount(slide_predicts).argmax() if len(slide_predicts) else 1

        slide_predicts_probs_mean = np.asarray(slide_predicts_probs).mean(axis=0)
        final_predict = int(slide_predicts_probs_mean.argmax()) if len(slide_predicts) else 1
        final_predict_prob = slide_predicts_probs_mean[final_predict] if len(slide_predicts) else -1

        metric_msg = ""
        if slide.label is not None:
            er = ERROR_TABLE[slide.label, final_predict]
            errors.append(er)
            metric = 1 - sum(errors) / len(errors)
            metric_msg = f"{metric:.3f} gt={slide.label}"
        logger.info(f"Process {filename} ({i + 1}/{len(filenames)}): "
                    f"{metric_msg} pred={final_predict} {final_predict_prob:.2f} {slide_predicts}")
        if slide.label is not None:  # and final_predict != slide.label:
            dbg_info.show(slide.name)

    roi_pred_df = pd.DataFrame(roi_predicts)
    roi_pred_df.to_csv(args.roi_probs_path, index=False)
    if not args.do_not_use_random_forest:
        if args.train_random_forest:
            rf_clf = train_random_forest(roi_pred_df)
            pickle.dump(rf_clf, open(args.model_random_forest, "wb"))
        else:
            rf_clf = pickle.load(open(args.model_random_forest, "rb"))
        submission = predict_random_forest(rf_clf, roi_pred_df)
    else:
        submission = roi_pred_df.groupby("filename")[["prob0", "prob1", "prob2", "prob3"]].mean()
        submission = submission.eq(submission.max(1), axis=0).astype(int)\
            .rename(columns={"prob0": 0, "prob1": 1, "prob2": 2, "prob3": 3})

    # keep submission format sorting
    submission = pd.DataFrame({"filename": filenames}).merge(submission, on="filename")
    submission.to_csv("submission.csv", index=False)

    if args.local_test:
        er = 0
        for _, row in submission.iterrows():
            gt = all_slides.get_by_name(row.filename).label
            pred = row.iloc[1:].values.argmax()
            er += ERROR_TABLE[gt, pred]
        metric = 1 - er / len(submission)
    else:
        metric = None
    return metric


def main(args):
    if args.only_slide:
        slides = [args.only_slide]
    elif args.all_data:
        slides = pd.read_csv(args.metadata_path).filename.values
    else:
        submission_format_df = pd.read_csv(args.submission_format_path, index_col="filename")
        slides = submission_format_df.index
    metric = predict(args, slides)
    if metric:
        logging.info(f"Final metric: {metric:.3f}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--epth-model-path", default="./models/epth.pth")
    parser.add_argument("--epth-batch-size", type=int, default=128)
    parser.add_argument("--epth-min-prob", type=float, default=0.7)

    parser.add_argument("--clf-model-path", default="./models/clf.pth")

    parser.add_argument("--do-not-use-random-forest", action="store_true")
    parser.add_argument("--train-random-forest", action="store_true")
    parser.add_argument("--model-random-forest", default="./models/rf-clf.pkl")

    parser.add_argument("--tif-path", default="./data")
    parser.add_argument("--metadata-path", default="./data/test_metadata.csv")
    parser.add_argument("--submission-format-path", default="./data/submission_format.csv")
    parser.add_argument("--annotations-path", default=None)
    parser.add_argument("--labels-path", default=None)

    parser.add_argument("--roi-probs-path", default="./roi_probs.csv")
    parser.add_argument("--log-path", default="./submission.log")
    parser.add_argument("--local-test", action="store_true")
    parser.add_argument("--dbg", action="store_true")
    parser.add_argument("--test-clf", action="store_true")

    parser.add_argument("--only-slide", default=None)
    parser.add_argument("--all-data", action="store_true")

    main(parser.parse_args())
