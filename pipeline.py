import logging
from argparse import ArgumentParser
from pathlib import Path

from time import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from main import predict
from src.slide import SlidesListLocalTrain
from src.tools import ERROR_TABLE
from train_clf import train_clf
from train_clf_epithelium import train_clf_epth
from train_segm import train_segm


SEED = 22


def folds_iter(fold_n, y):
    skf = StratifiedKFold(n_splits=fold_n, random_state=SEED)
    x_fake = np.zeros(len(y))
    for train_index, test_index in skf.split(x_fake, y):
        yield train_index, test_index


class SegmentTrainConfig:
    wandb_run_name = None
    log_dir = None
    seed = SEED

    batch_size = 32
    num_epochs = 20
    workers_n = 12
    lr = 0.02

    image_size = 256
    pad = 0.3
    use_each_slide_times = 20
    region_size_annotations = 10


class ClfEpthTrainCfg:
    wandb_run_name = None
    log_dir = None
    tune = False
    seed = SEED

    network_type = "efficientnet-b0"
    dropout_p = 0.2
    batch_size = 256
    num_epochs = 30
    workers_n = 12
    lr = 0.02

    image_size = 128
    pad = 0


class ClfTrainConfig:
    wandb_run_name = None
    log_dir = None
    tune = False
    seed = SEED

    network_type = "efficientnet-b1"
    dropout_p = 0.3
    batch_size = 48
    num_epochs = 30
    workers_n = 12
    lr = 0.02

    image_size = 256
    pad = 0


class TestConfig:
    epth_model_path = None
    epth_model_type = None
    epth_image_size = None
    epth_batch_size = 128
    epth_pad = None

    clf_model_path = None
    clf_model_type = None
    clf_image_sz = None

    tif_path = "/mnt/disk2/tissuene-dataset"
    metadata_path = "/home/sheh/datasets/TissueNet/train_metadata.csv"
    annotations_path = "/home/sheh/datasets/TissueNet/train_annotations.csv"
    labels_path = "/home/sheh/datasets/TissueNet/train_labels.csv"
    log_path = None
    roi_probs_path = None
    dbg = False
    local_test = True
    test_clf = False
    use_random_forest = True
    random_forest_model = None
    train_random_forest = False


def main(args):
    started_at = int(time())
    all_slides = SlidesListLocalTrain()
    labels = [s.label for s in all_slides]

    clf_epth_config = ClfEpthTrainCfg()
    clf_config = ClfTrainConfig()
    test_config = TestConfig()

    predicts = []
    metric_sum = 0
    for i, (train_index, test_index) in enumerate(folds_iter(args.folds_n, labels)):
        filenames_train = [all_slides[i].name for i in train_index]
        filenames_test = [all_slides[i].name for i in test_index]

        train_slides, test_slides = all_slides.split_train_test(filenames_test)

        df_annotations = pd.read_csv(test_config.annotations_path)
        df_annotations_train = df_annotations[df_annotations.filename.isin(filenames_train)]
        df_annotations_test = df_annotations[df_annotations.filename.isin(filenames_test)]

        if args.check:
            # train_slides = train_slides[:10]
            # test_slides = test_slides[:5]
            clf_epth_config.num_epochs = 1
            clf_epth_config.workers_n = 0
            clf_config.num_epochs = 1
            clf_config.workers_n = 0

        # # train segm
        clf_epth_config.log_dir = f"./logdir-clf-epth-fold-{i}"
        clf_epth_config.wandb_run_name = f"{started_at}-epth-fold-{i}"
        train_clf_epth(clf_epth_config, train_slides, test_slides)

        # train clf
        clf_config.log_dir = f"./logdir-clf-fold-{i}"
        clf_config.wandb_run_name = f"{started_at}-clf-fold-{i}"
        train_clf(clf_config, all_slides, df_annotations_train, df_annotations_test)

        # infer test
        test_config.epth_model_path = Path(clf_epth_config.log_dir) / "checkpoints" / "best.pth"
        test_config.epth_model_type = clf_epth_config.network_type
        test_config.epth_image_size = clf_epth_config.image_size
        test_config.epth_batch_size = clf_epth_config.batch_size
        test_config.epth_pad = clf_epth_config.pad

        test_config.clf_model_path = Path(clf_config.log_dir) / "checkpoints" / "best.pth"
        test_config.clf_model_type = clf_config.network_type
        test_config.clf_image_size = clf_config.image_size

        # train random forest
        test_config.random_forest_model = f"./rf_fold{i}.clf"
        if test_config.use_random_forest:
            test_config.roi_probs_path = f"roi_probs_fold{i}_train.csv"
            test_config.train_random_forest = True
            predict(test_config, filenames_train)

        test_config.roi_probs_path = f"roi_probs_fold{i}_train.csv"
        test_config.train_random_forest = False
        test_config.roi_probs_path = f"roi_probs_fold{i}.csv"
        metric = predict(test_config, filenames_test)
        metric_sum += metric
        logging.info(f">>>>>  Fold #{i} metric: {metric:.3f}")

    logging.info(f"Final avg metric: {metric_sum/args.fold_n:.3f}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--folds-n", default=5, type=int)
    parser.add_argument("--check", action="store_true")
    main(parser.parse_args())
