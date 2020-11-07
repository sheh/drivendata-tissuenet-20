import logging
from pathlib import Path

import math
import os
from argparse import ArgumentParser

import pandas as pd
import cv2
import torch
import numpy as np
from catalyst.core import SchedulerCallback, CheckpointCallback, Callback, CallbackOrder, CallbackNode
from catalyst.data import BalanceClassSampler
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader, ConcatDataset
from catalyst import dl
from catalyst.data.cv import ToTensor
from catalyst.contrib.datasets import MNIST
from catalyst.utils import metrics
from torchvision.transforms import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import albumentations as A

from src.callbacks import WandbCustomCallback, WandbCustomInputCallback
from src.efficientnet import EfficientNet
from src.annotation_dataset import AnnotationDataset, Annotation3ClassDataset, PseudoAnnotationDataset
from src.region_mask_dataset import RegionMaskDataset
from src.roi_probs_dataset import RoiProbsDataset
from src.slide import SlidesList, SlidesListLocalTrain, UnlabeledSlidesListLocalTrain
from src.tools import create_bb_grid, draw_gt_annotations, Annotation, draw_pred_annotations, ERROR_TABLE, \
    best_page_for_sz


class FreezeModelForNEpochCallback(Callback):

    def __init__(self, epoch_n):
        super().__init__(CallbackOrder.external, CallbackNode.all)
        self._epoch_n = epoch_n

    def on_epoch_start(self, runner: "IRunner"):
        need_freeze = runner.epoch < self._epoch_n
        for param in runner.model.parameters():
            param.requires_grad = not need_freeze
        # always keep fc trainable
        for param in runner.model._fc.parameters():
            param.requires_grad = True


class CustomRunner(dl.Runner):

    def _prepare_for_stage(self, stage: str):
        super()._prepare_for_stage(stage)

    def predict_batch(self, batch):
        # model inference step
        return self.model(batch[0].to(self.device))

    def _handle_batch(self, batch):
        # model train/valid step
        x, y = batch

        y_hat = self.model(x)
        y_hat_sig = y_hat.sigmoid()
        loss = F.cross_entropy(y_hat, y.argmax(dim=1))

        y_hat_classes = y_hat_sig.detach().cpu().numpy().argmax(axis=1)
        y_classes = y.cpu().numpy().argmax(axis=1)

        main_metric = 1 - ERROR_TABLE[y_classes, y_hat_classes].sum() / y_classes.shape[0]
        precision, recall, f1, _ = precision_recall_fscore_support(y_classes, y_hat_classes, average='micro')
        accuracy = accuracy_score(y_classes, y_hat_classes)

        stat = {"loss": loss,
                "accuracy": accuracy,
                "main_metric": main_metric,
                "precision": precision,
                "recall": recall,
                "f1": f1,
        }

        self.batch_metrics.update(stat)

        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()


def get_transforms(args):
    pre_transforms = Compose([
        ToPILImage(),
        Resize((args.image_size, args.image_size)),
    ])
    hard_transforms = A.Compose([
        A.RandomRotate90(),
        A.VerticalFlip(),
        A.HorizontalFlip(),
        A.CoarseDropout(max_height=12, max_width=12),
        A.RandomBrightnessContrast(),
        A.ElasticTransform(),
    ])
    post_transforms = Compose([
        ToTensor(),
        #Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        Normalize(mean=(0.73108001, 0.54549926, 0.67233236), std=(0.11515842, 0.13832434, 0.11472176)),
    ])

    def train_transforms(img):
        img = pre_transforms(img)
        img = np.asarray(img).copy()
        img = hard_transforms(image=img)["image"]
        return post_transforms(img)

    val_trainsforms = Compose([pre_transforms, post_transforms])
    return train_transforms, val_trainsforms


def train_clf(args, slides, train_df, test_df):

    model = load_model(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loaders = get_loaders(args, slides, train_df, test_df)

    if args.resume:
        # scheduler = StepLR(optimizer, step_size=int(2*args.num_epochs/3))
        scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-5)

    callbacks = [
        SchedulerCallback(),
    ]
    if os.getenv("USE_WANDB"):
        wandb_run_name = f"clf-{args.network_type}-{args.image_size}-{args.pad}"
        callbacks.append(WandbCustomInputCallback(project="tissuenet", name=wandb_run_name))

    runner = CustomRunner()
    runner.train(
        model=model,
        optimizer=optimizer,
        loaders=loaders,
        logdir=args.log_dir,
        num_epochs=args.num_epochs,
        main_metric="main_metric",
        minimize_metric=False,
        verbose=True,
        load_best_on_end=True,
        initial_seed=args.seed,
        checkpoint_data={"network_type": args.network_type, "pad": args.pad, "image_size": args.image_size},
        scheduler=scheduler,
        callbacks=callbacks,
        fp16=True,
        resume=None,
    )
    # model inference
    # for prediction in runner.predict_loader(loader=loaders["valid"]):
    #     assert prediction.detach().cpu().numpy().shape[-1] == 4
    # model tracing
    # runner.trace(loader=loaders["valid"], logdir=args.log_dir)
    return model


def load_model(args):
    if args.use_epth:
        state_dict = torch.load(args.use_epth)
        print(f"Epth clf val metrics: {state_dict['valid_metrics']}")
        model_name = state_dict["checkpoint_data"]["network_type"]
        model = EfficientNet.from_name(model_name=model_name, num_classes=4, dropout_rate=args.dropout_p)
        model_state_dict = state_dict["model_state_dict"]
        model_state_dict.pop('_fc.weight')
        model_state_dict.pop('_fc.bias')
        model.load_state_dict(model_state_dict, strict=False)
    elif args.resume:
        model = EfficientNet.from_name(model_name=args.network_type, num_classes=4, dropout_rate=args.dropout_p)
        state_dict = torch.load(args.resume)
        print(f"Clf val metrics: {state_dict['valid_metrics']}")
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model = EfficientNet.from_pretrained(model_name=args.network_type,
                                             dropout_rate=args.dropout_p, num_classes=4)
    return model


def get_loaders(args, slides, train_df, test_df):

    test_dataset, train_dataset, train_labels = get_datasets(args, slides, train_df, test_df)
    # train_labels = [s.label for s in train_slides] * args.use_each_slide_times
    common_loader_args = dict(
        pin_memory=True,
        num_workers=args.workers_n,
        batch_size=args.batch_size,
    )
    loaders = {
        "train": DataLoader(train_dataset, shuffle=True,
                            #sampler=BalanceClassSampler(train_labels),
                            **common_loader_args),
        "valid": DataLoader(test_dataset, shuffle=False, **common_loader_args),
    }
    return loaders


def get_datasets(args, slides, train_df, test_df):
    print("Original annotations:")
    print("Train:")
    print(train_df.groupby("annotation_class").filename.count())
    print("Test:")
    print(test_df.groupby("annotation_class").filename.count())

    train_transforms, val_trainsforms = get_transforms(args)
    common_dataset_args = dict(
        pad=args.pad,
        image_size=args.image_size,
    )
    if args.use_pseudo_annotations:
        unlabeled_slides = UnlabeledSlidesListLocalTrain()
        train_pseudo_unlabeled = create_pseudo_labeled_dataset(args, "pseudo-clf.csv", common_dataset_args,
                                                               unlabeled_slides, train_transforms)
        train_datasets = [
            train_pseudo_unlabeled
        ]
        test_datasets = [
            AnnotationDataset(slides, train_df, transform=train_transforms, **common_dataset_args),
            AnnotationDataset(slides, test_df, transform=val_trainsforms, **common_dataset_args),
        ]
        train_labels = train_df.annotation_class.values.tolist() + test_df.annotation_class.values.tolist()
    else:
        train_datasets = [
            AnnotationDataset(slides, train_df, transform=train_transforms, **common_dataset_args),
        ]
        test_datasets = [
            AnnotationDataset(slides, test_df, transform=val_trainsforms, **common_dataset_args),
        ]
        train_labels = train_df.annotation_class.values

    train_dataset = ConcatDataset(train_datasets)
    test_dataset = ConcatDataset(test_datasets)
    return test_dataset, train_dataset, train_labels


def create_pseudo_labeled_dataset(args, filename, common_dataset_args, slides, train_transforms):
    df = pd.read_csv(filename)
    df = df[df["pseudo_class"] >= 0]
    df = df[df["prob"] > 0.8]
    sample_size = df.groupby("pseudo_class").filename.count().values.min()
    df_pseudo_stratif = pd.concat([
        df[df.pseudo_class == 0].sample(sample_size, random_state=args.seed, replace=True),
        df[df.pseudo_class == 1].sample(sample_size, random_state=args.seed, replace=True),
        df[df.pseudo_class == 2].sample(sample_size, random_state=args.seed, replace=True),
        df[df.pseudo_class == 3].sample(sample_size, random_state=args.seed, replace=True),
    ]).reset_index()
    print(filename)
    print(df_pseudo_stratif.groupby("pseudo_class").filename.count())
    return PseudoAnnotationDataset(slides, df_pseudo_stratif, transform=train_transforms, **common_dataset_args)


def main(args):
    train_df = pd.read_csv(args.csv_path / "train_annotations.csv")
    test_df = pd.read_csv(args.csv_path / "test_annotations.csv", )
    slides = SlidesListLocalTrain()
    train_clf(args, slides, train_df, test_df)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--tif-path", type=lambda x: Path(x))
    parser.add_argument("--csv-path", type=lambda x: Path(x))
    parser.add_argument("--log-dir", default="./logdir-clf")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--network-type", default="efficientnet-b1")
    parser.add_argument("--dropout-p", default=0.2, type=float)
    parser.add_argument("--batch-size", default=48, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--pad", default=0, type=float)
    parser.add_argument("--image-size", default=256, type=int)
    parser.add_argument("--num-epochs", default=40, type=int)
    parser.add_argument("--workers-n", default=10, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--all-data", action="store_true")
    parser.add_argument("--use-epth", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--use-pseudo-annotations", action="store_true")
    main(parser.parse_args())
