from pathlib import Path

import math
import os
from argparse import ArgumentParser

import pandas as pd
import cv2
import torch
import numpy as np
from catalyst.core import SchedulerCallback, CheckpointCallback
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
from src.annotation_dataset import AnnotationDataset, Annotation3ClassDataset, EpitheliumDataset
from src.region_mask_dataset import RegionMaskDataset
from src.resnet import resnet18, resnext50_32x4d
from src.roi_probs_dataset import RoiProbsDataset
from src.slide import SlidesList, SlidesListLocalTrain, UnlabeledSlidesListLocalTrain
from src.tools import create_bb_grid, draw_gt_annotations, Annotation, draw_pred_annotations, ERROR_TABLE, \
    best_page_for_sz


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

        precision, recall, f1, _ = precision_recall_fscore_support(y_classes, y_hat_classes, average='binary')
        accuracy = accuracy_score(y_classes, y_hat_classes)

        stat = {"loss": loss,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,

        }
        if y.shape[1] == 4:
            main_metric = 1 - ERROR_TABLE[y_classes, y_hat_classes].sum() / y_classes.shape[0]
            stat.update({
                "main_metric": main_metric,
                "precision2": precision[2], "precision3": precision[3],
            })

        self.batch_metrics.update(stat)

        if self.is_train_loader:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()


def get_transforms(image_size):
    pre_transforms = Compose([
        ToPILImage(),
        Resize((image_size, image_size)),
    ])
    hard_transforms = A.Compose([
        A.RandomRotate90(),
        A.VerticalFlip(),
        A.HorizontalFlip(),
        #A.RandomBrightnessContrast(),
        A.CoarseDropout(max_height=10, max_width=10),
        #A.ElasticTransform(),
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


def train_clf_epth(args, train_slides, test_slides):

    model = load_model(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loaders = get_loaders(args, train_slides, test_slides)

    callbacks = [
        SchedulerCallback(),
    ]
    if os.getenv("USE_WANDB"):
        wandb_run_name = f"epithelium-{args.network_type}-{args.image_size}-{args.pad}"
        callbacks.append(WandbCustomInputCallback(project="tissuenet", name=wandb_run_name))

    runner = CustomRunner()

    runner.train(
        model=model,
        optimizer=optimizer,
        loaders=loaders,
        logdir=args.log_dir,
        num_epochs=args.num_epochs,
        main_metric="f1",
        minimize_metric=False,
        verbose=True,
        load_best_on_end=True,
        initial_seed=args.seed,
        checkpoint_data={"network_type": args.network_type, "pad": args.pad, "image_size": args.image_size},
        scheduler=CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6),
        callbacks=callbacks,
        fp16=True,
    )
    return model


def load_model(args):
    if args.network_type.startswith("efficientnet"):
        model = EfficientNet.from_pretrained(model_name=args.network_type,
                                             dropout_rate=args.dropout_p, num_classes=2)
        # model = EfficientNet.from_name(model_name=args.network_type, num_classes=2, dropout_rate=args.dropout_p)
    elif args.network_type == "resnet18":
        model = resnet18(pretrained=True)
        model.fc = torch.nn.Linear(512 * 1, 2)
    elif args.network_type == "resnext50_32x4d":
        model = resnext50_32x4d(pretrained=True)
        model.fc = torch.nn.Linear(512 * 4, 2)
    else:
        raise NotImplementedError(f"Known network: {args.network_type}")
    return model


def build_df(slides):
    annotations = []
    for s in slides:
        for a in s.annotations:
            annotations.append(dict(
                filename=s.name,
                x1=a.bb.x1,
                y1=a.bb.y1,
                x2=a.bb.x2,
                y2=a.bb.y2,
                annotation_class=1,
            ))
    df = pd.read_csv("manual_annotations.csv")
    filenames = [s.name for s in slides]
    df = df[df.filename.isin(filenames)]
    df = df.append(annotations, ignore_index=True)
    return df


def get_loaders(args, train_slides, test_slides):
    train_transforms, val_transforms = get_transforms(args.image_size)
    common_dataset_args = dict(
        pad=args.pad,
        image_size=args.image_size,
    )

    train_df = build_df(train_slides)
    test_df = build_df(test_slides)
    train_datasets = [EpitheliumDataset(train_slides, train_df,
                                      transform=train_transforms, **common_dataset_args)]
    test_dataset = EpitheliumDataset(test_slides, test_df,
                                 transform=val_transforms, **common_dataset_args)

    cls_1_train = train_df.annotation_class.sum()
    cls_0_train = len(train_df) - train_df.annotation_class.sum()

    if not args.not_use_unlabeled:
        train_unlabeled_slides = UnlabeledSlidesListLocalTrain()
        train_unlabeled_df = pd.read_csv("manual_annotations_unlabeled.csv")
        cls_1_train += train_unlabeled_df.annotation_class.sum()
        cls_0_train += len(train_unlabeled_df) - train_unlabeled_df.annotation_class.sum()
        train_unlabeled_dataset = EpitheliumDataset(train_unlabeled_slides, train_unlabeled_df,
                                                    transform=train_transforms, **common_dataset_args)
        train_datasets.append(train_unlabeled_dataset)

    if args.all_data:
        cls_1_train += test_df.annotation_class.sum()
        cls_0_train += len(test_df) - test_df.annotation_class.sum()
        train_datasets.append(test_dataset)

    train_dataset = ConcatDataset(train_datasets)

    print(f"Train: {cls_1_train} {cls_0_train}")
    print(f"Test: {test_df.annotation_class.sum()} {len(test_df) - test_df.annotation_class.sum()}")
    common_loader_args = dict(
        pin_memory=True,
        num_workers=args.workers_n,
        batch_size=args.batch_size,
    )
    loaders = {
        "train": DataLoader(train_dataset,
                            sampler=BalanceClassSampler(train_df.annotation_class.values),
                            shuffle=False, **common_loader_args),
        "valid": DataLoader(test_dataset, shuffle=False, **common_loader_args),
    }
    return loaders


def main(args):
    train_slides = SlidesList.from_csv(
        args.tif_path,
        args.csv_path / "train_metadata.csv",
        args.csv_path / "train_annotations.csv",
        args.csv_path / "train_labels.csv",
    )
    test_slides = SlidesList.from_csv(
        args.tif_path,
        args.csv_path / "test_metadata.csv",
        args.csv_path / "test_annotations.csv",
        args.csv_path / "test_labels.csv",
    )
    train_clf_epth(args, train_slides, test_slides)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--tif-path", type=lambda x: Path(x), default=Path("/mnt/data/tissuenet-dataset"))
    parser.add_argument("--csv-path", type=lambda x: Path(x), default=Path("./inference-data-2"))
    parser.add_argument("--log-dir", default="./logdir-clf-epithelium")
    parser.add_argument("--network-type", default="efficientnet-b1")
    parser.add_argument("--dropout-p", default=0.5, type=float)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--pad", default=0, type=float)
    parser.add_argument("--image-size", default=512, type=int)
    parser.add_argument("--num-epochs", default=40, type=int)
    parser.add_argument("--workers-n", default=12, type=int)
    parser.add_argument("--lr", default=0.005, type=float)
    parser.add_argument("--all-data", action="store_true")
    parser.add_argument("--not-use-unlabeled", action="store_true")
    main(parser.parse_args())
