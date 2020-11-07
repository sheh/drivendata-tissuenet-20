from pathlib import Path

import cv2
import wandb
from catalyst.dl import ConfusionMatrixCallback, IRunner, Callback, CallbackOrder, CallbackNode, CallbackScope, \
    WandbLogger
import numpy as np
from wandb.sklearn import confusion_matrix


class ConfusionMatrixWandbCallback(ConfusionMatrixCallback):

    def on_stage_end(self, runner: IRunner):
        class_names = self.class_names or [
            str(i) for i in range(self.num_classes)
        ]
        wandb.log({'confusion_matrix': confusion_matrix(self.targets, self.outputs, class_names)},
                  commit=False,
                  )


class WandbCustomCallback(WandbLogger):

    def on_stage_end(self, runner: IRunner):
        wandb.save(str(Path(runner.logdir) / "checkpoints" / "best_full.path"))
        wandb.save(str(Path(runner.logdir) / "checkpoints" / "last_full.path"))


class WandbCustomInputCallback(WandbCustomCallback):

    def on_stage_start(self, runner: "IRunner"):
        super().on_stage_start(runner)
        images = []
        for batch in runner.loaders.get("train"):
            for img, cls in zip(batch[0], batch[1]):
                img = img.numpy().transpose((1, 2, 0))
                img = (img - img.min())
                img = (img / img.max() * 255).astype(np.uint8)
                images.append(wandb.Image(img, caption=f"cls: {cls}"))
            break
        wandb.log({"examples": images},
                  commit=False,
                  step=runner.global_sample_step)


class WandbCustomSegmCallback(WandbCustomCallback):

    def on_stage_start(self, runner: "IRunner"):
        super().on_stage_start(runner)
        images = []
        for batch in runner.loaders.get("train"):
            for img, mask in zip(batch[0], batch[1]):
                img = img.numpy().transpose((1, 2, 0))
                img = (img - img.min())
                img = (img / img.max() * 255).astype(np.uint8)
                mask = (255*mask.numpy()).astype(np.uint8)
                img = cv2.addWeighted(img, 0.6, cv2.cvtColor(mask[0], cv2.COLOR_GRAY2BGR), 0.4, 0)
                images.append(wandb.Image(img))
            break
        wandb.log({"examples": images},
                  commit=False,
                  step=runner.global_sample_step)

