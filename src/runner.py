try:
    from apex.amp import amp
except ImportError:
    print('Nvidia apex is not installed')
    pass

try:
    import os

    if os.environ.get("USE_WANDB", "1") == "1":
        from catalyst.contrib.dl.runner import SupervisedWandbRunner as Runner
    else:
        from catalyst.dl import SupervisedRunner as Runner, Experiment
except ImportError:
    from catalyst.dl import SupervisedRunner as Runner


class ModelRunner(Runner):
    def __init__(self, model=None, device=None):
        super().__init__(
            model=model, device=device, input_key="image", output_key=None
        )
