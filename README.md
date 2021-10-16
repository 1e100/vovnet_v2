# VOVNet

This is a reasonably clean implementation of a classifier using VOVNet
(https://arxiv.org/abs/1904.09730). A very simple architecture which,
nevertheless produces very good accuracy, and is especially useful for object
detection applications.

The following ImageNet checkpoints are included:
 - [VoVNet19](https://f002.backblazeb2.com/file/1e100-public/vovnet19-72.03-top1.pt), prec1=72.011, prec5=90.716
 - [VoVNet27](https://f002.backblazeb2.com/file/1e100-public/vovnet27-73.07-top1.pt), prec1=73.070, prec5=91.580
 - [VoVNet39](https://f002.backblazeb2.com/file/1e100-public/vovnet39-78.82-top1.pt), prec1=78.817, prec5=94.265
 - [VoVNet57](https://f002.backblazeb2.com/file/1e100-public/vovnet57-79.63-top1.pt), prec1=79.63, prec5=94.68

Higher accuracies should be possible, this was one of the very first attempts
to train this.  Training was done with SGD, EMA and smooth CE. 

Detailed parameters:
```python3
CONFIGS = {
    "vovnet19": {
        "dropout": 0.2,
        "batch_size": 512,
        "loss": "smooth_ce",
        "loss_kwargs": {"smoothing": 0.1},
        "epochs": 50,
        "optimizer": "sgd",
        "optimizer_kwargs": {"momentum": 0.9, "weight_decay": 2e-5, "nesterov": True},
        "scheduler": "cosine",
        "scheduler_kwargs": {
            "num_cycles": 1,
            "peak_lr": 3.0,
            "min_lr": 1e-7,
            "total_lr_decay": 0.1,
            "initial_warmup_step_fraction": 0.0,
            "cycle_warmup_step_fraction": 0.1,
        },
        "trainer_kwargs": {"use_ema": True},
    },
    "vovnet27_slim": {
        "dropout": 0.2,
        "batch_size": 400,
        "loss": "smooth_ce",
        "loss_kwargs": {"smoothing": 0.1},
        "epochs": 100,
        "optimizer": "sgd",
        "optimizer_kwargs": {"momentum": 0.9, "weight_decay": 2e-5, "nesterov": True},
        "scheduler": "cosine",
        "scheduler_kwargs": {
            "num_cycles": 1,
            "peak_lr": 2.0,
            "min_lr": 1e-7,
            "total_lr_decay": 0.1,
            "initial_warmup_step_fraction": 0.0,
            "cycle_warmup_step_fraction": 0.1,
        },
        "trainer_kwargs": {"use_ema": True},
    },
    "vovnet39": {
        "dropout": 0.2,
        "batch_size": 300,
        "loss": "smooth_ce",
        "loss_kwargs": {"smoothing": 0.1},
        "epochs": 100,
        "optimizer": "sgd",
        "optimizer_kwargs": {"momentum": 0.9, "weight_decay": 2e-5, "nesterov": True},
        "scheduler": "cosine",
        "scheduler_kwargs": {
            "num_cycles": 1,
            "peak_lr": 2.0,
            "min_lr": 1e-7,
            "total_lr_decay": 0.1,
            "initial_warmup_step_fraction": 0.0,
            "cycle_warmup_step_fraction": 0.1,
        },
        "trainer_kwargs": {"use_ema": True},
    },
    "vovnet57": {
        "dropout": 0.2,
        "batch_size": 256,
        "epochs": 100,
        "loss": "smooth_ce",
        "loss_kwargs": {"smoothing": 0.1},
        "optimizer": "sgd",
        "optimizer_kwargs": {"momentum": 0.9, "weight_decay": 2e-5, "nesterov": True},
        "scheduler": "cosine",
        "scheduler_kwargs": {
            "num_cycles": 1,
            "peak_lr": 2.0,
            "min_lr": 1e-7,
            "total_lr_decay": 0.1,
            "initial_warmup_step_fraction": 0.0,
            "cycle_warmup_step_fraction": 0.1,
        },
        "trainer_kwargs": {"use_ema": True},
    },
    "vovnet99": {
        "dropout": 0.2,
        "batch_size": 160,
        "epochs": 100,
        "loss": "smooth_ce",
        "loss_kwargs": {"smoothing": 0.1},
        "optimizer": "sgd",
        "optimizer_kwargs": {"momentum": 0.9, "weight_decay": 2e-5, "nesterov": True},
        "scheduler": "cosine",
        "scheduler_kwargs": {
            "num_cycles": 1,
            "peak_lr": 0.8,
            "min_lr": 1e-7,
            "total_lr_decay": 0.1,
            "initial_warmup_step_fraction": 0.0,
            "cycle_warmup_step_fraction": 0.1,
        },
        "trainer_kwargs": {"use_ema": True},
    },
}
```
After each of these runs I recommend another run, also with EMA, but at 1/50th
the max learning rate (or so), to take up the slack on accuracy.
