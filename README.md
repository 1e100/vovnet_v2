# VOVNet

This is a reasonably clean implementation of a classifier using VOVNet
(https://arxiv.org/abs/1904.09730). A very simple architecture which,
nevertheless produces very good accuracy, and is especially useful for object
detection applications.

A VoVNet checkpoint with top1 of 77.86% is also included. Higher accuracies
should be possible, this was one of the very first attempts to train this.
Training was done with SGD, EMA and soft CE. 

Detailed parameters:
```python3
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
        "peak_lr": 1.5,
        "min_lr": 1e-7,
        "total_lr_decay": 0.1,
        "initial_warmup_step_fraction": 0.0,
        "cycle_warmup_step_fraction": 0.1,
    },
    "trainer_kwargs": {"use_ema": True},
},
```
