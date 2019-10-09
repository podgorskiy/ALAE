from bisect import bisect_right
import torch
import numpy as np


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 1.0,
        warmup_iters=1,
        last_epoch=-1,
        reference_batch_size=128
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.batch_size = 1
        self.reference_batch_size = reference_batch_size
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            alpha = float(self.last_epoch) / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            # * float(self.batch_size)
            # / float(self.reference_batch_size)
            for base_lr in self.base_lrs
        ]


class ComboMultiStepLR:
    def __init__(
        self,
        optimizers,
        **kwargs
    ):
        self.schedulers = dict()
        for name, opt in optimizers.items():
            self.schedulers[name] = WarmupMultiStepLR(opt, **kwargs)
        self.last_epoch = 0

    def set_batch_size(self, batch_size):
        for x in self.schedulers.values():
            x.set_batch_size(batch_size)

    def step(self, epoch=None):
        for x in self.schedulers.values():
            x.step(epoch)
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

    def state_dict(self):
        return {key: value.state_dict() for key, value in self.schedulers.items()}

    def load_state_dict(self, state_dict):
        for k, x in self.schedulers.items():
            x.__dict__.update(state_dict[k])
        last_epochs = [x.last_epoch for k, x in self.schedulers.items()]
        assert np.all(np.asarray(last_epochs) == last_epochs[0])
        self.last_epoch = last_epochs[0] + 1

    def start_epoch(self):
        return self.last_epoch
