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
        reference_batch_size=128,
        lr=[]
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
        self.lod = 0
        self.reference_batch_size = reference_batch_size

        self.optimizer = optimizer
        self.base_lrs = []
        for _ in self.optimizer.param_groups:
            self.base_lrs.append(lr)

        self.last_epoch = last_epoch

        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
            last_epoch = 0

        self.last_epoch = last_epoch

        self.optimizer._step_count = 0
        self._step_count = 0
        self.step(last_epoch)

    def set_batch_size(self, batch_size, lod):
        self.batch_size = batch_size
        self.lod = lod
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            alpha = float(self.last_epoch) / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr[self.lod]
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            # * float(self.batch_size)
            # / float(self.reference_batch_size)
            for base_lr in self.base_lrs
        ]

    def state_dict(self):
        return {
            "last_epoch": self.last_epoch
        }

    def load_state_dict(self, state_dict):
        self.__dict__.update(dict(last_epoch=state_dict["last_epoch"]))


class ComboMultiStepLR:
    def __init__(
        self,
        optimizers, base_lr,
        **kwargs
    ):
        self.schedulers = dict()
        for name, opt in optimizers.items():
            self.schedulers[name] = WarmupMultiStepLR(opt, lr=base_lr, **kwargs)
        self.last_epoch = 0

    def set_batch_size(self, batch_size, lod):
        for x in self.schedulers.values():
            x.set_batch_size(batch_size, lod)

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
            x.load_state_dict(state_dict[k])

        last_epochs = [x.last_epoch for k, x in self.schedulers.items()]
        assert np.all(np.asarray(last_epochs) == last_epochs[0])
        self.last_epoch = last_epochs[0]

    def start_epoch(self):
        return self.last_epoch
