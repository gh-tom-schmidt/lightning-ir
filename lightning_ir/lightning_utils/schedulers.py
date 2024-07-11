from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence

from lightning import Callback, LightningModule, Trainer

from ..base import LightningIRModule


class LambdaWarmupScheduler(Callback, ABC):

    def __init__(
        self,
        keys: Sequence[str],
        num_warmup_steps: int,
        num_delay_steps: int = 0,
        num_training_steps: int = -1,
    ) -> None:
        super().__init__()
        self.keys = keys
        self.num_warmup_steps = num_warmup_steps
        self.num_delay_steps = num_delay_steps
        self.num_training_steps = num_training_steps
        self.values: Dict[str, float] = {}

    @abstractmethod
    def lr_lambda(self, current_step: int) -> float: ...

    def step(self, key: str, current_step: int) -> float:
        value = self.values[key]
        return value * self.lr_lambda(current_step)

    def get_value(self, sub_keys: Sequence[str], obj: object) -> object:
        for sub_key in sub_keys:
            try:
                obj = obj[int(sub_key)]
            except ValueError:
                obj = getattr(obj, sub_key)
        return obj

    def set_value(self, sub_keys: Sequence[str], obj: object, value: float) -> None:
        obj = self.get_value(sub_keys[:-1], obj)
        setattr(obj, sub_keys[-1], value)

    def on_train_start(self, trainer: Trainer, pl_module: LightningIRModule) -> None:
        if self.num_training_steps == -1:
            self.num_training_steps = trainer.estimated_stepping_batches
        for key in self.keys:
            sub_keys = key.split(".")
            self.values[key] = float(self.get_value(sub_keys, pl_module))

    def on_train_batch_start(
        self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int
    ) -> None:
        step = trainer.global_step + 1
        for key in self.keys:
            value = self.step(key, step)
            sub_keys = key.split(".")
            self.set_value(sub_keys, pl_module, value)


class LinearSchedulerWithWarmup(LambdaWarmupScheduler):
    def lr_lambda(self, current_step: int) -> float:
        if current_step < self.num_delay_steps:
            return 0.0
        current_step -= self.num_delay_steps + 1
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        return max(
            0.0,
            float(self.num_training_steps - current_step)
            / float(max(1, self.num_training_steps - self.num_warmup_steps)),
        )


class ConstantSchedulerWithWarmup(LambdaWarmupScheduler):

    def lr_lambda(self, current_step: int) -> float:
        if current_step < self.num_delay_steps:
            return 0.0
        current_step -= self.num_delay_steps + 1
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1.0, self.num_warmup_steps))
        return 1.0
