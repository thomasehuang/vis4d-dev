"""Callback for freezing model parameters."""

from __future__ import annotations

from torch import nn

from vis4d.common import ArgsType
from vis4d.common.logging import rank_zero_info
from vis4d.engine.callbacks.trainer_state import TrainerState
from vis4d.engine.loss_module import LossModule

from .base import Callback
from .trainer_state import TrainerState


class FreezeCallback(Callback):
    """Callback for model parameter freezing."""

    def __init__(
        self,
        *args: ArgsType,
        freeze_keys: str | list[str] | None = None,
        freeze_epoch: int = -1,
        verbose: bool = False,
        **kwargs: ArgsType,
    ) -> None:
        """Init callback.

        Args:
            freeze_keys (str, list[str], optional): Keys of the model
                parameters to freeze. Defaults to None (all).
            freeze_epoch (int, optional): Epoch to freeze the model. Defaults
                to -1 (start of training).
            verbose (bool, optional): Whether to print verbose messages.
                Defaults to False.
        """
        super().__init__(*args, **kwargs)
        self.freeze_keys = freeze_keys
        if self.freeze_keys and not isinstance(self.freeze_keys, list):
            self.freeze_keys = [self.freeze_keys]
        self.freeze_epoch = freeze_epoch
        self.verbose = verbose
        self.frozen = False

    def on_train_epoch_start(
        self,
        trainer_state: TrainerState,
        model: nn.Module,
        loss_module: LossModule,
    ) -> None:
        """Hook to run at the beginning of a training epoch."""
        if (
            self.frozen
            or (
                trainer_state["current_epoch"] < self.freeze_epoch - 1
                and self.freeze_epoch != -1
            )
        ):
            return
        rank_zero_info("Freezing model parameters.")
        if self.freeze_keys is None:
            for name, param in model.named_parameters():
                param.requires_grad = False
                if self.verbose:
                    rank_zero_info(f"Freezing {name}.")
        else:
            for name, param in model.named_parameters():
                if any(name.startswith(key) for key in self.freeze_keys):
                    param.requires_grad = False
                    if self.verbose:
                        rank_zero_info(f"Freezing {name}.")
                else:
                    param.requires_grad = True
        self.frozen = True
