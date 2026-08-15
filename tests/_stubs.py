"""Shared test stub classes."""

from __future__ import annotations

from argparse import Namespace
from typing import Any

import lightning as L
from lightning.pytorch.loggers.logger import Logger


class TrainerStub(L.Trainer):
    def __init__(
        self,
        *,
        experiment: object,
        global_step: int,
        current_epoch: int = 0,
        estimated_stepping_batches: int | None = None,
        max_epochs: float | None = None,
    ) -> None:
        self._logger: Logger | None = LoggerStub(experiment)
        self._global_step = global_step
        self._current_epoch = current_epoch
        self._estimated_stepping_batches = estimated_stepping_batches
        self._max_epochs = max_epochs

    @property
    def logger(self) -> Logger | None:
        return self._logger

    @logger.setter
    def logger(self, logger: Logger | None) -> None:
        self._logger = logger

    @property
    def global_step(self) -> int:
        return self._global_step

    @property
    def current_epoch(self) -> int:
        return self._current_epoch

    @current_epoch.setter
    def current_epoch(self, value: int) -> None:
        self._current_epoch = value

    @property
    def estimated_stepping_batches(self) -> float | int:
        return self._estimated_stepping_batches  # type: ignore[return-value]

    @property
    def max_epochs(self) -> int | None:
        return self._max_epochs  # type: ignore[return-value]


class LoggerStub(Logger):
    def __init__(self, experiment: object) -> None:
        super().__init__()
        self._experiment = experiment

    @property
    def experiment(self) -> object:
        return self._experiment

    @property
    def name(self) -> str:
        return "stub"

    @property
    def version(self) -> str:
        return "0"

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        return None

    def log_hyperparams(
        self, params: Namespace | dict[str, Any], *args: Any, **kwargs: Any
    ) -> None:
        return None
