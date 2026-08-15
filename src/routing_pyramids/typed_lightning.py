from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Generic, ParamSpec, TypeVar

from lightning.pytorch import LightningModule
from torch import Tensor

BatchT = TypeVar("BatchT")
ForwardP = ParamSpec("ForwardP")
ForwardOutputT = TypeVar("ForwardOutputT")
LightningStepOutput = Tensor | Mapping[str, Any] | None
StepOutputT = TypeVar("StepOutputT", bound=LightningStepOutput)
PredictOutputT = TypeVar("PredictOutputT")


class TypedLightningModule(
    LightningModule,
    Generic[BatchT, ForwardP, ForwardOutputT, StepOutputT, PredictOutputT],
):
    """
    Typed wrapper around Lightning's variadic hook surface.

    Several lightning hooks has signature ``(*args, **kwargs)``,
    which cannot be overridden with concrete signatures.
    This class centralizes type checker ignore statements.
    """

    # type: ignore[override]
    def forward(
        self, *args: ForwardP.args, **kwargs: ForwardP.kwargs
    ) -> ForwardOutputT:
        raise NotImplementedError

    # type: ignore[override]
    def training_step(
        self,
        batch: BatchT,
        batch_idx: int,
        dataloader_idx: int | None = None,
    ) -> StepOutputT:
        raise NotImplementedError

    # type: ignore[override]
    def validation_step(
        self,
        batch: BatchT,
        batch_idx: int,
        dataloader_idx: int | None = None,
    ) -> StepOutputT:
        raise NotImplementedError

    # type: ignore[override]
    def test_step(
        self,
        batch: BatchT,
        batch_idx: int,
        dataloader_idx: int | None = None,
    ) -> StepOutputT:
        raise NotImplementedError

    # type: ignore[override]
    def predict_step(
        self,
        batch: BatchT,
        batch_idx: int,
        dataloader_idx: int | None = None,
    ) -> PredictOutputT:
        raise NotImplementedError


__all__ = ["LightningStepOutput", "TypedLightningModule"]
