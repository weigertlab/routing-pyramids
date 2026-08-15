"""Sampling helpers for deterministic dataset traversal and temporal subsampling."""

import torch


def fixed_shuffled_indices(num_items: int, seed: int | None = None) -> list[int]:
    """
    Build a fixed shuffled index order.

    Parameters
    ----------
    num_items : int
        Number of items in the dataset.
    seed : int, optional
        Seed for the local generator. If not provided, uses
        ``torch.initial_seed()``.

    Returns
    -------
    list[int]
        Deterministic permutation of ``range(num_items)``.
    """
    if num_items < 0:
        raise ValueError(f"num_items must be non-negative, got {num_items}")

    effective_seed = torch.initial_seed() if seed is None else int(seed)
    generator = torch.Generator()
    generator.manual_seed(effective_seed)
    return torch.randperm(num_items, generator=generator).tolist()


def validate_frame_indices(
    frame_indices: torch.Tensor,
    *,
    batch_size: int | None = None,
    timesteps: int | None = None,
    source_length: int | None = None,
) -> torch.Tensor:
    """
    Validate frame indices and return a ``long`` tensor with shape ``(B, T)``.

    Parameters
    ----------
    frame_indices : Tensor
        Frame indices with shape ``(T,)`` or ``(B, T)``.
    batch_size : int, optional
        Expected batch size if ``frame_indices`` is 2-D.
    timesteps : int, optional
        Expected number of timesteps.
    source_length : int, optional
        Source window length used for basic sanity checks.
    """
    if frame_indices.ndim == 1:
        frame_indices = frame_indices.unsqueeze(0)
    elif frame_indices.ndim != 2:
        raise ValueError(
            "frame_indices must have shape (T,) or (B, T), "
            f"got {tuple(frame_indices.shape)}"
        )

    frame_indices = frame_indices.to(dtype=torch.long)
    bsz, t = frame_indices.shape

    if batch_size is not None and bsz != batch_size:
        raise ValueError(
            f"frame_indices batch size mismatch: expected {batch_size}, got {bsz}"
        )
    if timesteps is not None and t != timesteps:
        raise ValueError(
            f"frame_indices time size mismatch: expected {timesteps}, got {t}"
        )

    if source_length is not None:
        source_length = int(source_length)
        if source_length <= 0:
            raise ValueError(f"source_length must be positive, got {source_length}")
        if source_length < t:
            raise ValueError(
                "source_length must be >= number of provided timesteps, "
                f"got source_length={source_length}, timesteps={t}"
            )

    return frame_indices


def temporal_subsample_indices(
    source_length: int,
    keep_length: int,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Sample temporal indices with fixed endpoints.

    The first and last frames are always preserved, and the remaining
    ``keep_length - 2`` indices are sampled uniformly without replacement
    from the middle range ``[1, source_length - 2]``.

    Parameters
    ----------
    source_length : int
        Number of frames in the source window.
    keep_length : int
        Number of frames to keep after subsampling.

    Returns
    -------
    torch.Tensor
        Long tensor of shape ``(keep_length,)`` sorted in ascending order.
    """
    if source_length <= 0:
        raise ValueError(f"source_length must be positive, got {source_length}")
    if keep_length <= 0:
        raise ValueError(f"keep_length must be positive, got {keep_length}")
    if keep_length > source_length:
        raise ValueError(
            "keep_length must be <= source_length, "
            f"got keep_length={keep_length}, source_length={source_length}"
        )

    if keep_length == source_length:
        return torch.arange(source_length, dtype=torch.long)

    if keep_length < 2:
        raise ValueError(
            "keep_length must be at least 2 when source_length > keep_length, "
            f"got keep_length={keep_length}, source_length={source_length}"
        )

    if source_length < 2:
        raise ValueError(
            "source_length must be at least 2 when source_length > keep_length, "
            f"got source_length={source_length}, keep_length={keep_length}"
        )

    middle_count = keep_length - 2
    if middle_count == 0:
        return torch.tensor([0, source_length - 1], dtype=torch.long)

    middle_total = source_length - 2
    middle = torch.randperm(middle_total, generator=generator)[:middle_count] + 1
    middle, _ = middle.sort()
    return torch.cat(
        [
            torch.tensor([0], dtype=torch.long),
            middle.to(dtype=torch.long),
            torch.tensor([source_length - 1], dtype=torch.long),
        ]
    )
