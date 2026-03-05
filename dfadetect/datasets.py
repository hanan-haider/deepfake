"""Common preprocessing functions for audio data."""
import functools
import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import torch
import torchaudio
from torchaudio.functional import apply_codec



class TransformDataset(torch.utils.data.Dataset):
    """A generic transformation dataset.

    Takes another dataset as input, which provides the base input.
    When retrieving an item from the dataset, the provided transformation gets applied.

    Args:
        dataset: A dataset which return a (waveform, sample_rate)-pair.
        transformation: The torchaudio transformation to use.
        needs_sample_rate: Does the transformation need the sampling rate?
        transform_kwargs: Kwargs for the transformation.
    """

    def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            transformation: Callable,
            needs_sample_rate: bool = False,
            transform_kwargs: dict = {},
    ) -> None:
        super().__init__()
        self._dataset = dataset

        self._transform_constructor = transformation
        self._needs_sample_rate = needs_sample_rate
        self._transform_kwargs = transform_kwargs

        self._transform = None

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        waveform, sample_rate = self._dataset[index]

        if self._transform is None:
            if self._needs_sample_rate:
                self._transform = self._transform_constructor(
                    sample_rate, **self._transform_kwargs)
            else:
                self._transform = self._transform_constructor(
                    **self._transform_kwargs)

        return self._transform(waveform), sample_rate

