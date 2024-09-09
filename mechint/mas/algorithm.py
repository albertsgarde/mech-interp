import itertools
import math
import random
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import wandb
from datasets import IterableDataset  # type: ignore[import]
from jaxtyping import Float
from torch import Tensor
from transformer_lens.hook_points import HookPoint  # type: ignore[import]
from transformer_lens.HookedTransformer import HookedTransformer  # type: ignore[import]

from mechint.layer import Layer  # type: ignore[import]

from ..device import Device, get_device
from .sample_loader import SampleDataset
from .weighted_samples_store import WeightedSamplesStore


@dataclass
class BinRange:
    start: float
    end: float
    num_bins: int


@dataclass
class BinSet:
    ranges: list[BinRange] | None = None
    bin_list: list[float] | None = None


@dataclass
class MASParams:
    """
    Parameters for the MAS algorithm.

    Args:
        high_activation_weighting: How much to prefer samples with high activation.
        sample_overlap: The number of tokens that overlap between samples.
        num_max_samples: The number of samples to store per feature.
        sample_length_pre: The number of tokens to store before the high activation token.
        sample_length_post: The number of tokens to store after the high activation token.
        samples_to_check: The number of samples to check.
        seed: The seed to use for sampling.
        activation_bins: The bins to use for the activation histogram.
    """

    high_activation_weighting: float
    firing_threshold: float
    sample_overlap: int
    num_max_samples: int
    sample_length_pre: int
    sample_length_post: int
    seed: int
    activation_bins: BinSet
    samples_to_check: int | None = None
    max_time: float | None = None

    def __post_init__(self):
        if self.sample_overlap < 0:
            raise ValueError("Sample overlap must be at least 0.")
        if self.num_max_samples and self.num_max_samples <= 0:
            raise ValueError("Number of max samples must be greater than 0.")
        if self.sample_length_pre < 0:
            raise ValueError("Sample length pre must be at least 0.")
        if self.sample_length_post <= 0:
            raise ValueError("Sample length post must be greater than 0.")
        if not self.samples_to_check and not self.max_time:
            raise ValueError("Either samples to check or max time must be set.")
        if self.samples_to_check and self.samples_to_check <= 0:
            raise ValueError("Samples to check must be greater than 0.")
        if self.max_time and self.max_time <= 0:
            raise ValueError("Max time must be greater than 0.")
        if isinstance(self.activation_bins, dict):
            if "ranges" in self.activation_bins:
                self.activation_bins["ranges"] = [
                    BinRange(**range_dict) for range_dict in self.activation_bins["ranges"]
                ]

            self.activation_bins = BinSet(**self.activation_bins)


def run(
    model: HookedTransformer,
    dataset: IterableDataset,
    layers: list[Layer],
    params: MASParams,
    device: Device,
    log_wandb: bool,
) -> WeightedSamplesStore:
    with torch.no_grad():
        device = get_device()
        print(f"Using device: {device.torch()}")

        if params.samples_to_check:
            dataset = dataset.take(params.samples_to_check)  # type: ignore[reportUnknownMemberType]

        if model.tokenizer is None:
            raise ValueError("Model must have tokenizer.")
        if model.tokenizer.pad_token_id is None:
            raise ValueError("Model tokenizer must have pad token.")
        if model.cfg is None:
            raise ValueError("Model must have config.")

        context_size = model.cfg.n_ctx
        print(f"Model context size: {context_size}")

        sample_dataset = SampleDataset(context_size, params.sample_overlap, model, dataset)

        bins: list[float] = params.activation_bins.bin_list if params.activation_bins.bin_list is not None else []
        if params.activation_bins.ranges is not None:
            for range in params.activation_bins.ranges:
                bins.extend(np.linspace(range.start, range.end, range.num_bins, endpoint=True).tolist())
        bins.sort()

        num_total_features = sum([layer.num_features for layer in layers])
        rng = random.Random(params.seed)
        mas_store = WeightedSamplesStore(
            bins,
            params.high_activation_weighting,
            params.firing_threshold,
            params.num_max_samples,
            num_total_features,
            context_size,
            params.sample_length_pre,
            params.sample_length_post,
            model.tokenizer.pad_token_id,
            rng,
            device,
        )

        activation_scratch = torch.zeros((context_size, num_total_features), device=device.torch())

        def create_hook(
            layer: Layer, slice: slice
        ) -> tuple[str, Callable[[Float[Tensor, "batch context neurons_per_layer"], HookPoint], None]]:
            assert layer.num_features == slice.stop - slice.start

            def hook(activation: Float[Tensor, "batch context neurons_per_layer"], hook: HookPoint) -> None:
                activation_scratch[:, slice] = layer.activation_map(activation)[0, :, :]

            return (layer.hook_id, hook)

        indices = np.cumsum([0] + [layer.num_features for layer in layers])
        slices = [slice(start, end) for start, end in zip(indices[:-1], indices[1:], strict=True)]
        hooks = [create_hook(layer, slice) for layer, slice in zip(layers, slices, strict=True)]

        last_percentage = -1

        model_time = 0.0
        mas_time = 0.0
        start_time = time.time()
        for i, sample in itertools.islice(enumerate(sample_dataset), params.samples_to_check):
            model_start_time = time.time()
            model.run_with_hooks(sample.tokens, fwd_hooks=hooks)
            model_time += time.time() - model_start_time
            mas_start_time = time.time()
            mas_store.add_sample(sample, activation_scratch)
            mas_time += time.time() - mas_start_time
            assert mas_store.num_samples_added() == i + 1
            if log_wandb:
                time_elapsed = time.time() - start_time
                log_dict = {
                    "samples_processed": mas_store.num_samples_added(),
                    "model_time_percent": model_time / time_elapsed * 100,
                    "mas_time_percent": mas_time / time_elapsed * 100,
                }
                wandb.log(log_dict)

            if params.samples_to_check:
                cur_percentage = int(math.floor(i / params.samples_to_check * 100))
                if cur_percentage > last_percentage:
                    print(f"{cur_percentage}%")
                    last_percentage = cur_percentage
            if params.max_time and time.time() - start_time > params.max_time:
                print("Time limit reached.")
                break
        samples_processed = mas_store.num_samples_added()
        end_time = time.time()
        time_elapsed = end_time - start_time
        print(f"Samples processed: {samples_processed}")
        print(f"Time taken: {time_elapsed:.2f}s")
        time_per_sample = f"{time_elapsed / samples_processed / 1000:.2f}ms" if samples_processed > 0 else "NA"
        print(f"Time taken per sample: {time_per_sample}")

        print(f"Model time: {model_time:.2f}s ({model_time/time_elapsed*100:.2f}%)")
        print(f"MAS time: {mas_time:.2f}s ({mas_time/time_elapsed*100:.2f}%)")

        mas_store._sort_samples()
        return mas_store
