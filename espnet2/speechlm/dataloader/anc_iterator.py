"""AncDataLoader iterator wrapper for ESPnet2 SpeechLM training."""

import logging
from pathlib import Path
from typing import List, Optional, Iterator, Union

import numpy as np
import torch

from anc_data.anc_dataloader import AncDataLoader
from anc_data.anc_processor import Processor


def decode_audio(wav_bytes: bytes):
    """Decode raw WAV bytes from parquet into an audio array and sample rate.

    Uses soundfile to properly handle channel count, sample rate, bit depth,
    and WAV header variations.

    Args:
        wav_bytes: Raw WAV file bytes (including header).

    Returns:
        Tuple of (audio_array, sample_rate) where audio_array is a float32
        np.ndarray with shape [num_channels, num_samples].
    """
    import io
    import soundfile as sf

    audio, sample_rate = sf.read(io.BytesIO(wav_bytes))
    # sf.read returns [num_samples] for mono, [num_samples, num_channels] for multi
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]   # [1, num_samples]
    else:
        audio = audio.T                # [num_channels, num_samples]
    return audio.astype(np.float32), sample_rate


def convert_sample(item):
    """Convert a raw parquet row into the format expected by SpeechLMPreprocessor.

    Mirrors the output of CombinedDataset.__getitem__:
        key, data_dict = (task, dataset_name, sample_id), {"audio1": ..., "text1": ...}

    Args:
        item: Dict from parquet row with keys: 'index', 'wav', 'text',
            'task', 'dataset_name', etc.

    Returns:
        Tuple of (key, data_dict) where:
            key = (task, dataset_name, sample_id)
            data_dict = {"audio1": (np.ndarray, sample_rate), "text1": str}
    """
    sample_id = str(item["index"])
    task = str(item["task"])
    dataset_name = str(item["dataset_name"])

    data_dict = {}

    if "wav" in item and item["wav"] is not None:
        data_dict["audio1"] = decode_audio(item["wav"])

    if "text" in item and item["text"] is not None:
        data_dict["text1"] = str(item["text"])

    key = (task, dataset_name, sample_id)
    return key, data_dict


def parse_manifest_paths(specifier: str) -> List[List[List[str]]]:
    """Parse space-separated manifest file paths into 3-layer nested list.

    Each manifest file contains lines of comma-separated parquet paths.
    Multiple manifests (space-separated) map to multiple data sources.

    Args:
        specifier: Space-separated manifest file paths.

    Returns:
        Nested list: [[[parquet_files...], ...], ...] — one outer list per manifest,
        each containing groups of parquet file paths.
    """
    paths = []
    for manifest_path in specifier.strip().split():
        dataset = []
        with open(manifest_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    dataset.append(line.split(","))
        paths.append(dataset)
    return paths


class AncComposeProcessor(Processor):
    """Processor for compose batching in AncDataLoader.

    Handles per-sample transform and token length estimation for the composer.
    Collation is handled by AncIteratorFactory, not here.

    Args:
        max_seq_len: Maximum sequence length for composed batches.
        collate_fn: Collation function (used only for token length estimation).
    """

    def __init__(self, max_seq_len: int, collate_fn=None):
        self.max_seq_len = max_seq_len
        self._collate_fn = collate_fn

    def get_token_length_fn(self, sample):
        """Estimate token length from a raw parquet sample."""
        ans = self._collate_fn([convert_sample(sample)])

        if "position_ids" in ans:  # pack mode
            return ans['position_ids'].max() + 1
        else:  # bucket mode
            return ans['seqs'].size(1)

    def transform(self, item, is_last_sample=False):
        """Pass through the sample as-is."""
        return [item]

    def batch_transform(self, list_of_items, is_last_batch=False):
        """Convert raw parquet items and collate into training batches.

        When compose mode is enabled, list_of_items is a generator of
        micro-batches (each a list of raw samples). Each micro-batch is
        converted and collated into one training batch.

        When compose mode is disabled, list_of_items is a flat list of
        raw samples that gets collated into a single batch.
        """
        results = []
        for item in list_of_items:
            if isinstance(item, list):
                # Compose mode: item is a micro-batch of raw samples
                converted = [convert_sample(s) for s in item]
            else:
                # Non-compose mode: item is a single raw sample
                converted = [convert_sample(item)]
            if converted:
                results.append(self._collate_fn(converted))
        return results if results else []


class AncIteratorFactory:
    """
    Iterator factory using AncDataLoader, matching DataIteratorFactory interface.

    Args:
        paths: Nested list of parquet file paths, or a string of space-separated
               manifest file paths that will be parsed via parse_manifest_paths().
               List structure: [[[file1, file2], ...], ...]
        collate_fn: Collation function from SpeechLMPreprocessor.
        batch_size: Max token length per composed batch (passed as max_seq_len).
        rank: Distributed training rank (default: 0)
        world_size: Total number of distributed workers (default: 1)
        shuffle: Whether to shuffle data (default: False)
        repeat: Whether to loop data indefinitely (default: True).
            Set to False for validation to allow iteration to terminate.
        seed: Random seed for reproducibility (default: 0)
        num_workers: Number of data loading workers (default: 4)
        ds_args: Dict of dataset arguments passed to AncDataLoader (default: {}).
        loader_state: Path to directory for saving/loading checkpoints (default: None)
        ckpt_interval: Interval for internal checkpointing in AncDataLoader (default: None)
    """

    def __init__(
        self,
        paths: Union[str, List],
        collate_fn,
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = False,
        repeat: bool = True,
        seed: int = 0,
        num_workers: int = 4,
        ds_args: Optional[dict] = None,
        loader_state: Optional[str] = None,
        ckpt_interval: Optional[int] = None,
    ):
        if isinstance(paths, str):
            paths = parse_manifest_paths(paths)
        self.paths = paths
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.repeat = repeat
        self.seed = seed
        self.num_workers = num_workers
        self.ds_args = ds_args if ds_args is not None else {}

        # Setup loader state directory
        self.loader_state_dir: Optional[Path] = None
        if loader_state is not None:
            self.loader_state_dir = Path(loader_state)
            self.loader_state_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Loader state directory: {self.loader_state_dir}")

        self.ckpt_interval = ckpt_interval
        self._processor = AncComposeProcessor(
            max_seq_len=batch_size,
            collate_fn=collate_fn,
        )

        self._loader = AncDataLoader(
            paths=self.paths,
            batch_size=64,
            num_workers=self.num_workers,
            rank=self.rank,
            world=self.world_size,
            processor=self._processor,
            data_type="parquet",
            shuffle=self.shuffle,
            drop_last=True,
            seed=self.seed,
            repeat=self.repeat,
            need_xrank_sync=self.repeat,
            enable_compose=True,
            ckpt_interval=ckpt_interval,
            ds_args=self.ds_args,
        )

        # Persistent iterator for the forever loader
        self._loader_iter: Optional[Iterator] = None
        # Track if checkpoint has been loaded for current session
        self._checkpoint_loaded = False

    def _get_loader_iter(self) -> Iterator:
        """Get or create the persistent loader iterator."""
        if self._loader_iter is None:
            self._loader_iter = iter(self._loader)
        return self._loader_iter

    def build_iter(
        self, global_step: int = 0, length: Optional[int] = None
    ) -> Iterator:
        """Build and return an iterator that yields a fixed number of batches.

        This method returns a chunk of batches from the underlying forever
        iterator. Each call continues from where the previous call left off.

        Before iteration:
            - Loads matching checkpoint if found (only once per session).

        After iteration (if length is specified):
            - Saves checkpoint with the new global_step (global_step + length).

        Args:
            global_step: Current global training step (for checkpoint matching).
            length: Number of batches to yield. If None, iterate indefinitely.

        Returns:
            Iterator yielding exactly `length` batches (or indefinitely if None).
        """
        # Try to load checkpoint before first iteration (only once)
        if not self._checkpoint_loaded and self.loader_state_dir is not None:
            ckpt_name = (
                f"ckpt_rank{self.rank}_world{self.world_size}_step{global_step}.pt"
            )
            ckpt_path = self.loader_state_dir / ckpt_name
            if ckpt_path.exists():
                try:
                    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                    self._loader.set_checkpoint(ckpt)
                    logging.info(f"Loaded loader checkpoint from {ckpt_path}")
                except Exception as e:
                    logging.warning(f"Failed to load dataloader checkpoint: {e}")
            self._checkpoint_loaded = True

        loader_iter = self._get_loader_iter()

        if length is None:
            return loader_iter

        end_step = global_step + length

        def _chunk_iterator():
            for _ in range(length):
                yield next(loader_iter)

            # Save checkpoint after chunk is consumed
            if self.loader_state_dir is None or self.ckpt_interval is None:
                return
            try:
                ckpt = self._loader.get_checkpoint()
                if ckpt:
                    ckpt_name = (
                        f"ckpt_rank{self.rank}_world{self.world_size}"
                        f"_step{end_step}.pt"
                    )
                    ckpt_path = self.loader_state_dir / ckpt_name
                    torch.save(ckpt, ckpt_path)
                    logging.info(f"Saved loader checkpoint to {ckpt_path}")
            except Exception as e:
                logging.warning(f"Failed to save checkpoint: {e}")

        return _chunk_iterator()

    def __iter__(self) -> Iterator:
        return self._get_loader_iter()

    def __len__(self) -> int:
        return len(self._loader)
