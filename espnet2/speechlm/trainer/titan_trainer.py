# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""TorchTitan-based trainer implementation for SpeechLM training.

This trainer uses FSDP2 for data parallelism, providing an alternative to
DeepSpeed-based training. It maintains interface compatibility with DeepSpeedTrainer.
"""

import gc
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
import wandb
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)

from torchtitan.distributed import utils as dist_utils

from espnet2.speechlm.utils.data import to_device
from espnet2.speechlm.utils.model_summary import model_summary
from espnet2.speechlm.model.speechlm.parallel_utils import (
    init_parallel_dims,
    parallel_strategies,
)

logger = logging.getLogger(__name__)


class TitanTrainer:
    """TorchTitan-based trainer with FSDP2 support for SpeechLM.

    This trainer provides distributed training using PyTorch's native FSDP2
    (Fully Sharded Data Parallel) instead of DeepSpeed ZeRO. It maintains
    interface compatibility with DeepSpeedTrainer for easy switching.

    IMPORTANT: wandb is MANDATORY and must be initialized before creating this trainer.
    The trainer will raise an error if wandb.run is None.
    Wandb should always be initialized in offline mode for local-only logging.
    All training metrics, losses, and stats are logged to local wandb files.

    Key Features:
        - FSDP2/HSDP for memory-efficient data parallelism
        - Activation checkpointing for memory optimization
        - PyTorch Distributed Checkpoint (DCP) for reshardable checkpoints
        - Compatible with existing DataIteratorFactory interface
    """

    def __init__(
        self,
        train_data_factory,
        valid_data_factories: Dict,
        model: nn.Module,
        resume_path: Optional[Path],
        output_dir: Path,
        trainer_args: Dict[str, Any],
    ):
        """Initialize TorchTitan trainer.

        Args:
            train_data_factory: Training data iterator factory
            valid_data_factories: Dictionary of validation data factories
            model: Model to train (HuggingFace model)
            resume_path: Path to checkpoint for resuming training
            output_dir: Directory for saving outputs
            trainer_args: Training configuration dictionary containing:
                - max_step: Maximum number of training steps
                - log_interval: Steps between logging
                - save_interval: Steps between checkpoints
                - freeze_param: List of parameter prefixes to freeze
                - titan_config: TorchTitan configuration dict with:
                    - dp_shard: FSDP degree (-1 = auto)
                    - dp_replicate: HSDP replicate degree (default: 1)
                    - mixed_precision_param: Parameter dtype (default: "bfloat16")
                    - mixed_precision_reduce: Reduce dtype (default: "float32")
                    - gradient_clipping: Max gradient norm (default: 1.0)
                    - optimizer: Optimizer config dict
                    - lr_scheduler: LR scheduler config dict
                    - activation_checkpoint: "none", "selective", or "full"
        """
        if wandb.run is None:
            raise RuntimeError(
                "wandb must be initialized before creating TitanTrainer. "
                "Please call wandb.init() first."
            )

        self.train_data_factory = train_data_factory
        self.valid_data_factories = valid_data_factories
        self.output_dir = Path(output_dir)
        self.trainer_args = trainer_args
        (self.output_dir / "checkpoints").mkdir(exist_ok=True, parents=True)

        self.global_step = 0
        self.max_step = trainer_args["max_step"]
        self.save_interval = trainer_args["save_interval"]
        self.log_interval = trainer_args["log_interval"]
        self.gradient_accumulation_steps = trainer_args.get(
            "gradient_accumulation_steps", 1
        )

        # Freeze parameters
        for t in trainer_args.get("freeze_param", []):
            for k, p in model.named_parameters():
                if k.startswith(t + ".") or k == t:
                    logger.info(f"Setting {k}.requires_grad = False")
                    p.requires_grad = False

        # Initialize distributed environment and ParallelDims
        titan_config = trainer_args.get("titan_config", {})

        # Determine training dtype and cast model before FSDP wrapping
        # This ensures all parameters have uniform dtype for FSDP
        dtype_str = titan_config.get("mixed_precision_param", "bfloat16")
        self.dtype = getattr(torch, dtype_str)
        self.max_norm = titan_config.get("gradient_clipping", 1.0)

        self.parallel_dims, self.local_rank, self.global_rank = init_parallel_dims(
            titan_config
        )

        self.device = torch.device(f"cuda:{self.local_rank}")
        model = model.to(device=self.device, dtype=self.dtype)

        parallel_strategy = titan_config.get("parallel_strategy", "qwen3")
        parallelize_fn = parallel_strategies[parallel_strategy]
        self.model = parallelize_fn(model, self.parallel_dims, titan_config)

        logger.info(model_summary(model))

        # Build optimizer and scheduler (after parallelization)
        self._build_optimizer_scheduler()

        # Load checkpoint if exists
        self._load_checkpoint(resume_path)

        # Disable automatic GC to prevent distributed straggler issues.
        # Random GC pauses on one rank stall all ranks at collectives.
        # We run a lightweight gen-1 collection every gc_freq steps instead.
        self.gc_freq = titan_config.get("gc_freq", 1000)
        gc.disable()
        gc.collect()
        logger.info(f"Disabled automatic GC, will collect every {self.gc_freq} steps")

        # Log configuration
        wandb.config.update({"titan_config": titan_config})
        logger.info("Successfully initialized TitanTrainer with configuration:")
        logger.info(f"  FSDP enabled: {self.parallel_dims.fsdp_enabled}")
        logger.info(f"  dp_shard: {self.parallel_dims.dp_shard}")
        logger.info(f"  dp_replicate: {self.parallel_dims.dp_replicate}")
        logger.info(f"  gradient_accumulation_steps: {self.gradient_accumulation_steps}")

    def _build_optimizer_scheduler(self):
        """Create optimizer and LR scheduler after parallelization."""
        opt_config = self.trainer_args.get("optimizer", {})
        lr_config = self.trainer_args.get("lr_scheduler", {})

        # Get trainable parameters
        params = [p for p in self.model.parameters() if p.requires_grad]
        total_params = sum(p.numel() for p in params)
        logger.info(f"Total trainable parameters: {total_params:,}")

        # Build optimizer
        optimizer_name = opt_config.get("name", "AdamW")
        optimizer_cls = getattr(torch.optim, optimizer_name)

        optimizer_kwargs = {
            "lr": opt_config.get("lr", 1e-4),
            "betas": (opt_config.get("beta1", 0.9), opt_config.get("beta2", 0.95)),
            "eps": opt_config.get("eps", 1e-8),
            "weight_decay": opt_config.get("weight_decay", 0.01),
        }

        # Use fused optimizer if available (faster on CUDA)
        if optimizer_name == "AdamW" and torch.cuda.is_available():
            optimizer_kwargs["fused"] = True

        self.optimizer = optimizer_cls(params, **optimizer_kwargs)
        logger.info(
            f"Created {optimizer_name} optimizer with lr={optimizer_kwargs['lr']}"
        )

        # Build LR scheduler with warmup and cosine decay
        warmup_steps = lr_config.get("warmup_steps", 1000)
        decay_type = lr_config.get("decay_type", "cosine")

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))

            if decay_type == "cosine":
                progress = (step - warmup_steps) / max(1, self.max_step - warmup_steps)
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            elif decay_type == "linear":
                progress = (step - warmup_steps) / max(1, self.max_step - warmup_steps)
                return max(0.0, 1.0 - progress)
            elif decay_type == "constant":
                return 1.0
            else:
                raise ValueError(
                    f"Unknown decay_type: {decay_type}. "
                    f"Supported types: 'cosine', 'linear', 'constant'"
                )

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda
        )
        logger.info(
            f"Created LR scheduler with warmup_steps={warmup_steps}, "
            f"decay_type={decay_type}"
        )

    def _save_checkpoint(self, step: int) -> None:
        """Save checkpoint using PyTorch Distributed Checkpoint.

        Args:
            step: Current training step
        """
        checkpoint_dir = self.output_dir / "checkpoints" / f"step_{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Get state dicts
        model_state = get_model_state_dict(self.model)
        optimizer_state = get_optimizer_state_dict(self.model, self.optimizer)

        state = {
            "model": model_state,
            "optimizer": optimizer_state,
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "global_step": self.global_step,
        }

        # Save using DCP
        dcp.save(state, checkpoint_id=str(checkpoint_dir))

        if self.global_rank == 0:
            logger.info(f"Saved checkpoint to {checkpoint_dir}")

    def _load_checkpoint(self, resume_path: Optional[Path]) -> None:
        """Load checkpoint for resuming training.

        Args:
            resume_path: Optional path to checkpoint directory
        """
        checkpoint_dir = None

        # Step 1: Check resume_path
        if resume_path and resume_path.exists():
            checkpoint_dir = resume_path

        # Step 2: Check latest checkpoint in output_dir
        elif (self.output_dir / "checkpoints").exists():
            ckpt_base = self.output_dir / "checkpoints"
            checkpoints = [
                d for d in ckpt_base.iterdir()
                if d.is_dir() and d.name.startswith("step_")
            ]
            if checkpoints:
                # Sort by step number and get latest
                checkpoint_dir = sorted(
                    checkpoints,
                    key=lambda x: int(x.name.split("step_")[-1]),
                    reverse=True,
                )[0]

        if checkpoint_dir and checkpoint_dir.is_dir():
            try:
                # Prepare empty state dict structures
                model_state = get_model_state_dict(self.model)
                optimizer_state = get_optimizer_state_dict(self.model, self.optimizer)

                state = {
                    "model": model_state,
                    "optimizer": optimizer_state,
                    "lr_scheduler": {},
                    "global_step": 0,
                }

                # Load using DCP
                dcp.load(state, checkpoint_id=str(checkpoint_dir))

                # Apply loaded states
                set_model_state_dict(
                    self.model,
                    state["model"],
                    options=StateDictOptions(strict=False),
                )
                set_optimizer_state_dict(
                    self.model,
                    self.optimizer,
                    state["optimizer"],
                )

                if state["lr_scheduler"]:
                    self.lr_scheduler.load_state_dict(state["lr_scheduler"])

                self.global_step = state["global_step"]

                logger.info(
                    f"Loaded checkpoint: {checkpoint_dir} | step={self.global_step}"
                )
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
                logger.info("Starting from step 0")
        else:
            logger.info("No checkpoint found, starting from step 0")

    def _all_reduce_stats(self, stats: Dict[str, torch.Tensor]) -> None:
        """Perform async all_reduce on statistics for efficient multi-GPU sync.

        Args:
            stats: Dictionary of statistics tensors to reduce across GPUs.
                   Modified in-place to contain the averaged values.
        """
        if not dist.is_initialized():
            return

        handles = []
        world_size = dist.get_world_size()

        # Launch all async all_reduce operations (non-blocking)
        for key in stats:
            if not isinstance(stats[key], torch.Tensor):
                stats[key] = torch.tensor(stats[key], device=self.device)

            handle = dist.all_reduce(
                stats[key],
                op=dist.ReduceOp.SUM,
                async_op=True,
            )
            handles.append((key, handle))

        # Wait for all operations to complete and compute mean
        for key, handle in handles:
            if handle is not None:
                handle.wait()
            stats[key] = stats[key] / world_size

    def run(self) -> None:
        """Main training loop."""
        logger.info(f"Starting training from step {self.global_step} to {self.max_step}")

        while self.global_step < self.max_step:
            self.train()
            self.valid()

            # Save checkpoint
            self._save_checkpoint(self.global_step)

        logger.info("Training completed!")

    def train(self) -> None:
        """Execute one training epoch (save_interval optimizer steps).

        With gradient accumulation, each optimizer step consumes
        ``gradient_accumulation_steps`` micro-batches. The iterator is
        sized so that ``save_interval`` optimizer steps are performed.
        """
        self.model.train()
        grad_accum = self.gradient_accumulation_steps

        # Request enough micro-batches for save_interval optimizer steps.
        # Use global_step directly as batch offset (not multiplied by
        # grad_accum) so checkpoint resume is independent of grad_accum.
        iterator = self.train_data_factory.build_iter(
            global_step=self.global_step,
            length=self.save_interval * grad_accum,
        )
        data_iter = iter(iterator)

        for _ in range(self.save_interval):
            iter_start = time.time()
            self.optimizer.zero_grad(set_to_none=True)

            # Accumulate gradients over micro-batches
            accumulated_stats = {}
            for _micro in range(grad_accum):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                batch = to_device(batch, self.device, dtype=self.dtype)

                out = self.model(**batch)
                loss = out["loss"] / grad_accum
                loss.backward()

                # Accumulate stats (sum, will average later)
                for k, v in out["stats"].items():
                    if k not in accumulated_stats:
                        accumulated_stats[k] = v.detach()
                    else:
                        accumulated_stats[k] = accumulated_stats[k] + v.detach()

            # Average stats over micro-batches
            for k in accumulated_stats:
                accumulated_stats[k] = accumulated_stats[k] / grad_accum

            # Gradient clipping (torchtitan version handles DTensor/FSDP2)
            grad_norm = dist_utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_norm,
                foreach=True,
            )

            # Optimizer step
            self.optimizer.step()
            self.lr_scheduler.step()

            # Sync and log statistics
            self._all_reduce_stats(accumulated_stats)
            stats = {
                f"train/{k}": float(v.cpu())
                for k, v in accumulated_stats.items()
            }
            stats["train/lr"] = self.lr_scheduler.get_last_lr()[0]
            stats["train/grad_norm"] = (
                grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            )
            stats["time/iter"] = time.time() - iter_start

            # Log to wandb
            wandb.log(stats, step=self.global_step)

            # Console logging (rank 0 only, 4 significant digits)
            if self.global_rank == 0 and self.global_step % self.log_interval == 0:
                short = {k: f"{v:.4g}" for k, v in stats.items()}
                logger.info(f"step {self.global_step}, stats: {short}")

            # Periodic lightweight GC to reclaim memory without straggler stalls
            if self.global_step > 1 and self.global_step % self.gc_freq == 0:
                gc.collect(1)

            self.global_step += 1

    def valid(self) -> None:
        """Run validation on all validation datasets."""
        self.model.eval()

        for name, factory in self.valid_data_factories.items():
            iterator = factory.build_iter()

            # Sync the number of validation steps across ranks so that all
            # ranks iterate the same number of times (avoiding all_reduce deadlock).
            local_len = torch.tensor([len(iterator)], device=self.device, dtype=torch.long)
            dist.all_reduce(local_len, op=dist.ReduceOp.MIN)
            num_valid_steps = int(local_len.item())

            # Collect all batch metrics
            all_stats = {}

            with torch.no_grad():
                for _, batch in zip(range(num_valid_steps), iterator):
                    batch = to_device(batch, self.device, dtype=self.dtype)
                    out = self.model(**batch)

                    stats = out["stats"]
                    self._all_reduce_stats(stats)

                    stats = {k: float(v.cpu()) for k, v in stats.items()}
                    for key, value in stats.items():
                        if key not in all_stats:
                            all_stats[key] = []
                        all_stats[key].append(value)

            # Compute averages and log
            all_stats = {
                f"val/{name}/{key}": sum(value) / len(value)
                for key, value in all_stats.items()
            }
            wandb.log(all_stats, step=self.global_step)

            if self.global_rank == 0:
                short = {k: f"{v:.4g}" for k, v in all_stats.items()}
                logger.info(f"Validation [{name}]: {short}")
