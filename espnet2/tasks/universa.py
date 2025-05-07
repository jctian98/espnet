import argparse
import json
import logging
from pathlib import Path
from typing import Callable, Collection, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.frontend.s3prl import S3prlFrontend
from espnet2.asr.frontend.windowing import SlidingWindow
from espnet2.tasks.abs_task import AbsTask
from espnet2.text.phoneme_tokenizer import g2p_choices
from espnet2.torch_utils.initialize import initialize
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import UniversaCollateFn
from espnet2.train.preprocessor import UniversaProcessor
from espnet2.train.trainer import Trainer
from espnet2.universa.abs_universa import AbsUniversa
from espnet2.universa.base import UniversaBase
from espnet2.universa.base_flexible_type import UniversaBaseFlexibleType
from espnet2.universa.ar_universa import ARUniversa
from espnet2.universa.espnet_model import ESPnetUniversaModel
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import str2bool, str_or_none

frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(
        default=DefaultFrontend,
        sliding_window=SlidingWindow,
        s3prl=S3prlFrontend,
    ),
    type_check=AbsFrontend,
    default="default",
)
universa_choices = ClassChoices(
    "universa",
    classes=dict(
        base=UniversaBase,
        base_flex=UniversaBaseFlexibleType,
        ar_universa=ARUniversa,
    ),
    type_check=AbsUniversa,
    default="base",
)


def parse_metrics_meta(file_path, return_list=False):
    """
    Parse metric data from a text file.

    Args:
        file_path (str): Path to the text file containing metric data
        return_list (bool): If True, return only a list of metric names
                           If False, return a dictionary of metrics and their types

    Returns:
        dict or list: Dictionary with metric names as keys and types as values,
                     or a list of metric names if return_list is True
    """
    metrics_dict = {}
    metrics_list = []

    try:
        # Open and read the file
        with open(file_path, "r") as file:
            # Process each line in the file
            for line in file:
                line = line.strip()
                if line:  # Skip empty lines
                    parts = line.split()
                    if return_list:
                        # Only add the metric name to the list
                        metrics_list.append(parts[0])
                    elif len(parts) >= 2:
                        # Add the metric name and type to the dictionary
                        metric_name = parts[0]
                        metric_type = parts[1]
                        metrics_dict[metric_name] = metric_type

        return metrics_list if return_list else metrics_dict

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error processing file: {e}")
        return None


class UniversaTask(AbsTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    # Add variable objects configurations
    class_choices_list = [
        # --frontend and --frontend_conf
        frontend_choices,
        # --universa and --universa_conf
        universa_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

    @classmethod
    @typechecked
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        # NOTE(kamo): Use '_' instead of '-' to avoid confusion
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        required = parser.get_default("required")
        required += ["metric2id"]

        group.add_argument(
            "--metric2id",
            type=str,
            help="The mapping of metric to id",
        )
        group.add_argument(
            "--metric2type",
            type=str_or_none,
            default=None,
            help="The mapping of metric to type",
        )
        group.add_argument(
            "--metric_pad_value",
            type=float,
            default=-100,
            help="The padding value for metrics",
        )
        group.add_argument(
            "--token_list",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to token",
        )
        group.add_argument(
            "--metric_token_info",
            type=str_or_none,
            default=None,
            help="A token mapping int-id to token",
        )
        group.add_argument(
            "--metric_token_pad_value",
            type=int,
            default=0,
            help="The padding value for metric token",
        )
        group.add_argument(
            "--tokenize_numerical_metric",
            type=str2bool,
            default=False,
            help="Whether to tokenize numerical metrics or not",
        )
        group.add_argument(
            "--init",
            type=lambda x: str_or_none(x.lower()),
            default=None,
            help="The initialization method",
            choices=[
                "chainer",
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
                None,
            ],
        )
        group.add_argument(
            "--model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetUniversaModel),
            help="The keyword arguments for model class.",
        )
        group.add_argument(
            "--use_ref_audio",
            default=True,
            type=str2bool,
            help="Use reference wav for training or not",
        )
        group.add_argument(
            "--use_ref_text",
            default=True,
            type=str2bool,
            help="Use reference text for training or not",
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=True,
            help="Apply preprocessing to data or not",
        )
        group.add_argument(
            "--token_type",
            type=str,
            default="bpe",
            choices=["bpe", "char", "word", "phn"],
            help="The text will be tokenized in the specified level token",
        )
        group.add_argument(
            "--bpemodel",
            type=str_or_none,
            default=None,
            help="The model file of sentencepiece",
        )
        parser.add_argument(
            "--non_linguistic_symbols",
            type=str_or_none,
            help="non_linguistic_symbols file path",
        )
        parser.add_argument(
            "--cleaner",
            type=str_or_none,
            choices=[None, "tacotron", "jaconv", "vietnamese", "korean_cleaner"],
            default=None,
            help="Apply text cleaning",
        )
        parser.add_argument(
            "--g2p",
            type=str_or_none,
            choices=g2p_choices,
            default=None,
            help="Specify g2p method if --token_type=phn",
        )
        group.add_argument(
            "--sequential_metric",
            type=str2bool,
            default=False,
            help="Whether to use sequential metric or not",
        )
        group.add_argument(
            "--randomize_sequential_metric",
            type=str2bool,
            default=False,
            help="Whether to randomize sequential metric or not",
        )

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    @typechecked
    def build_collate_fn(cls, args: argparse.Namespace, train: bool) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        metrics_list = parse_metrics_meta(args.metric2id, return_list=True)

        # parse metrics2type
        if args.metric2type is None:
            numerical_metrics = metrics_list
            categorical_metrics = []
        else:
            metric2type = parse_metrics_meta(args.metric2type, return_list=False)
            if not args.tokenize_numerical_metric:
                numerical_metrics, categorical_metrics = [], []
                for metric_name in metric2type.keys():
                    if metric2type[metric_name] == "numerical":
                        numerical_metrics.append(metric_name)
                    else:
                        categorical_metrics.append(metric_name)
            else:
                categorical_metrics = metrics_list
                numerical_metrics = []
                for metric_name in metric2type.keys():
                    metric2type[metric_name] = "categorical"

        if args.sequential_metric:
            not_sequence = []
        else:
            not_sequence = ["metrics"]

        # To differentiate the padding value for metrics' value
        return UniversaCollateFn(
            numerical_metrics=numerical_metrics,
            categorical_metrics=categorical_metrics,
            sequential_metric=args.sequential_metric,
            float_pad_value=0.0,
            metric_pad_value=args.metric_pad_value,
            metric_token_pad_value=args.metric_token_pad_value,
            int_pad_value=0,
            not_sequence=not_sequence,
            randomize=args.randomize_sequential_metric,
        )

    @classmethod
    @typechecked
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:

        if args.metric2type is not None:
            metric2type = parse_metrics_meta(args.metric2type, return_list=False)

        if args.use_preprocessor:
            retval = UniversaProcessor(
                train=train,
                token_type=args.token_type,
                token_list=args.token_list,
                bpemodel=args.bpemodel,
                non_linguistic_symbols=args.non_linguistic_symbols,
                text_cleaner=args.cleaner,
                g2p_type=args.g2p,
                metric2type=metric2type,
                metric_token_info=args.metric_token_info,
                tokenize_numerical_metric=args.tokenize_numerical_metric,
                reduce_offset=not args.sequential_metric,
            )
        else:
            retval = None
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = ("metrics", "audio")
        else:
            # Inference mode
            retval = ("audio",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = (
                "ref_audio",
                "ref_text",
            )
        else:
            # Inference mode
            retval = (
                "ref_audio",
                "ref_text",
            )
        return retval

    @classmethod
    @typechecked
    def build_model(cls, args: argparse.Namespace) -> ESPnetUniversaModel:
        # Load metric2id
        metric2id = {
            m: i
            for i, m in enumerate(parse_metrics_meta(args.metric2id, return_list=True))
        }
        metric2type = (
            parse_metrics_meta(args.metric2type, return_list=False)
            if args.metric2type
            else None
        )
        if args.tokenize_numerical_metric:
            metric2type = {k: "categorical" for k, v in metric2type.items()}

        # Process metric token list
        if args.metric_token_info is not None:
            if isinstance(args.metric_token_info, str):
                with open(args.metric_token_info, "r") as f:
                    metric_token_info = json.load(f)
            else:
                raise ValueError("metric_token_info must be str or list")
            # NOTE(jiatong): related to espnet2.universa.metric_tokenizer.metric_tokenizer
            metric_vocab_size = (
                len(metric_token_info["VOCAB"]) + 4
            )  # +2 for <pad> and <unk> and +2 for <sos> and <eos>
            logging.info("Metric vocabulary size: " + str(metric_vocab_size))

        else:
            metric_vocab_size = None

        # Process text token list
        if args.token_list is not None:
            if isinstance(args.token_list, str):
                with open(args.token_list, "r") as f:
                    token_list = [line.rstrip() for line in f]

                # Overwriting token_list to keep it as "portable"
                args.token_list = list(token_list)
            elif isinstance(args.token_list, (tuple, list)):
                token_list = list(args.token_list)
            else:
                raise ValueError("token_list must be str or list")
            vocab_size = len(args.token_list)
            logging.info("Vocabulary size: " + str(vocab_size))
        else:
            if args.use_ref_text:
                raise ValueError("token_list is required when use_ref_text is True")
            token_list, vocab_size = None, None

        # 1. frontend
        # Extract source features in the model
        frontend_class = frontend_choices.get_class(args.frontend)
        frontend = frontend_class(**args.frontend_conf)
        raw_input_size = frontend.output_size()

        # 2. universa
        universa_class = universa_choices.get_class(args.universa)
        universa = universa_class(
            input_size=raw_input_size,
            metric2id=metric2id,
            vocab_size=vocab_size,
            use_ref_audio=args.use_ref_audio,
            use_ref_text=args.use_ref_text,
            metric_pad_value=args.metric_pad_value,
            metric_token_pad_value=args.metric_token_pad_value,
            metric2type=metric2type,
            metric_vocab_size=metric_vocab_size,
            metric_token_info=metric_token_info,
            sequential_metrics=args.sequential_metric,
            **args.universa_conf,
        )

        # 3. Build model
        model = ESPnetUniversaModel(
            frontend=frontend, universa=universa, **args.model_conf
        )

        # 4. Initialize
        if args.init is not None:
            initialize(model, args.init)
        return model
