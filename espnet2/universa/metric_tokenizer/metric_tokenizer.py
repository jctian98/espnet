import json
from abc import ABC, abstractmethod
from typing import Iterable, Dict, Union, List, Tuple
from typeguard import typechecked


class AbsMetricTokenizer(ABC):
    @abstractmethod
    def metric2token(
        self, metrics: Dict[str, Union[float, str, int]]
    ) -> List[Tuple[int, int]]:
        raise NotImplementedError

    @abstractmethod
    def token2metric(self, tokens: Iterable[int]) -> str:
        raise NotImplementedError


class MetricTokenizer(AbsMetricTokenizer):
    def __init__(self, tokenizer_config_path: str, tokenize_metric: List[str]):
        """
        Initialize the MetricTokenizer with a configuration file path.

        Args:
            tokenizer_config_path: Path to the tokenizer configuration JSON file
            tokenize_metric: List of metric names to be tokenized
        """
        with open(tokenizer_config_path, "r") as f:
            config = json.load(f)

        self.tokenizer_config = config["tokenizer"]
        self.vocab = config["VOCAB"]
        self.tokenize_metric = tokenize_metric

        # Build inverse mapping for faster lookups
        self.vocab_indices = {token: idx + 2 for idx, token in enumerate(self.vocab)}
        self.vocab_indices["<pad>"] = 0  # Add padding token index
        self.vocab_indices["<unk>"] = 1  # Add unknown token index

        # Extract metric types and their thresholds/categories
        self.metrics = {}
        for metric_name, values in self.tokenizer_config.items():
            self.metrics[metric_name] = values

    def get_token_index(self, metric_name: str, value_index: int) -> Tuple[int, int]:
        """
        Get the token indices for a metric and its value.

        Args:
            metric_name: Name of the metric
            value_index: Index of the value for this metric

        Returns:
            Tuple of (meta_label_index, value_index) in the vocabulary
        """
        meta_label_token = f"{metric_name}_meta_label"
        value_token = f"{metric_name}_{value_index}"

        meta_label_index = self.vocab_indices.get(meta_label_token)
        value_index = self.vocab_indices.get(value_token)

        if meta_label_index is None or value_index is None:
            raise ValueError(f"Invalid token: {meta_label_token} or {value_token}")

        return meta_label_index, value_index

    def _get_value_index(self, metric_name: str, value: Union[float, str, int]) -> int:
        """
        Determine the appropriate token index for a given metric value.

        Args:
            metric_name: Name of the metric
            value: The actual value of the metric

        Returns:
            Index of the appropriate value token
        """
        if metric_name == "category":
            # For categorical values, find its index in the list
            try:
                return self.tokenizer_config[metric_name].index(value)
            except ValueError:
                raise ValueError(
                    f"Invalid category value: {value} for metric {metric_name}"
                )
        else:
            # For numerical metrics with thresholds
            thresholds = self.tokenizer_config[metric_name]
            # Find the first threshold that the value is less than
            for i, threshold in enumerate(thresholds):
                if value < threshold:
                    return i
            # If value is greater than all thresholds, return the last index
            return len(thresholds) - 1

    @typechecked
    def metric2token(
        self, metrics: Dict[str, Union[float, str, int, Tuple[int, int]]]
    ) -> Dict[str, Tuple[int, int]]:
        """
        Convert metrics dictionary to token indices.

        Args:
            metrics: Dictionary of metric names and their values
            return_dict: If True, return a dictionary of token indices instead of a list

        Returns:
            Dictionary of token indices for each metric (meta_label, value)
        """
        token_indices = {}

        for metric_name, value in metrics.items():
            if metric_name not in self.metrics:
                raise ValueError(f"Unknown metric: {metric_name}")

            if (
                self.tokenize_metric is not None
                and metric_name not in self.tokenize_metric
            ):
                continue

            if type(value) == tuple:
                # already tokenized
                token_indices[metric_name] = value
                continue

            # Get the index of the value in the corresponding metric's range
            value_index = self._get_value_index(metric_name, value)

            # Get the token indices for this metric and value
            token_pair = self.get_token_index(metric_name, value_index)
            token_indices[metric_name] = token_pair

        return token_indices

    @typechecked
    def token2metric(
        self, tokens: Iterable[int], return_dict: bool = False
    ) -> Union[str, Dict[str, Union[float, int, str, Tuple[float, float]]]]:
        """
        Convert token indices back to a metric representation.

        Args:
            tokens: Iterable of token indices
            return_dict: If True, return a dictionary instead of a string

        Returns:
            String representation of the metrics
        """
        tokens_list = list(tokens)
        if len(tokens_list) % 2 != 0:
            raise ValueError(
                "Expected an even number of tokens (meta_label, value pairs)"
            )

        result = {}
        for i in range(0, len(tokens_list), 2):
            meta_label_idx = tokens_list[i]
            value_idx = tokens_list[i + 1]

            meta_label = self.vocab[meta_label_idx]
            value_token = self.vocab[value_idx]

            assert meta_label.endswith(
                "_meta_label"
            ), f"Invalid meta_label: {meta_label}"
            # Extract metric name from meta_label
            metric_name = meta_label.split("_meta_label")[0]

            # Extract value index from value token
            value_index = int(value_token.split("_")[-1])

            if metric_name not in self.tokenizer_config.keys():
                raise ValueError(f"Unknown metric in decoding: {metric_name}")

            # For category, get the actual category value
            if metric_name == "category":
                result[metric_name] = self.tokenizer_config[metric_name][value_index]
            else:
                # For numerical metrics, represent as ranges
                thresholds = self.tokenizer_config[metric_name]
                if value_index == 0:
                    result[metric_name] = f"<{thresholds[0]}"
                elif value_index == len(thresholds):
                    result[metric_name] = f">={thresholds[-1]}"
                else:
                    result[metric_name] = (
                        f"[{thresholds[value_index-1]}, {thresholds[value_index]})"
                    )

        if return_dict:
            return result
        # Format the result as a string
        formatted_result = ", ".join([f"{k}: {v}" for k, v in result.items()])
        return formatted_result
