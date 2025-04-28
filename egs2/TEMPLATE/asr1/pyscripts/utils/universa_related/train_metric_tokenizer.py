#!/usr/bin/env python3

import argparse
import json
import numpy as np
import os
from typing import Dict, List, Any, Union


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a metric tokenizer")
    parser.add_argument(
        "--input", required=True, type=str, help="Path to the metric.scp jsonl file"
    )
    parser.add_argument(
        "--metric2type", required=True, type=str, 
        help="Path to file mapping metrics to their types (numerical/categorical)"
    )
    parser.add_argument(
        "--metric2id", required=True, type=str,
        help="Path to file mapping metrics to their IDs (for ordering purposes)"
    )
    parser.add_argument(
        "--tokenizer_info", required=True, type=str, help="Path to output token list JSON file"
    )
    parser.add_argument(
        "--token_size", required=True, type=int, 
        help="Default number of tokens/intervals for numerical metrics"
    )
    parser.add_argument(
        "--metric2token_size", type=str, default=None,
        help="Optional path to file mapping metrics to their individual token sizes"
    )
    parser.add_argument(
        "--token_method", required=True, type=str, choices=["percentile"],
        help="Method for tokenization (currently only 'percentile' is supported)"
    )
    parser.add_argument(
        "--percentile_distribution", type=str, default="linear",
        choices=["linear", "exponential", "logarithmic", "normal", "quadratic"],
        help="Default distribution strategy for percentiles (default: linear)"
    )
    parser.add_argument(
        "--metric2percentile_distribution", type=str, default=None,
        help="Optional path to file mapping metrics to their individual percentile distributions"
    )
    parser.add_argument(
        "--categorical_only", action="store_true",
        help="If set, only process categorical metrics and ignore numerical ones"
    )
    return parser.parse_args()


def load_metric2type(filepath: str) -> Dict[str, str]:
    """Load metric to type mapping file."""
    metric2type = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Invalid line in metric2type file: {line}")
            metric, metric_type = parts
            if metric_type not in ["numerical", "categorical"]:
                raise ValueError(f"Invalid metric type: {metric_type}, must be 'numerical' or 'categorical'")
            metric2type[metric] = metric_type
    return metric2type



def load_metric2id(filepath: str) -> Dict[str, int]:
    """Load metric to ID mapping file.
    The ID for each metric is determined by its line number in the file (0-indexed).
    """
    metric2id = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            metric = line.strip()
            if not metric:
                continue
            metric2id[metric] = i
    return metric2id



def load_metric2token_size(filepath: str) -> Dict[str, int]:
    """Load metric to token size mapping file."""
    metric2token_size = {}
    
    if not filepath:
        return metric2token_size
        
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Invalid line in metric2token_size file: {line}")
            
            metric, token_size = parts
            try:
                token_size = int(token_size)
                if token_size <= 0:
                    raise ValueError(f"Token size must be a positive integer: {token_size}")
                metric2token_size[metric] = token_size
            except ValueError:
                raise ValueError(f"Invalid token size: {token_size}, must be a positive integer")
    
    return metric2token_size


def load_metric2percentile_distribution(filepath: str) -> Dict[str, str]:
    """Load metric to percentile distribution mapping file."""
    metric2percentile_distribution = {}
    
    valid_distributions = ["linear", "exponential", "logarithmic", "normal", "quadratic"]
    
    if not filepath:
        return metric2percentile_distribution
        
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Invalid line in metric2percentile_distribution file: {line}")
            
            metric, distribution = parts
            if distribution not in valid_distributions:
                raise ValueError(f"Invalid distribution: {distribution}. Valid options are: {', '.join(valid_distributions)}")
            
            metric2percentile_distribution[metric] = distribution
    
    return metric2percentile_distribution


def collect_metric_values(input_file: str, metric2type: Dict[str, str]) -> Dict[str, List[Any]]:
    """Collect all values for each metric in the input file."""
    metric_values = {metric: [] for metric in metric2type}
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line = line.split(maxsplit=1)[1]  # Skip the key part
            
            try:
                data = json.loads(line)
                for metric, value in data.items():
                    if metric in metric2type:
                        metric_values[metric].append(value)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line as JSON: {line}")
                continue
    
    return metric_values


def generate_percentiles(
    token_size: int, 
    distribution: str = "linear", 
) -> np.ndarray:
    """Generate percentile points based on the specified distribution."""
    if distribution == "linear":
        # Evenly spaced percentiles
        return np.linspace(0, 100, token_size - 1)
    
    elif distribution == "exponential":
        # More intervals at higher values
        x = np.linspace(0, 5, token_size - 1)  # Using 0 to 5 for exponential range
        percentiles = 100 * (np.exp(x) - 1) / (np.exp(5) - 1)
        return percentiles
    
    elif distribution == "logarithmic":
        # More intervals at lower values
        x = np.linspace(0, 5, token_size - 1)  # Using 0 to 5 for logarithmic range
        percentiles = 100 * np.log(x + 1) / np.log(6)
        # Replace any invalid values (from log(0+1))
        percentiles[0] = 0
        return percentiles
    
    elif distribution == "normal":
        # More intervals around the middle (50th percentile)
        # Using normal distribution CDF transformed to 0-100 range
        x = np.linspace(-2.5, 2.5, token_size - 1)  # -2.5 to 2.5 standard deviations
        from scipy.stats import norm
        percentiles = 100 * norm.cdf(x)
        # Ensure endpoints are exactly 0 and 100
        percentiles[0] = 0
        percentiles[-1] = 100
        return percentiles
    
    elif distribution == "quadratic":
        # More intervals at both extremes
        x = np.linspace(-1, 1, token_size - 1)
        # Quadratic transformation to focus on extremes
        percentiles = 50 * (1 + np.sign(x) * x**2)
        return percentiles
    
    # Default to linear if no valid option is specified
    return np.linspace(0, 100, token_size - 1)


def create_tokenizer(
    metric_values: Dict[str, List[Any]], 
    metric2type: Dict[str, str], 
    metric2id: Dict[str, int],
    default_token_size: int,
    metric2token_size: Dict[str, int],
    token_method: str,
    default_percentile_distribution: str = "linear",
    metric2percentile_distribution: Dict[str, str] = None,
    categorical_only: bool = False,
) -> Dict[str, List[Union[float, str]]]:
    """Create a tokenizer based on collected metric values."""
    tokenizer = {}
    vocab = []
    
    # Initialize metric2percentile_distribution if None
    if metric2percentile_distribution is None:
        metric2percentile_distribution = {}
    
    # Sort metrics by their IDs from metric2id (for consistent ordering)
    sorted_metrics = sorted(
        [(metric, metric2id.get(metric, float('inf'))) for metric in metric_values.keys()],
        key=lambda x: x[1]
    )
    
    for metric, _ in sorted_metrics:
        vocab.append("{}_meta_label".format( metric))
        values = metric_values[metric]
        metric_type = metric2type[metric]
        
        # Get token size for this metric (use default if not specified)
        token_size = metric2token_size.get(metric, default_token_size)

        assert token_size > 1, (
            f"Token size must be greater than 1 for metric '{metric}', got {token_size}"
        )
        
        # Get percentile distribution for this metric (use default if not specified)
        percentile_distribution = metric2percentile_distribution.get(
            metric, default_percentile_distribution
        )
        
        if metric_type == "categorical":
            # For categorical metrics, simply get unique values
            unique_values = sorted(set(values))
            tokenizer[metric] = unique_values

            # Create category tokens for VOCAB field
            for idx, _ in enumerate(unique_values):
                vocab.append(f"{metric}_{idx}")

            print(f"Metric '{metric}' (categorical): found {len(unique_values)} unique categories")
            
        elif metric_type == "numerical" and not categorical_only:
            # Convert to numeric values, ignoring non-numeric
            numeric_values = []
            for v in values:
                try:
                    numeric_values.append(float(v))
                except (ValueError, TypeError):
                    print(f"Warning: Non-numeric value '{v}' for metric '{metric}' will be ignored")
            
            if not numeric_values:
                print(f"Warning: No valid numeric values for metric '{metric}'")
                tokenizer[metric] = []
                continue
                
            if token_method == "percentile":
                # Generate percentiles based on the specified distribution
                percentiles = generate_percentiles(
                    token_size, percentile_distribution
                )

                assert len(percentiles) == token_size - 1, (
                    f"Expected {token_size - 1} percentiles, got {len(percentiles)} for metric '{metric}'"
                )
                assert len(percentiles) <= len(numeric_values), (
                    f"More intervals than data points for metric '{metric}': {len(percentiles)} vs {len(numeric_values)}"
                )

                # Calculate the actual intervals based on the data
                intervals = np.percentile(numeric_values, percentiles)

                tokenizer[metric] = intervals.tolist()

                # Create interval tokens for VOCAB field (one less than intervals)
                for idx in range(len(intervals) + 1):
                    vocab.append(f"{metric}_{idx}")

                print(f"Metric '{metric}' (numerical): created {len(intervals)-1} intervals "
                      f"using {token_size} tokens with '{percentile_distribution}' distribution")

    return tokenizer, vocab


def main():
    args = parse_args()
    
    # Load metric to type mapping
    metric2type = load_metric2type(args.metric2type)

    # Load metric to ID mapping for ordering
    metric2id = load_metric2id(args.metric2id)
    print(f"Loaded {len(metric2id)} metric IDs for ordering")

    # Load metric to token size mapping (if provided)
    metric2token_size = {}
    if args.metric2token_size:
        metric2token_size = load_metric2token_size(args.metric2token_size)
        print(f"Loaded individual token sizes for {len(metric2token_size)} metrics")
    
    # Load metric to percentile distribution mapping (if provided)
    metric2percentile_distribution = {}
    if args.metric2percentile_distribution:
        metric2percentile_distribution = load_metric2percentile_distribution(args.metric2percentile_distribution)
        print(f"Loaded individual percentile distributions for {len(metric2percentile_distribution)} metrics")
    
    # Collect all metric values from the input file
    metric_values = collect_metric_values(args.input, metric2type)
    
    # Create tokenizer and vocabulary
    tokenizer, vocab = create_tokenizer(
        metric_values, 
        metric2type, 
        metric2id,
        args.token_size,
        metric2token_size,
        args.token_method,
        args.percentile_distribution,
        metric2percentile_distribution,
        args.categorical_only,
    )
    
    # Create the final output with both tokenizer and vocabulary
    output = {
        "tokenizer": tokenizer,
        "VOCAB": vocab
    }
    
    with open(args.tokenizer_info, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)
    
    print(f"Tokenizer with {len(vocab)} vocabulary tokens saved to {args.tokenizer_info}")


if __name__ == "__main__":
    main()