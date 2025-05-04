#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from typing import List, Set
from espnet2.tasks.universa import parse_metrics_meta
# Create confusion matrix
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import scipy.stats
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np
import scipy


def get_parser():
    parser = argparse.ArgumentParser(description="Universal evaluation script")
    parser.add_argument(
        "--level",
        type=str,
        default="utt",
        choices=["utt", "sys"],
    )
    parser.add_argument(
        "--ref_metrics",
        type=str,
        required=True,
        help="reference metrics file",
    )
    parser.add_argument(
        "--pred_metrics",
        type=str,
        required=True,
        help="metrics prediction file",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        required=True,
        help="output file",
    )
    parser.add_argument(
        "--sys_info",
        type=str,
        default=None,
        help="system information file",
    )
    parser.add_argument(
        "--skip_missing",
        type=bool,
        default=False,
        help="skip missing utterances",
    )
    parser.add_argument(
        "--metric2type",
        type=str,
        default=None,
        help="metric2type information",
    )
    return parser


def calculate_regression_metrics(ref_metric_scores, pred_metric_scores, prefix="utt"):
    """Calculate comprehensive metrics for numerical predictions/scores.
    
    Args:
        ref_metric_scores: List/array of reference/ground truth scores (numerical)
        pred_metric_scores: List/array of predicted scores (numerical)
        prefix: Prefix for the metric names in the output dictionary
        
    Returns:
        Dictionary containing regression evaluation metrics
    """
    if len(ref_metric_scores) != len(pred_metric_scores):
        raise ValueError(
            f"Number of samples mismatch: {len(ref_metric_scores)} != {len(pred_metric_scores)}"
        )
    
    # Convert inputs to numpy arrays if they aren't already
    ref_metric_scores = np.array(ref_metric_scores, dtype=float)
    pred_metric_scores = np.array(pred_metric_scores, dtype=float)
    
    # Check for NaN values
    if np.isnan(ref_metric_scores).any():
        raise ValueError("Input reference arrays contain NaN values")
    if np.isnan(pred_metric_scores).any():
        raise ValueError("Input prediction arrays contain NaN values")
    
    # Basic error metrics
    mse = mean_squared_error(ref_metric_scores, pred_metric_scores)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(ref_metric_scores, pred_metric_scores)
    
    # Correlation metrics
    # Handle potential errors with correlation calculations
    try:
        lcc = np.corrcoef(ref_metric_scores, pred_metric_scores)[0, 1]
    except:
        lcc = np.nan
    
    try:
        srcc, srcc_pvalue = scipy.stats.spearmanr(ref_metric_scores, pred_metric_scores)
    except:
        srcc = np.nan
        srcc_pvalue = np.nan
    
    try:
        ktau, ktau_pvalue = scipy.stats.kendalltau(ref_metric_scores, pred_metric_scores)
    except:
        ktau = np.nan
        ktau_pvalue = np.nan
    
    # Goodness of fit
    r2 = r2_score(ref_metric_scores, pred_metric_scores)
    
    # Calculate min, max, mean errors
    abs_errors = np.abs(ref_metric_scores - pred_metric_scores)
    min_abs_error = np.min(abs_errors)
    max_abs_error = np.max(abs_errors)
    mean_abs_error = np.mean(abs_errors)
    std_abs_error = np.std(abs_errors)
    
    return {
        f"{prefix}_mse": mse,
        f"{prefix}_rmse": rmse,
        f"{prefix}_mae": mae,
        f"{prefix}_lcc": lcc,
        f"{prefix}_srcc": srcc,
        f"{prefix}_srcc_pvalue": srcc_pvalue,
        f"{prefix}_ktau": ktau,
        f"{prefix}_ktau_pvalue": ktau_pvalue,
        f"{prefix}_r2": r2,
        f"{prefix}_min_abs_error": min_abs_error,
        f"{prefix}_max_abs_error": max_abs_error,
        f"{prefix}_mean_abs_error": mean_abs_error,
        f"{prefix}_std_abs_error": std_abs_error,
    }


def calculate_classification_metrics(ref_classes, pred_classes, prefix="cls"):
    """Calculate classification-level metrics for string labels.
    
    Args:
        ref_classes: List of reference/ground truth class labels (strings)
        pred_classes: List of predicted class labels (strings)
        prefix: Prefix for the metric names in the output dictionary
        
    Returns:
        Dictionary containing classification metrics
    """
    if len(ref_classes) != len(pred_classes):
        raise ValueError(
            f"Number of samples mismatch: {len(ref_classes)} != {len(pred_classes)}"
        )
    
    # Accuracy
    correct = sum(1 for r, p in zip(ref_classes, pred_classes) if r == p)
    accuracy = correct / len(ref_classes)
    
    # Get unique classes from both lists
    unique_classes = sorted(set(ref_classes + pred_classes))
    
    # cm = confusion_matrix(ref_classes, pred_classes, labels=unique_classes)
    
    # Calculate precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(
        ref_classes, pred_classes, average='macro', labels=unique_classes, zero_division=0
    )
    
    # Calculate weighted versions
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        ref_classes, pred_classes, average='weighted', labels=unique_classes, zero_division=0
    )
    
    return {
        f"{prefix}_accuracy": accuracy,
        f"{prefix}_precision_macro": precision,
        f"{prefix}_recall_macro": recall,
        f"{prefix}_f1_macro": f1,
        f"{prefix}_precision_weighted": precision_weighted,
        f"{prefix}_recall_weighted": recall_weighted,
        f"{prefix}_f1_weighted": f1_weighted,
        # f"{prefix}_confusion_matrix": cm,
        # f"{prefix}_classes": unique_classes
    }


def calculate_system_classification_metrics(ref_metric, pred_metric, metric, prefix="sys"):
    """Calculate system-level metrics for classification tasks.
    
    Args:
        ref_metric: Dictionary mapping system IDs to lists of reference class labels
        pred_metric: Dictionary mapping system IDs to lists of predicted class labels
        metric: The name of the metric being evaluated
        prefix: Prefix for the metric names in the output dictionary
        
    Returns:
        Dictionary containing system-level classification metrics
    """
    # Collect all reference and prediction classes across systems
    all_ref_classes = []
    all_pred_classes = []
    
    # Create system-level confusion matrices
    system_accuracies = []
    system_f1_scores = []
    system_precisions = []
    system_recalls = []
    
    # Process each system
    for sys_id in pred_metric.keys():
        sys_pred_classes = pred_metric[sys_id]
        sys_ref_classes = ref_metric[sys_id]
        
        # Add to the overall collection
        all_ref_classes.extend(sys_ref_classes)
        all_pred_classes.extend(sys_pred_classes)
        
        # Calculate per-system metrics
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        # System accuracy
        sys_accuracy = accuracy_score(sys_ref_classes, sys_pred_classes)
        system_accuracies.append(sys_accuracy)
        
        # System F1 (macro)
        try:
            sys_f1 = f1_score(sys_ref_classes, sys_pred_classes, average='macro', zero_division=0)
            system_f1_scores.append(sys_f1)
        except:
            system_f1_scores.append(np.nan)
            
        # System precision & recall
        try:
            sys_precision = precision_score(sys_ref_classes, sys_pred_classes, average='macro', zero_division=0)
            system_precisions.append(sys_precision)
        except:
            system_precisions.append(np.nan)
            
        try:
            sys_recall = recall_score(sys_ref_classes, sys_pred_classes, average='macro', zero_division=0)
            system_recalls.append(sys_recall)
        except:
            system_recalls.append(np.nan)
    
    # Overall metrics across all systems
    overall_accuracy = accuracy_score(all_ref_classes, all_pred_classes)
    
    try:
        overall_f1 = f1_score(all_ref_classes, all_pred_classes, average='macro', zero_division=0)
        overall_precision = precision_score(all_ref_classes, all_pred_classes, average='macro', zero_division=0)
        overall_recall = recall_score(all_ref_classes, all_pred_classes, average='macro', zero_division=0)
    except:
        overall_f1 = np.nan
        overall_precision = np.nan
        overall_recall = np.nan
    
    # Get unique classes
    unique_classes = sorted(set(all_ref_classes + all_pred_classes))
    
    # Create overall confusion matrix
    # try:
    #     cm = confusion_matrix(all_ref_classes, all_pred_classes, labels=unique_classes)
    # except:
    #     cm = None
    
    # Calculate mean and std of system-level metrics
    mean_system_accuracy = np.mean(system_accuracies)
    std_system_accuracy = np.std(system_accuracies)
    
    mean_system_f1 = np.mean(system_f1_scores)
    std_system_f1 = np.std(system_f1_scores)
    
    mean_system_precision = np.mean(system_precisions)
    std_system_precision = np.std(system_precisions)
    
    mean_system_recall = np.mean(system_recalls)
    std_system_recall = np.std(system_recalls)
    
    return {
        f"{prefix}_{metric}_overall_accuracy": overall_accuracy,
        f"{prefix}_{metric}_overall_f1": overall_f1,
        f"{prefix}_{metric}_overall_precision": overall_precision,
        f"{prefix}_{metric}_overall_recall": overall_recall,
        f"{prefix}_{metric}_mean_system_accuracy": mean_system_accuracy,
        f"{prefix}_{metric}_std_system_accuracy": std_system_accuracy,
        f"{prefix}_{metric}_mean_system_f1": mean_system_f1,
        f"{prefix}_{metric}_std_system_f1": std_system_f1,
        f"{prefix}_{metric}_mean_system_precision": mean_system_precision,
        f"{prefix}_{metric}_std_system_precision": std_system_precision,
        f"{prefix}_{metric}_mean_system_recall": mean_system_recall,
        f"{prefix}_{metric}_std_system_recall": std_system_recall,
        # f"{prefix}_{metric}_confusion_matrix": cm,
        # f"{prefix}_{metric}_classes": unique_classes
    }


def load_sys_info(sys_info_file: str):
    utt2sys = {}
    with open(sys_info_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            utt2sys[parts[0]] = parts[1]
    return utt2sys


def load_metrics(metrics_file, detect_metric_names=False):
    utt2metrics = {}
    metric_names = set()
    with open(metrics_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Invalid line: {line}")
            utt, metrics = parts
            try:
                # utt2metrics[utt] = json.loads(metrics.replace("'", '"').replace("inf", "0"))
                utt2metrics[utt] = json.loads(metrics)
            except:
                raise ValueError("original line: {}, {}".format(line, metrics.replace("'", '"')))
            if detect_metric_names:
                metric_names = set(utt2metrics[utt].keys())
                metric_names.update(metric_names)

    return utt2metrics, metric_names


if __name__ == "__main__":
    args = get_parser().parse_args()
    ref_metrics, ref_metric_names = load_metrics(
        args.ref_metrics, detect_metric_names=True
    )
    pred_metrics, metric_names = load_metrics(
        args.pred_metrics, detect_metric_names=True
    )
    if args.metric2type is None:
        metric2type = {metric_name: "numerical" for metric_name in metric_names}
    else:
        metric2type = parse_metrics_meta(args.metric2type)
 
    sys_info = load_sys_info(args.sys_info) if args.sys_info else None
    assert (
        sys_info is not None or args.level == "utt"
    ), "System information is required for system-level evaluation"
    final_result = {}
    # for metric in metric_names:
    for metric in ["pysepm_fwsegsnr"]:
        metric_count = {
            "miss_all": 0,
            "miss_part_ref": 0,
            "miss_part_pred": 0,
            "match": 0,
        }
        if metric not in ref_metric_names:
            metric_count["miss_all"] += 1
        if args.level == "utt":
            pred_metric, ref_metric = [], []
        else:
            pred_metric, ref_metric = {}, {}
        for utt in pred_metrics.keys():
            # Checks for missing utterances and metrics
            if utt not in ref_metrics.keys():
                metric_count["miss_part_ref"] += 1
            if metric not in pred_metrics[utt]:
                if args.skip_missing:
                    metric_count["miss_part_pred"] += 1
                    continue
                raise ValueError(f"Missing metric: {metric} in prediction metric.scp")
            if metric not in ref_metrics[utt]:
                if args.skip_missing:
                    metric_count["miss_part_ref"] += 1
                    continue
                raise ValueError(f"Missing metric: {metric} in reference metric.scp")
            if args.level == "utt":
                pred_metric.append(pred_metrics[utt][metric])
                ref_metric.append(ref_metrics[utt][metric])
            else:
                sys_id = sys_info[utt]
                if sys_id not in pred_metric:
                    pred_metric[sys_id] = []
                    ref_metric[sys_id] = []
                pred_metric[sys_id].append(pred_metrics[utt][metric])
                ref_metric[sys_id].append(ref_metrics[utt][metric])

        if args.level == "utt":
            if metric2type[metric] == "numerical":
                print(ref_metric, "--" * 100, flush=True)
                print(pred_metric, "--" * 100, flush=True)
                eval_results = calculate_regression_metrics(
                    ref_metric, pred_metric, prefix="utt_{}".format(metric)
                )
            else:
                eval_results = calculate_classification_metrics(
                    ref_metric, pred_metric, prefix="utt_{}".format(metric)
                )
        else:
            if metric2type[metric] == "numerical":
                pred_sys_avg = []
                ref_sys_avg = []
                for sys_id in pred_metric.keys():
                    sys_pred_metrics = np.array(pred_metric[sys_id])
                    sys_ref_metrics = np.array(ref_metric[sys_id])
                    sys_pred_avg = np.mean(sys_pred_metrics)
                    sys_ref_avg = np.mean(sys_ref_metrics)
                    pred_sys_avg.append(sys_pred_avg)
                    ref_sys_avg.append(sys_ref_avg)
                eval_results = calculate_regression_metrics(
                    ref_sys_avg, pred_sys_avg, prefix="sys_{}".format(metric)
                )
            elif metric2type[metric] == "classification":
                eval_results = calculate_system_classification_metrics(
                    ref_metric, pred_metric, metric, prefix="sys"
                )

        final_result.update(eval_results)
    for key in final_result.keys():
        final_result[key] = float(final_result[key])
    with open(args.out_file, "w") as f:
        json.dump(final_result, f, indent=4)
    logging.info(f"Results saved to {args.out_file}")

# Example usage:
# python universa_eval.py --level utt --ref_metrics ref_metrics.scp --pred_metrics pred_metrics.scp --out_file result.json
