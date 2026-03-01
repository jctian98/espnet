#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Stage 4: LLM-as-judge quality validation and filtering for SVS examples."""

import argparse
import asyncio
import json
import os
from typing import Any, Dict, Optional, Tuple

from sft_vllm_client import (
    DEFAULT_MODEL,
    MassiveQueryProcessor,
    get_processed_indices,
    parse_vllm_urls,
)

# Validation statistics
def _new_stats() -> Dict[str, int]:
    return {
        "success": 0,
        "no_response": 0,
        "bad_json": 0,
        "bad_structure": 0,
        "bad_score_range": 0,
    }


_stats = _new_stats()
_structure_issues: Dict[str, int] = {}

SYSTEM_PROMPT = (
    "You are a STRICT quality judge for singing voice synthesis (SVS) "
    "training data. Your task is to critically evaluate SVS training "
    "examples and identify any flaws or issues.\n\n"
    "IMPORTANT: Be critical and look for problems. Most examples should "
    "score 3-4, not 5. A score of 5 means PERFECT with zero issues — "
    "this should be rare.\n\n"
    "You must output ONLY valid JSON with the exact structure specified. "
    "No other text, explanation, or markdown formatting."
)

USER_PROMPT_TEMPLATE = """\
Critically evaluate a singing voice synthesis (SVS) training example. \
Your job is to FIND FLAWS, not to praise.

## Scoring Guidelines (BE STRICT)
For each dimension, identify flaws first, then assign a score:

- Score 5 (Exceptional): ZERO flaws. RARE — only ~10% deserve this.
- Score 4 (Good): Minor imperfections. Most good examples belong here.
- Score 3 (Acceptable): Noticeable issues but still usable for training.
- Score 2 (Poor): Significant issues that may harm training quality.
- Score 1 (Unacceptable): Critical flaws. Should not be used.

## Evaluation Dimensions

- voice_style_match: Does the caption describe the requested singing style?
  * Flaws: wrong genre/emotion/energy than requested, mismatched vocal type
    (e.g. user asks for soprano but caption describes bass)
  * If the request mentions non-singing elements (background music,
    instrumental accompaniment, intro/outro), check whether the caption
    addresses them appropriately
- lyrics_faithfulness: Are the requested lyrics accurately reflected?
  * Flaws: lyrics in caption differ substantially from those in the request,
    words omitted or changed, melody described doesn't match the passage
- structure_completeness: Are all 3 CoT parts (A, B, C) present and adequate?
  * Flaws: missing parts, empty or one-line sections, off-topic content
- reasoning_coherence: Does the SVS reasoning flow logically?
  * Flaws: contradictions, jumps in logic, vocal advice that contradicts genre
- cot_caption_consistency: Does the CoT plan align with what the caption describes?
  * Flaws: CoT mentions ornaments or dynamics not present in caption,
    contradictions between planned delivery and described performance
  * If accompaniment or non-singing elements are mentioned in the request,
    check that the CoT and caption handle them consistently

## Output Format
- If score < 5: include "flaws" field explaining the issues
- If score = 5: just include "score"

Output JSON only:
{{"voice_style_match": {{"score": 4}}, \
"lyrics_faithfulness": {{"flaws": "...", "score": 3}}, \
"structure_completeness": {{"score": 5}}, \
"reasoning_coherence": {{"flaws": "...", "score": 3}}, \
"cot_caption_consistency": {{"flaws": "...", "score": 4}}}}

## SVS Training Example to Evaluate

**User Request:**
{user_request}

**Thinking Trace (assistant's reasoning):**
{cot}

**Rich Caption (target singing audio description):**
{caption}"""

EXPECTED_DIMENSIONS = [
    "voice_style_match",
    "lyrics_faithfulness",
    "structure_completeness",
    "reasoning_coherence",
    "cot_caption_consistency",
]


def parse_judge_response(response: str) -> Optional[Dict[str, Any]]:
    """Parse JSON response from judge LLM."""
    if response is None:
        _stats["no_response"] += 1
        return None

    data = None
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
        except json.JSONDecodeError:
            pass

    if data is None:
        _stats["bad_json"] += 1
        return None

    if not _validate_response_structure(data):
        _stats["bad_structure"] += 1
        return None

    if not _validate_score_range(data):
        _stats["bad_score_range"] += 1
        return None

    _stats["success"] += 1
    return data


def _validate_response_structure(data: Dict[str, Any]) -> bool:
    for dim in EXPECTED_DIMENSIONS:
        if dim not in data:
            key = f"missing_dim:{dim}"
            _structure_issues[key] = _structure_issues.get(key, 0) + 1
            return False
        if not isinstance(data[dim], dict):
            key = f"not_dict:{dim}"
            _structure_issues[key] = _structure_issues.get(key, 0) + 1
            return False
        if "score" not in data[dim]:
            key = f"no_score:{dim}"
            _structure_issues[key] = _structure_issues.get(key, 0) + 1
            return False
        if not isinstance(data[dim]["score"], (int, float)):
            key = f"bad_type:{dim}"
            _structure_issues[key] = _structure_issues.get(key, 0) + 1
            return False
    return True


def _validate_score_range(data: Dict[str, Any]) -> bool:
    for dim in EXPECTED_DIMENSIONS:
        score = data[dim]["score"]
        if score < 1 or score > 5:
            return False
    return True


def compute_overall_score(
    scores: Dict[str, Any],
    min_threshold: float,
    avg_threshold: float,
) -> Tuple[float, float, bool]:
    """Return (avg_score, min_score, overall_pass).

    Called only after parse_judge_response validates all dimensions exist
    with numeric scores, so no need for defensive checks here.
    """
    all_scores = [scores[dim]["score"] for dim in EXPECTED_DIMENSIONS]
    avg = sum(all_scores) / len(all_scores)
    mn = min(all_scores)
    passed = mn >= min_threshold and avg >= avg_threshold
    return round(avg, 2), round(mn, 2), passed


def build_judge_query(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Build a quality-judge query for an SVS sample."""
    user_request = sample.get("user_request", "")
    cot = sample.get("cot", "")
    caption = sample.get("caption", "")
    example_id = sample.get("example_id", "")

    if not user_request or not caption:
        return None

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                user_request=user_request,
                cot=cot,
                caption=caption,
            ),
        },
    ]

    return {
        "idx": example_id,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 2048,
        "json_mode": True,
        "metadata": sample,
    }


def make_process_judge_result(
    min_threshold: float, avg_threshold: float
):
    """Create process function capturing threshold parameters."""

    def process_judge_result(
        result: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        response = result.get("response")
        metadata = result.get("metadata", {})
        example_id = metadata.get("example_id", "")

        scores = parse_judge_response(response)
        if scores is None:
            return None

        avg_score, min_score, overall_pass = compute_overall_score(
            scores, min_threshold, avg_threshold
        )

        return {
            "example_id": example_id,
            "dataset": metadata.get("dataset", ""),
            "audio_path": metadata.get("audio_path", ""),
            "duration": metadata.get("duration", 0.0),
            "caption": metadata.get("caption", ""),
            "whisper_lyrics": metadata.get("whisper_lyrics", ""),
            "extracted_lyrics": metadata.get("extracted_lyrics", ""),
            "user_request": metadata.get("user_request", ""),
            "cot": metadata.get("cot", ""),
            "scores": scores,
            "avg_score": avg_score,
            "min_score": min_score,
            "overall_pass": overall_pass,
            "idx": result["idx"],
        }

    return process_judge_result


def load_input_data(input_file: str) -> list:
    samples = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def compute_summary(results_file: str) -> Dict[str, Any]:
    """Compute aggregate statistics from judge results.

    Records in judge_results.jsonl have already been validated by
    parse_judge_response, so scores are guaranteed to be well-formed.
    """
    results = load_input_data(results_file)
    if not results:
        return {}

    total = len(results)
    passed = sum(1 for r in results if r.get("overall_pass", False))

    score_sums: Dict[str, float] = {dim: 0.0 for dim in EXPECTED_DIMENSIONS}
    score_counts: Dict[str, int] = {dim: 0 for dim in EXPECTED_DIMENSIONS}
    failure_breakdown: Dict[str, int] = {}

    for r in results:
        scores = r.get("scores", {})
        for dim in EXPECTED_DIMENSIONS:
            if dim in scores:
                s = scores[dim]["score"]
                score_sums[dim] += s
                score_counts[dim] += 1
                if s < 3:
                    failure_breakdown[dim] = (
                        failure_breakdown.get(dim, 0) + 1
                    )

    avg_scores = {
        dim: round(score_sums[dim] / score_counts[dim], 2)
        for dim in EXPECTED_DIMENSIONS
        if score_counts[dim] > 0
    }

    return {
        "total_samples": total,
        "passed_samples": passed,
        "pass_rate": round(passed / total, 4) if total > 0 else 0,
        "avg_scores": avg_scores,
        "failure_breakdown": dict(
            sorted(failure_breakdown.items(), key=lambda x: x[1], reverse=True)
        ),
    }


async def main_async(args) -> None:
    print(f"Loading input data from: {args.input_file}")
    samples = load_input_data(args.input_file)
    print(f"Loaded {len(samples)} samples")

    if args.num_samples > 0:
        samples = samples[: args.num_samples]
        print(f"Limited to {len(samples)} samples")

    os.makedirs(args.output_dir, exist_ok=True)
    judge_file = os.path.join(args.output_dir, "judge_results.jsonl")

    processed_ids = set()
    if os.path.exists(judge_file) and args.resume:
        processed_ids = get_processed_indices(judge_file, idx_key="example_id")
        print(f"Resuming: {len(processed_ids)} already processed")
    else:
        with open(judge_file, "w"):
            pass

    base_urls = parse_vllm_urls(args.vllm_url)
    model = args.model or DEFAULT_MODEL
    workers_per_server = max(1, args.num_workers // len(base_urls))

    global _stats
    _stats = _new_stats()
    _structure_issues.clear()

    queries = []
    for sample in samples:
        eid = sample.get("example_id", "")
        if eid in processed_ids:
            continue
        query = build_judge_query(sample)
        if query is not None:
            queries.append(query)

    print(f"Queries to process: {len(queries)}")

    if not queries:
        print("All samples already processed!")
    else:
        processor = MassiveQueryProcessor(
            base_urls=base_urls,
            model=model,
            workers_per_server=workers_per_server,
            timeout=args.timeout,
            checkpoint_interval=10000,
        )

        process_fn = make_process_judge_result(args.min_score, args.avg_score)
        await processor.process_all(
            queries=queries,
            output_file=judge_file,
            process_fn=process_fn,
        )

    # Filter passing examples
    print("\nFiltering passing examples...")
    filtered_file = os.path.join(args.output_dir, "filtered.jsonl")
    passed_count = 0
    failed_count = 0
    bottleneck_counts: Dict[str, int] = {}
    below_threshold_counts: Dict[str, int] = {}

    judge_only_keys = {"scores", "avg_score", "min_score", "overall_pass", "idx"}

    with open(filtered_file, "w", encoding="utf-8") as out_f:
        with open(judge_file, "r", encoding="utf-8") as in_f:
            for line in in_f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("overall_pass", False):
                    sample = {
                        k: v
                        for k, v in record.items()
                        if k not in judge_only_keys
                    }
                    out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    passed_count += 1
                else:
                    failed_count += 1
                    scores = record.get("scores", {})
                    worst_score = float("inf")
                    worst_dim = ""
                    for dim in EXPECTED_DIMENSIONS:
                        val = scores.get(dim, {})
                        if isinstance(val, dict) and "score" in val:
                            s = val["score"]
                            if s < args.min_score:
                                below_threshold_counts[dim] = (
                                    below_threshold_counts.get(dim, 0) + 1
                                )
                            if s < worst_score:
                                worst_score = s
                                worst_dim = dim
                    if worst_dim:
                        bottleneck_counts[worst_dim] = (
                            bottleneck_counts.get(worst_dim, 0) + 1
                        )

    summary = compute_summary(judge_file)
    summary["filtered_passed"] = passed_count
    summary["filtered_failed"] = failed_count
    summary_file = os.path.join(args.output_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    final_count = len(
        get_processed_indices(judge_file, idx_key="example_id")
    )
    print(f"\nDone! Total judged : {final_count}/{len(samples)}")
    print(f"Passed filter      : {passed_count}")
    print(f"Failed filter      : {failed_count}")
    print(f"Pass rate          : {summary.get('pass_rate', 0):.1%}")
    print(f"Filtered output    : {filtered_file}")
    print(f"Summary            : {summary_file}")

    if failed_count > 0:
        print("\n" + "=" * 60)
        print(f"Filter Failure Analysis ({failed_count} failed):")
        print(f"\n  Bottleneck dimension (worst score per failed):")
        print(f"  {'Dimension':<40} {'Count':>8} {'Percent':>8}")
        print("  " + "-" * 56)
        for dim, count in sorted(
            bottleneck_counts.items(), key=lambda x: x[1], reverse=True
        ):
            pct = count / failed_count * 100
            print(f"  {dim:<40} {count:>8} {pct:>7.1f}%")

        print(f"\n  Dimensions below min_score ({args.min_score}):")
        print(f"  {'Dimension':<40} {'Count':>8} {'Percent':>8}")
        print("  " + "-" * 56)
        for dim, count in sorted(
            below_threshold_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            pct = count / failed_count * 100
            print(f"  {dim:<40} {count:>8} {pct:>7.1f}%")
        print("=" * 60)

    total_validated = sum(_stats.values())
    if total_validated > 0:
        print("\n" + "=" * 60)
        print("Parse Validation Summary:")
        print(f"  {'Reason':<25} {'Count':>8} {'Percent':>8}")
        print("-" * 60)
        for reason, count in _stats.items():
            pct = count / total_validated * 100
            print(f"  {reason:<25} {count:>8} {pct:>7.1f}%")
        print("-" * 60)
        print(f"  {'Total':<25} {total_validated:>8}")
        print("=" * 60)

    if _structure_issues:
        sorted_issues = sorted(
            _structure_issues.items(), key=lambda x: x[1], reverse=True
        )
        print("\n" + "=" * 60)
        print("Structure Issue Breakdown:")
        print(f"  {'Issue':<50} {'Count':>8}")
        print("-" * 60)
        for issue, count in sorted_issues:
            print(f"  {issue:<50} {count:>8}")
        print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 4: LLM-as-judge quality validation for SVS examples."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Stage 3 CoT JSONL.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--vllm_url",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM API base URL(s).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Text LLM model name.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=256,
        help="Concurrent API requests.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Request timeout in seconds.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Number of samples (-1 for all).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing progress.",
    )
    parser.add_argument(
        "--min_score",
        type=float,
        default=3.0,
        help="Minimum score per dimension (default: 3).",
    )
    parser.add_argument(
        "--avg_score",
        type=float,
        default=3.5,
        help="Minimum average score (default: 3.5).",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="Prompt version (for consistency; reserved for future use).",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
