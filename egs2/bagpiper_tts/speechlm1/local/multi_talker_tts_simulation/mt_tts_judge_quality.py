#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Stage 4: LLM-as-judge quality validation for multi-talker TTS examples."""

import argparse
import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from sft_vllm_client import (
    DEFAULT_MODEL,
    MassiveQueryProcessor,
    get_processed_indices,
    parse_vllm_urls,
)

# Validation statistics (reset per run in main_async)
_stats = {
    "success": 0,
    "no_response": 0,
    "bad_json": 0,
    "bad_structure": 0,
    "bad_score_range": 0,
}
_structure_issues: Dict[str, int] = {}

SYSTEM_PROMPT = (
    "You are a STRICT quality judge for text-to-speech training data. "
    "Your task is to critically evaluate training examples and identify "
    "any flaws or issues.\n\n"
    "IMPORTANT: Be critical and look for problems. Most examples should "
    "score 3-4, not 5. A score of 5 means PERFECT with zero issues - "
    "this should be rare.\n\n"
    "You must output ONLY valid JSON with the exact structure specified. "
    "No other text, explanation, or markdown formatting."
)

USER_PROMPT_TEMPLATE = """\
Critically evaluate a multi-talker text-to-speech training example. \
Your job is to FIND FLAWS and issues, not to praise the example.

## Scoring Guidelines (BE STRICT)
For each dimension, first identify any flaws/issues, then assign a score:

- Score 5 (Exceptional): ZERO flaws. Perfect. RARE - only ~10% deserve this.
- Score 4 (Good): Minor imperfections. Most good examples should be here.
- Score 3 (Acceptable): Noticeable issues but still usable for training.
- Score 2 (Poor): Significant issues that may harm training quality.
- Score 1 (Unacceptable): Critical flaws. Should not be used.

## Evaluation Dimensions

- voice_attribute_match: Does the caption describe EACH speaker's voice as \
requested? Check per-speaker alignment.
  * Flaws: wrong number of speakers, gender/emotion/accent mismatch for a \
specific speaker, speakers not clearly differentiated
- transcription_attribution: Is each speaker's transcription correctly \
included and attributed in the user request?
  * Flaws: missing words or sentences, text assigned to wrong speaker, \
hallucinated words not in the original, speakers' lines merged or confused. \
Note: natural speech disfluencies (repetitions like "I I", fillers like \
"um", "ah", false starts) are NORMAL in spontaneous speech and should NOT \
be penalized — only penalize actual errors or hallucinated content.
- structure_completeness: Are all 3 CoT parts (A, B, C) present and \
substantive? Part A and C should analyze speakers individually. Part B \
should discuss dialogue flow, turn-taking, and pacing.
  * Flaws: missing parts, empty or single-line sections, Part A or C fails \
to distinguish between speakers at all
  * NOT a flaw: Part B discussing dialogue dynamics at a conversation level \
rather than per-speaker — that is expected and correct
- reasoning_coherence: Does the reasoning flow logically across speakers?
  * Flaws: jumps in logic, contradictions, speaker transitions that don't \
make sense, inconsistent characterization of the same speaker
- cot_caption_consistency: Does the CoT align with the caption without \
CONTRADICTING it? The CoT is expected to elaborate, infer, and add \
reasoning beyond the caption — only penalize actual contradictions.
  * Flaws: CoT states a speaker's gender/accent/emotion differently than \
the caption, attributes swapped between speakers, factual contradictions
  * NOT a flaw: CoT adding voice descriptions, reasoning, or inferences \
that go beyond but do not contradict the caption — that is the purpose of \
a reasoning trace

## Output Format
- If score < 5: include "flaws" field explaining the issues
- If score = 5: just include "score"

Output JSON only:
{{"voice_attribute_match": {{"score": 4}}, \
"transcription_attribution": {{"score": 5}}, \
"structure_completeness": {{"score": 4}}, \
"reasoning_coherence": {{"score": 5}}, \
"cot_caption_consistency": {{"flaws": "...", "score": 4}}}}

## Multi-Talker TTS Training Example to Evaluate

**User Request:**
{user_request}

**Thinking Trace (assistant's reasoning):**
{cot}

**Rich Caption (target audio description):**
{caption}"""

# Expected dimensions for validation (flat, no categories)
EXPECTED_DIMENSIONS = [
    "voice_attribute_match",
    "transcription_attribution",
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

    # Try direct parsing first
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        # Try to extract JSON from response
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
    """Validate that response has expected flat structure."""
    for dim in EXPECTED_DIMENSIONS:
        if dim not in data:
            actual_keys = list(data.keys())
            key = f"missing_dim:{dim}|actual={actual_keys}"
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
    """Validate that all scores are in range 1-5."""
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
    """Compute overall score and pass/fail status.

    Returns (avg_score, min_score, overall_pass).
    """
    all_scores = []

    for dim in EXPECTED_DIMENSIONS:
        if dim in scores:
            val = scores[dim]
            if isinstance(val, dict) and "score" in val:
                score = val["score"]
                if isinstance(score, (int, float)):
                    all_scores.append(score)

    if not all_scores:
        return 0.0, 0.0, False

    avg_score = sum(all_scores) / len(all_scores)
    min_score = min(all_scores)
    overall_pass = min_score >= min_threshold and avg_score >= avg_threshold

    return round(avg_score, 2), round(min_score, 2), overall_pass


def build_judge_query(
    sample: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Build a query for quality judgment."""
    user_request = sample.get("user_request", "")
    cot = sample.get("cot", "")
    caption = sample.get("rich_caption", "")
    utt_id = sample.get("utt_id", "")

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
        "idx": utt_id,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 2048,
        "json_mode": True,
        "metadata": sample,
    }


def make_process_judge_result(min_threshold, avg_threshold):
    """Create a process function with threshold parameters."""

    def process_judge_result(
        result: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Process a quality judgment result."""
        response = result.get("response")
        metadata = result.get("metadata", {})
        utt_id = metadata.get("utt_id", "")

        scores = parse_judge_response(response)
        if scores is None:
            return None

        avg_score, min_score, overall_pass = compute_overall_score(
            scores, min_threshold, avg_threshold
        )

        return {
            "utt_id": utt_id,
            "text": metadata.get("text", ""),
            "rich_caption": metadata.get("rich_caption", ""),
            "audio_path": metadata.get("audio_path", ""),
            "user_request": metadata.get("user_request", ""),
            "cot": metadata.get("cot", ""),
            "scores": scores,
            "avg_score": avg_score,
            "min_score": min_score,
            "overall_pass": overall_pass,
            "idx": result["idx"],
        }

    return process_judge_result


def load_input_data(input_file: str) -> List[Dict[str, Any]]:
    """Load input JSONL file."""
    samples = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def compute_summary(
    results_file: str, min_threshold: float = 3.0
) -> Dict[str, Any]:
    """Compute summary statistics from judge results."""
    results = load_input_data(results_file)

    if not results:
        return {}

    total = len(results)
    passed = sum(1 for r in results if r.get("overall_pass", False))

    # Aggregate scores by dimension
    score_sums: Dict[str, float] = {}
    score_counts: Dict[str, int] = {}

    for r in results:
        scores = r.get("scores", {})
        for dim in EXPECTED_DIMENSIONS:
            if dim in scores:
                val = scores[dim]
                if isinstance(val, dict) and "score" in val:
                    score_sums[dim] = score_sums.get(dim, 0) + val["score"]
                    score_counts[dim] = score_counts.get(dim, 0) + 1

    # Compute averages
    avg_scores: Dict[str, float] = {}
    for dim in EXPECTED_DIMENSIONS:
        if score_counts.get(dim, 0) > 0:
            avg_scores[dim] = round(
                score_sums[dim] / score_counts[dim], 2
            )

    # Count failures by dimension (score < min_threshold)
    failure_breakdown: Dict[str, int] = {}
    for r in results:
        scores = r.get("scores", {})
        for dim in EXPECTED_DIMENSIONS:
            if dim in scores:
                val = scores[dim]
                if isinstance(val, dict) and "score" in val:
                    if val["score"] < min_threshold:
                        failure_breakdown[dim] = (
                            failure_breakdown.get(dim, 0) + 1
                        )

    failure_breakdown = dict(
        sorted(failure_breakdown.items(), key=lambda x: x[1], reverse=True)
    )

    return {
        "total_samples": total,
        "passed_samples": passed,
        "pass_rate": round(passed / total, 4) if total > 0 else 0,
        "avg_scores": avg_scores,
        "failure_breakdown": failure_breakdown,
    }


async def main_async(args):
    """Main async function."""
    # Load input data (stage 3 output)
    print(f"Loading input data from {args.input_file}...")
    samples = load_input_data(args.input_file)
    print(f"Loaded {len(samples)} samples")

    # Limit samples if specified
    if args.num_samples > 0:
        samples = samples[: args.num_samples]
        print(f"Limited to {len(samples)} samples")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    judge_file = os.path.join(args.output_dir, "judge_results.jsonl")

    # Check for existing progress
    processed_ids = set()
    if os.path.exists(judge_file) and args.resume:
        processed_ids = get_processed_indices(judge_file, idx_key="utt_id")
        print(f"Resuming: {len(processed_ids)} already processed")
    else:
        open(judge_file, "w").close()

    # Parse URLs and initialize processor
    base_urls = parse_vllm_urls(args.vllm_url)
    model = args.model or DEFAULT_MODEL
    num_servers = len(base_urls)
    workers_per_server = max(1, args.num_workers // num_servers)

    # Reset validation statistics
    for key in _stats:
        _stats[key] = 0
    _structure_issues.clear()

    # Build queries
    queries = []
    for sample in samples:
        utt_id = sample.get("utt_id", "")
        if utt_id in processed_ids:
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

        process_fn = make_process_judge_result(
            args.min_score, args.avg_score
        )
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

    # Fields to carry forward to stage 5 (exclude judge-specific fields)
    judge_only_keys = {"scores", "avg_score", "min_score", "overall_pass", "idx"}

    with open(filtered_file, "w", encoding="utf-8") as out_f:
        with open(judge_file, "r", encoding="utf-8") as in_f:
            for line in in_f:
                if not line.strip():
                    continue
                judge_result = json.loads(line)
                if judge_result.get("overall_pass", False):
                    sample = {
                        k: v
                        for k, v in judge_result.items()
                        if k not in judge_only_keys
                    }
                    out_f.write(
                        json.dumps(sample, ensure_ascii=False) + "\n"
                    )
                    passed_count += 1
                else:
                    failed_count += 1
                    scores = judge_result.get("scores", {})
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

    # Compute and save summary
    summary = compute_summary(judge_file, min_threshold=args.min_score)
    summary["filtered_passed"] = passed_count
    summary["filtered_failed"] = failed_count
    summary_file = os.path.join(args.output_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    final_count = len(get_processed_indices(judge_file, idx_key="utt_id"))
    print(f"\nDone! Total judged: {final_count}/{len(samples)}")
    print(f"Passed filter: {passed_count}")
    print(f"Failed filter: {failed_count}")
    print(f"Pass rate: {summary.get('pass_rate', 0):.1%}")
    print(f"Filtered output: {filtered_file}")
    print(f"Summary: {summary_file}")

    # Print filter failure analysis
    if failed_count > 0:
        print("\n" + "=" * 60)
        print(f"Filter Failure Analysis ({failed_count} failed examples):")

        print("\n  Bottleneck dimension (worst score per failed example):")
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

    # Print validation statistics
    total_validated = sum(_stats.values())
    if total_validated > 0:
        print("\n" + "=" * 60)
        print("Validation Summary:")
        print(f"  {'Reason':<25} {'Count':>8} {'Percent':>8}")
        print("-" * 60)
        for reason, count in _stats.items():
            pct = count / total_validated * 100
            print(f"  {reason:<25} {count:>8} {pct:>7.1f}%")
        print("-" * 60)
        print(f"  {'Total':<25} {total_validated:>8}")
        print("=" * 60)

    # Print structure issue breakdown
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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "LLM-as-judge quality validation for multi-talker TTS examples."
        )
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input JSONL (stage 3 CoT output).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for judge results.",
    )
    parser.add_argument(
        "--vllm_url",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM API base URL.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name for vLLM API.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=256,
        help="Number of concurrent API requests.",
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
        help="Number of samples to process (-1 for all).",
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
        help="Minimum score threshold per dimension (default: 3).",
    )
    parser.add_argument(
        "--avg_score",
        type=float,
        default=3.5,
        help="Minimum average score threshold (default: 3.5).",
    )
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
