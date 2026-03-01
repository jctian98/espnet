#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Stage 1: Extract spoken text from rich captions via LLM, filter by WER <= 10%.

The reference transcript is the Whisper output from Stage 0 (not ground truth),
since no ground truth transcription may be available.  A WER threshold of 10%
is used for speech data (stricter than singing, since ASR is more accurate on
speech).
"""

import argparse
import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional, Set

from sft_vllm_client import (
    DEFAULT_MODEL,
    MassiveQueryProcessor,
    get_processed_indices,
    parse_vllm_urls,
)

# =============================================================================
# Prompts for spoken text extraction from speech captions
# =============================================================================
PROMPTS_BY_VERSION = {
    "v1": {
        "system": (
            "You are a precise text extraction assistant. Given a rich audio "
            "caption that describes a speech recording, extract the EXACT "
            "spoken text uttered by the speaker. Output ONLY the verbatim "
            "text with no quotes, no explanation, and no additional "
            "formatting. Include all lines of text that are described or "
            "quoted in the caption. If no spoken text is described, output "
            "exactly: NO_TEXT"
        ),
        "user": (
            "Extract the exact spoken text from this speech audio caption:"
            "\n\n{caption}"
        ),
    },
}


def get_prompts(version: str) -> Dict[str, str]:
    if version not in PROMPTS_BY_VERSION:
        available = list(PROMPTS_BY_VERSION.keys())
        raise ValueError(
            f"Version '{version}' not found. Available: {available}"
        )
    return PROMPTS_BY_VERSION[version]


def normalize_text(text: str) -> str:
    """Lowercase and remove non-alphanumeric characters for WER comparison."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate (edit distance at word level)."""
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()

    if not ref_words and not hyp_words:
        return 0.0
    if not ref_words:
        return float(len(hyp_words))
    if not hyp_words:
        return 1.0

    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,
                    d[i][j - 1] + 1,
                    d[i - 1][j - 1] + 1,
                )

    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def load_input_data(input_file: str) -> List[Dict[str, Any]]:
    """Load stage0 JSONL, skipping samples with has_whisper_transcription=False."""
    samples = []
    skipped_no_transcription = 0
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                if not record.get("has_whisper_transcription", True):
                    skipped_no_transcription += 1
                    continue
                samples.append(record)
    if skipped_no_transcription > 0:
        print(
            f"Skipped {skipped_no_transcription} samples "
            f"with has_whisper_transcription=False"
        )
    return samples


def build_extraction_query(
    idx: int,
    sample: Dict[str, Any],
    prompts: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    """Build a query to extract spoken text from the speech caption."""
    caption = sample.get("caption", "")
    if not caption or len(caption.strip()) < 20:
        return None

    messages = [
        {"role": "system", "content": prompts["system"]},
        {
            "role": "user",
            "content": prompts["user"].format(caption=caption),
        },
    ]

    return {
        "idx": idx,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 512,
        "metadata": sample,
    }


def process_extraction_result(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Compute WER (LLM-extracted vs Whisper reference) and emit record."""
    response = result.get("response")
    metadata = result.get("metadata", {})

    if response is None:
        return None

    extracted_lyrics = response.strip()

    if extracted_lyrics == "NO_TEXT":
        return None

    # Reference is the Whisper transcription from Stage 0
    whisper_lyrics = metadata.get("whisper_lyrics", "")
    if not whisper_lyrics:
        return None

    wer = compute_wer(whisper_lyrics, extracted_lyrics)

    return {
        "example_id": metadata.get("example_id", ""),
        "dataset": metadata.get("dataset", ""),
        "audio_path": metadata.get("audio_path", ""),
        "duration": metadata.get("duration", 0.0),
        "caption": metadata.get("caption", ""),
        "whisper_lyrics": whisper_lyrics,
        "extracted_lyrics": extracted_lyrics,
        "wer": round(wer, 4),
        "idx": result["idx"],
    }


async def main_async(args) -> None:
    prompts = get_prompts(args.version)
    print(f"Using prompt version: {args.version}")
    print(f"WER threshold: {args.wer_threshold}")

    print(f"Loading stage0 output from: {args.input_file}")
    samples = load_input_data(args.input_file)
    print(f"Total samples: {len(samples)}")

    if args.num_samples > 0:
        samples = samples[: args.num_samples]
        print(f"Limited to {len(samples)} samples")

    os.makedirs(args.output_dir, exist_ok=True)
    all_results_file = os.path.join(args.output_dir, "stage1_all.jsonl")
    filtered_file = os.path.join(args.output_dir, "stage1_filtered.jsonl")

    # Resume support
    processed_indices: Set[Any] = set()
    if os.path.exists(all_results_file) and args.resume:
        processed_indices = get_processed_indices(all_results_file)
        print(f"Resuming: {len(processed_indices)} already processed")
    else:
        open(all_results_file, "w").close()

    # Build queries for unprocessed samples
    queries = []
    for idx, sample in enumerate(samples):
        if idx in processed_indices:
            continue
        query = build_extraction_query(idx, sample, prompts)
        if query is not None:
            queries.append(query)

    print(f"Queries to process: {len(queries)}")

    if queries:
        base_urls = parse_vllm_urls(args.vllm_url)
        model = args.model or DEFAULT_MODEL
        workers_per_server = max(1, args.num_workers // len(base_urls))

        processor = MassiveQueryProcessor(
            base_urls=base_urls,
            model=model,
            workers_per_server=workers_per_server,
            timeout=args.timeout,
            checkpoint_interval=10000,
        )

        await processor.process_all(
            queries=queries,
            output_file=all_results_file,
            process_fn=process_extraction_result,
        )

    # Filter by WER threshold
    print(f"\nFiltering by WER <= {args.wer_threshold}...")
    total = 0
    kept = 0
    wer_distribution: Dict[float, int] = {}

    with open(filtered_file, "w", encoding="utf-8") as out_f:
        with open(all_results_file, "r", encoding="utf-8") as in_f:
            for line in in_f:
                if not line.strip():
                    continue
                record = json.loads(line)
                total += 1
                wer = record.get("wer", 1.0)

                wer_bucket = round(wer * 10) / 10  # round to nearest 0.1
                wer_distribution[wer_bucket] = (
                    wer_distribution.get(wer_bucket, 0) + 1
                )

                if wer <= args.wer_threshold:
                    kept += 1
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Write summary
    summary = {
        "total_extracted": total,
        "kept": kept,
        "filter_rate": round(kept / total, 4) if total > 0 else 0,
        "wer_threshold": args.wer_threshold,
        "wer_distribution": {
            str(k): v for k, v in sorted(wer_distribution.items())
        },
    }
    summary_file = os.path.join(args.output_dir, "stage1_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("WER Distribution:")
    print(f"{'WER Range':<15} {'Count':>8} {'Percent':>8}  Bar")
    print("-" * 60)
    for wer_bucket in sorted(wer_distribution.keys()):
        count = wer_distribution[wer_bucket]
        pct = count / total * 100 if total > 0 else 0
        bar = "#" * int(pct / 2)
        marker = " <-- kept" if wer_bucket <= args.wer_threshold else ""
        print(
            f"{wer_bucket:>5.1f}           {count:>8} {pct:>7.1f}%  {bar}{marker}"
        )
    print("-" * 60)
    print(f"{'Total':<15} {total:>8}")
    print(f"{'Kept':<15} {kept:>8} ({summary['filter_rate']:.1%})")
    print(f"{'Dropped':<15} {total - kept:>8}")
    print("=" * 60)
    print(f"\nFiltered output : {filtered_file}")
    print(f"Summary         : {summary_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1: Extract spoken text from captions and filter by WER."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Stage 0 output JSONL (with whisper_lyrics field).",
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
        help="vLLM API base URL(s), colon-separated.",
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
        "--resume",
        action="store_true",
        help="Resume from existing progress.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Number of samples to process (-1 for all).",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="Prompt version to use.",
    )
    parser.add_argument(
        "--wer_threshold",
        type=float,
        default=0.10,
        help="Maximum WER to keep a sample (default: 0.10).",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
