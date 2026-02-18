#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Stage 1: Extract transcription from rich captions via LLM, filter by WER=0."""

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
# Prompts for transcription extraction
# =============================================================================
PROMPTS_BY_VERSION = {
    "v1": {
        "system": (
            "You are a precise text extraction assistant. Given a rich audio "
            "caption that describes a speech recording, extract the EXACT words "
            "spoken by the speaker. Output ONLY the verbatim transcription text "
            "with no quotes, no explanation, and no additional formatting. "
            "If multiple sentences are spoken, include all of them. "
            "If no speech is described in the caption, output exactly: NO_SPEECH"
        ),
        "user": (
            "Extract the exact spoken words from this audio caption:\n\n"
            "{caption}"
        ),
    },
}


def get_prompts(version: str) -> Dict[str, str]:
    """Get prompts for a specific version."""
    if version not in PROMPTS_BY_VERSION:
        available = list(PROMPTS_BY_VERSION.keys())
        raise ValueError(
            f"Version '{version}' not found. Available versions: {available}"
        )
    return PROMPTS_BY_VERSION[version]


def normalize_text(text: str) -> str:
    """Normalize text for WER comparison: lowercase, remove punctuation."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate between reference and hypothesis.

    Uses edit distance at word level.
    """
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()

    if not ref_words and not hyp_words:
        return 0.0
    if not ref_words:
        return float(len(hyp_words))
    if not hyp_words:
        return 1.0

    # Dynamic programming for edit distance
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
                    d[i - 1][j] + 1,  # deletion
                    d[i][j - 1] + 1,  # insertion
                    d[i - 1][j - 1] + 1,  # substitution
                )

    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def load_kaldi_file(filepath: str) -> Dict[str, str]:
    """Load a Kaldi-format file (utt_id SPACE content)."""
    data = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                data[parts[0]] = parts[1]
    return data


def load_captions(filepath: str) -> Dict[str, Dict[str, Any]]:
    """Load captions JSONL file, keyed by utt_id."""
    captions = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                utt_id = record.get("utt_id", "")
                if utt_id:
                    captions[utt_id] = record
    return captions


def load_all_subsets(
    data_root: str, subsets: List[str]
) -> List[Dict[str, Any]]:
    """Load and join data from all subsets.

    Returns list of sample dicts with utt_id, text, audio_path,
    speaker_id, caption, duration, subset.
    """
    samples = []

    for subset in subsets:
        data_dir = os.path.join(data_root, "data", subset)
        caption_path = os.path.join(
            data_root, "rich_caption", subset, "captions_merged.jsonl"
        )

        # Load all source files
        text_data = load_kaldi_file(os.path.join(data_dir, "text"))
        wavscp_data = load_kaldi_file(os.path.join(data_dir, "wav.scp"))
        utt2spk_data = load_kaldi_file(os.path.join(data_dir, "utt2spk"))
        caption_data = load_captions(caption_path)

        print(
            f"  {subset}: {len(text_data)} text, "
            f"{len(wavscp_data)} wav.scp, "
            f"{len(utt2spk_data)} utt2spk, "
            f"{len(caption_data)} captions"
        )

        # Join on utt_id (only keep samples with all fields)
        common_ids = (
            set(text_data.keys())
            & set(wavscp_data.keys())
            & set(utt2spk_data.keys())
            & set(caption_data.keys())
        )

        for utt_id in sorted(common_ids):
            audio_rel_path = wavscp_data[utt_id]
            audio_abs_path = os.path.join(data_root, audio_rel_path)

            samples.append(
                {
                    "utt_id": utt_id,
                    "text": text_data[utt_id],
                    "audio_path": audio_abs_path,
                    "speaker_id": utt2spk_data[utt_id],
                    "caption": caption_data[utt_id].get("caption", ""),
                    "duration": caption_data[utt_id].get("duration", 0.0),
                    "subset": subset,
                }
            )

        print(f"  {subset}: {len(common_ids)} samples after join")

    return samples


def build_extraction_query(
    idx: int,
    sample: Dict[str, Any],
    prompts: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    """Build a query for transcription extraction."""
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
    """Process extraction result: compute WER and build output record."""
    response = result.get("response")
    metadata = result.get("metadata", {})

    if response is None:
        return None

    extracted_text = response.strip()

    # Skip if LLM says no speech
    if extracted_text == "NO_SPEECH":
        return None

    # Compute WER against ground truth
    ground_truth = metadata.get("text", "")
    wer = compute_wer(ground_truth, extracted_text)

    return {
        "utt_id": metadata.get("utt_id", ""),
        "text": ground_truth,
        "caption": metadata.get("caption", ""),
        "audio_path": metadata.get("audio_path", ""),
        "speaker_id": metadata.get("speaker_id", ""),
        "duration": metadata.get("duration", 0.0),
        "subset": metadata.get("subset", ""),
        "extracted_text": extracted_text,
        "wer": round(wer, 4),
        "idx": result["idx"],
    }


async def main_async(args):
    """Main async function."""
    prompts = get_prompts(args.version)
    print(f"Using prompt version: {args.version}")

    # Load data from all subsets
    subsets = [s.strip() for s in args.subsets.split(",")]
    print(f"Loading data from {args.data_root}")
    print(f"Subsets: {subsets}")
    samples = load_all_subsets(args.data_root, subsets)
    print(f"Total samples: {len(samples)}")

    # Limit samples if specified
    if args.num_samples > 0:
        samples = samples[: args.num_samples]
        print(f"Limited to {len(samples)} samples")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    all_results_file = os.path.join(args.output_dir, "stage1_all.jsonl")
    filtered_file = os.path.join(args.output_dir, "stage1_filtered.jsonl")

    # Check for existing progress
    processed_indices: Set[Any] = set()
    if os.path.exists(all_results_file) and args.resume:
        processed_indices = get_processed_indices(all_results_file)
        print(f"Resuming: {len(processed_indices)} already processed")
    else:
        open(all_results_file, "w").close()

    # Build queries
    queries = []
    for idx, sample in enumerate(samples):
        if idx in processed_indices:
            continue
        query = build_extraction_query(idx, sample, prompts)
        if query is not None:
            queries.append(query)

    print(f"Queries to process: {len(queries)}")

    if queries:
        # Parse URLs and initialize processor
        base_urls = parse_vllm_urls(args.vllm_url)
        model = args.model or DEFAULT_MODEL
        num_servers = len(base_urls)
        workers_per_server = max(1, args.num_workers // num_servers)

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

    # Filter for WER=0
    print("\nFiltering for WER=0...")
    total = 0
    kept = 0
    wer_distribution = {}

    with open(filtered_file, "w", encoding="utf-8") as out_f:
        with open(all_results_file, "r", encoding="utf-8") as in_f:
            for line in in_f:
                if not line.strip():
                    continue
                record = json.loads(line)
                total += 1
                wer = record.get("wer", 1.0)

                # Track WER distribution
                wer_bucket = round(wer, 1)
                wer_distribution[wer_bucket] = (
                    wer_distribution.get(wer_bucket, 0) + 1
                )

                if wer == 0.0:
                    kept += 1
                    out_f.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )

    # Write summary
    summary = {
        "total_extracted": total,
        "wer_0_kept": kept,
        "filter_rate": round(kept / total, 4) if total > 0 else 0,
        "wer_distribution": dict(sorted(wer_distribution.items())),
    }
    summary_file = os.path.join(args.output_dir, "stage1_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Print WER frequency statistics
    print("\n" + "=" * 60)
    print("WER Distribution:")
    print(f"{'WER Range':<15} {'Count':>8} {'Percent':>8}  Bar")
    print("-" * 60)
    for wer_bucket in sorted(wer_distribution.keys()):
        count = wer_distribution[wer_bucket]
        pct = count / total * 100 if total > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"{wer_bucket:>5.1f}           {count:>8} {pct:>7.1f}%  {bar}")
    print("-" * 60)
    print(f"{'Total':<15} {total:>8}")
    print(f"{'WER=0 (kept)':<15} {kept:>8} ({summary['filter_rate']:.1%})")
    print(f"{'WER>0 (dropped)':<15} {total - kept:>8}")
    print("=" * 60)

    print(f"\nOutput: {filtered_file}")
    print(f"Summary: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract transcription from rich captions and filter by WER."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root path to LibriTTS-R data.",
    )
    parser.add_argument(
        "--subsets",
        type=str,
        default="train-clean-100,train-clean-360,train-other-500",
        help="Comma-separated subset names.",
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
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
