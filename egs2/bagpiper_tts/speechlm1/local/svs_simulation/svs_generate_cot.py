#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Stage 3: Generate chain-of-thought reasoning traces for SVS examples."""

import argparse
import asyncio
import json
import os
from typing import Any, Dict, Optional

from sft_vllm_client import (
    DEFAULT_MODEL,
    MassiveQueryProcessor,
    get_processed_indices,
    parse_vllm_urls,
)

# =============================================================================
# Version-specific prompts for SVS CoT generation
# =============================================================================
PROMPTS_BY_VERSION = {
    "v1": {
        "system": (
            "You are a singing voice synthesis (SVS) system that thinks "
            "carefully before generating a performance. Given a user's SVS "
            "request and the rich audio description of the performance that "
            "was produced, explain the reasoning process that led from the "
            "request to that singing output.\n\n"
            "Your reasoning MUST follow this 3-part structure:\n\n"
            "Part A - Request Understanding:\n"
            "What vocal and stylistic attributes is the user asking for? "
            "Identify explicitly requested attributes (gender, vocal range, "
            "genre, emotion, energy, performance character) and note what "
            "is left unspecified that you need to decide.\n\n"
            "Part B - Vocal Delivery Planning:\n"
            "How should the lyrics be performed? Consider:\n"
            "- Melodic phrasing: where to breathe, swell, or hold notes\n"
            "- Dynamics: volume contour, emotional arc across the passage\n"
            "- Ornaments and expressive devices: vibrato, runs, melisma, "
            "belt, falsetto, or other stylistic flourishes\n"
            "- Challenging elements: high notes, rapid passages, consonant "
            "clusters, rhythmic syncopation\n"
            "- Pacing and temporal feel appropriate for the genre\n\n"
            "Part C - Voice & Acoustic Characterization:\n"
            "What specific vocal qualities and recording characteristics "
            "match the request? Consider timbre, breathiness, chest vs head "
            "voice register, resonance, vocal warmth, recording setting "
            "(studio, live, intimate), and any environmental acoustics that "
            "complete the performance picture.\n"
            "If the request mentions non-singing elements (background "
            "music, instrumental accompaniment, intro/outro without "
            "vocals), briefly address how they complement the vocal "
            "performance.\n\n"
            "RULES:\n"
            "- Output ONLY the reasoning trace\n"
            "- Do NOT copy or repeat the audio description verbatim\n"
            "- Keep the reasoning concise and focused on singing, with "
            "brief notes on accompaniment when relevant"
        ),
        "user": (
            "User SVS request:\n{user_request}\n\n"
            "Audio description of the performance that was produced:\n"
            "{caption}\n\n"
            "Explain the reasoning that led from the user's request to this "
            "singing output. Use Parts A, B, and C:"
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


def _new_stats() -> Dict[str, int]:
    return {
        "success": 0,
        "no_response": 0,
        "too_short": 0,
        "missing_parts": 0,
    }


_stats = _new_stats()

REQUIRED_MARKERS = ["Part A", "Part B", "Part C"]
MIN_COT_WORDS = 30


def validate_cot(trace: Optional[str], example_id: str = "") -> bool:
    """Validate that CoT contains required sections."""
    if trace is None:
        _stats["no_response"] += 1
        print(f"[DISCARD] id={example_id} reason=no_response")
        return False

    word_count = len(trace.split())
    if word_count < MIN_COT_WORDS:
        _stats["too_short"] += 1
        print(
            f"[DISCARD] id={example_id} reason=too_short "
            f"words={word_count}"
        )
        return False

    trace_lower = trace.lower()
    missing = [m for m in REQUIRED_MARKERS if m.lower() not in trace_lower]
    if missing:
        _stats["missing_parts"] += 1
        print(
            f"[DISCARD] id={example_id} reason=missing_parts missing={missing}"
        )
        return False

    _stats["success"] += 1
    return True


def build_cot_query(
    sample: Dict[str, Any],
    prompts: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    """Build a query for SVS CoT generation."""
    user_request = sample.get("user_request", "")
    caption = sample.get("caption", "")
    example_id = sample.get("example_id", "")

    if not user_request or not caption:
        return None

    messages = [
        {"role": "system", "content": prompts["system"]},
        {
            "role": "user",
            "content": prompts["user"].format(
                user_request=user_request,
                caption=caption,
            ),
        },
    ]

    return {
        "idx": example_id,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 1024,
        "metadata": sample,
    }


def process_cot_result(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process a CoT generation result."""
    response = result.get("response")
    metadata = result.get("metadata", {})
    example_id = metadata.get("example_id", "")

    if response is not None:
        response = response.strip()

    if not validate_cot(response, example_id):
        return None

    return {
        "example_id": example_id,
        "dataset": metadata.get("dataset", ""),
        "audio_path": metadata.get("audio_path", ""),
        "duration": metadata.get("duration", 0.0),
        "caption": metadata.get("caption", ""),
        "whisper_lyrics": metadata.get("whisper_lyrics", ""),
        "extracted_lyrics": metadata.get("extracted_lyrics", ""),
        "user_request": metadata.get("user_request", ""),
        "cot": response,
        "idx": result["idx"],
    }


def load_input_data(input_file: str) -> list:
    samples = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


async def main_async(args) -> None:
    prompts = get_prompts(args.version)
    print(f"Using prompt version: {args.version}")

    print(f"Loading input data from: {args.input_file}")
    samples = load_input_data(args.input_file)
    print(f"Loaded {len(samples)} samples")

    if args.num_samples > 0:
        samples = samples[: args.num_samples]
        print(f"Limited to {len(samples)} samples")

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "cot.jsonl")

    processed_ids = set()
    if os.path.exists(output_file) and args.resume:
        processed_ids = get_processed_indices(
            output_file, idx_key="example_id"
        )
        print(f"Resuming: {len(processed_ids)} already processed")
    else:
        with open(output_file, "w"):
            pass

    base_urls = parse_vllm_urls(args.vllm_url)
    model = args.model or DEFAULT_MODEL
    workers_per_server = max(1, args.num_workers // len(base_urls))

    global _stats
    _stats = _new_stats()

    queries = []
    for sample in samples:
        eid = sample.get("example_id", "")
        if eid in processed_ids:
            continue
        query = build_cot_query(sample, prompts)
        if query is not None:
            queries.append(query)

    print(f"Queries to process: {len(queries)}")

    if not queries:
        print("All samples already processed!")
        return

    processor = MassiveQueryProcessor(
        base_urls=base_urls,
        model=model,
        workers_per_server=workers_per_server,
        timeout=args.timeout,
        checkpoint_interval=10000,
    )

    await processor.process_all(
        queries=queries,
        output_file=output_file,
        process_fn=process_cot_result,
    )

    final_count = len(
        get_processed_indices(output_file, idx_key="example_id")
    )
    print(f"\nDone! Total success: {final_count}/{len(samples)}")
    print(f"Output: {output_file}")

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 3: Generate SVS CoT reasoning traces."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Stage 2 user requests JSONL.",
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
        "--resume",
        action="store_true",
        help="Resume from existing progress.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Number of samples (-1 for all).",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="Prompt version.",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
