#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Stage 3: Generate application-aware chain-of-thought reasoning for TTS."""

import argparse
import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from sft_vllm_client import (
    DEFAULT_MODEL,
    MassiveQueryProcessor,
    get_processed_indices,
    parse_vllm_urls,
)

# =============================================================================
# Version-specific prompts for TTS CoT generation
# =============================================================================
PROMPTS_BY_VERSION = {
    "v1": {
        "system": (
            "You are a text-to-speech (TTS) system that thinks carefully "
            "before generating speech. Given a user's TTS request, its "
            "application category, and the rich audio description of the "
            "target speech, explain your reasoning.\n\n"
            "Your reasoning MUST follow this 3-part structure:\n\n"
            "Part A - Request & Application Understanding:\n"
            "What is the user asking for? What application category does "
            "this serve (e.g., audiobook, news broadcast, podcast)? What "
            "specific vocal and stylistic expectations come with this "
            "application? Identify explicitly requested attributes and "
            "note what you need to decide.\n\n"
            "Part B - Speech Delivery Planning:\n"
            "How should the text be spoken for this specific application? "
            "Consider:\n"
            "- Prosody and phrasing appropriate for the application\n"
            "- Dynamics: volume contour, emotional arc\n"
            "- Application-specific conventions (e.g., news: even pacing "
            "and authority; audiobook: warmth and character; commercial: "
            "persuasion and energy)\n"
            "- Expressive elements, emphasis patterns, breathing, pitch "
            "variation\n"
            "- Pacing and temporal feel\n\n"
            "Part C - Voice & Acoustic Characterization:\n"
            "What specific vocal qualities and recording characteristics "
            "match both the request and the application? Consider timbre, "
            "breathiness, resonance, vocal warmth, recording setting, and "
            "environmental acoustics.\n\n"
            "RULES:\n"
            "- Output ONLY the reasoning trace\n"
            "- Do NOT copy the audio description verbatim\n"
            "- Keep reasoning concise and focused on speech\n"
            "- Reference the application context naturally"
        ),
        "user": (
            "User TTS request:\n{user_request}\n\n"
            "Application category: {application_category}\n\n"
            "Audio description of the speech that was produced:\n"
            "{caption}\n\n"
            "Explain the reasoning. Use Parts A, B, and C:"
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


_stats = {
    "success": 0,
    "no_response": 0,
    "too_short": 0,
    "missing_parts": 0,
}

REQUIRED_MARKERS = ["Part A", "Part B", "Part C"]


def validate_cot(trace: str, example_id: str = "") -> bool:
    """Validate that CoT contains required sections."""
    if trace is None:
        _stats["no_response"] += 1
        print(f"[DISCARD] id={example_id} reason=no_response")
        return False

    if len(trace.strip()) < 50:
        _stats["too_short"] += 1
        print(
            f"[DISCARD] id={example_id} reason=too_short "
            f"len={len(trace.strip())}"
        )
        return False

    trace_lower = trace.lower()
    missing = [m for m in REQUIRED_MARKERS if m.lower() not in trace_lower]
    if missing:
        _stats["missing_parts"] += 1
        print(
            f"[DISCARD] id={example_id} reason=missing_parts "
            f"missing={missing}"
        )
        return False

    _stats["success"] += 1
    return True


def build_cot_query(
    sample: Dict[str, Any],
    prompts: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    """Build a query for application-aware TTS CoT generation."""
    user_request = sample.get("user_request", "")
    caption = sample.get("caption", "")
    application_category = sample.get("application_category", "")
    example_id = sample.get("example_id", "")

    if not user_request or not caption:
        return None

    messages = [
        {"role": "system", "content": prompts["system"]},
        {
            "role": "user",
            "content": prompts["user"].format(
                user_request=user_request,
                application_category=application_category,
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
        "application_category": metadata.get("application_category", ""),
        "cot": response.strip(),
        "idx": result["idx"],
    }


def load_input_data(input_file: str) -> List[Dict[str, Any]]:
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
        open(output_file, "w").close()

    base_urls = parse_vllm_urls(args.vllm_url)
    model = args.model or DEFAULT_MODEL
    workers_per_server = max(1, args.num_workers // len(base_urls))

    for key in _stats:
        _stats[key] = 0

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
        description="Stage 3: Generate TTS CoT reasoning traces."
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
