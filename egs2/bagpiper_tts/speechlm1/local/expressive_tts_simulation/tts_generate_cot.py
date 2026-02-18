#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Stage 3: Generate chain-of-thought reasoning traces for TTS examples."""

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
# Version-specific prompts for CoT generation
# =============================================================================
PROMPTS_BY_VERSION = {
    "v1": {
        "system": (
            "You are a text-to-speech system that thinks before generating "
            "speech. Given a user's TTS request and the audio description "
            "that was produced, explain the reasoning process that led from "
            "the request to that audio output.\n\n"
            "Your reasoning MUST follow this 3-part structure:\n\n"
            "Part A - Intent Understanding:\n"
            "What voice characteristics is the user asking for? Identify "
            "the explicitly requested attributes (gender, age, emotion, "
            "accent, pace, etc.) and note what is left unspecified that "
            "you need to decide.\n\n"
            "Part B - Speech Planning:\n"
            "How should the text be delivered? Consider:\n"
            "- Prosody: which words to emphasize, where to place pauses\n"
            "- Emotional arc: does the emotion shift within the text?\n"
            "- Challenging elements: dialogue, names, numbers, punctuation\n"
            "- Pacing and rhythm for the specific content\n\n"
            "Part C - Voice & Environment Characterization:\n"
            "What specific voice qualities and recording characteristics "
            "match the request? Consider timbre, breathiness, resonance, "
            "vocal warmth, recording setting, and any other qualities that "
            "complete the audio picture.\n\n"
            "RULES:\n"
            "- Output ONLY the reasoning trace\n"
            "- Do NOT copy or repeat the audio description\n"
            "- Keep the reasoning concise and focused"
        ),
        "user": (
            "User request:\n{user_request}\n\n"
            "Audio description that was produced:\n{caption}\n\n"
            "Explain the reasoning that led from the user's request to "
            "this audio output. Use Parts A, B, and C:"
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


# Validation statistics (reset per run in main_async)
_stats = {
    "success": 0,
    "no_response": 0,
    "too_short": 0,
    "missing_parts": 0,
}

REQUIRED_MARKERS = ["Part A", "Part B", "Part C"]


def validate_cot(trace: str, utt_id: str = "") -> bool:
    """Validate that CoT contains required sections."""
    if trace is None:
        _stats["no_response"] += 1
        print(f"[DISCARD] utt_id={utt_id} reason=no_response")
        return False

    if len(trace.strip()) < 50:
        _stats["too_short"] += 1
        print(
            f"[DISCARD] utt_id={utt_id} reason=too_short "
            f"len={len(trace.strip())}"
        )
        return False

    trace_lower = trace.lower()
    missing = [m for m in REQUIRED_MARKERS if m.lower() not in trace_lower]

    if missing:
        _stats["missing_parts"] += 1
        print(
            f"[DISCARD] utt_id={utt_id} reason=missing_parts "
            f"missing={missing}"
        )
        return False

    _stats["success"] += 1
    return True


def build_cot_query(
    sample: Dict[str, Any],
    prompts: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    """Build a query for CoT generation."""
    user_request = sample.get("user_request", "")
    caption = sample.get("caption", "")
    utt_id = sample.get("utt_id", "")

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
        "idx": utt_id,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 1024,
        "metadata": sample,
    }


def process_cot_result(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process a CoT generation result."""
    response = result.get("response")
    metadata = result.get("metadata", {})
    utt_id = metadata.get("utt_id", "")

    if not validate_cot(response, utt_id):
        return None

    return {
        "utt_id": metadata.get("utt_id", ""),
        "text": metadata.get("text", ""),
        "caption": metadata.get("caption", ""),
        "audio_path": metadata.get("audio_path", ""),
        "speaker_id": metadata.get("speaker_id", ""),
        "duration": metadata.get("duration", 0.0),
        "subset": metadata.get("subset", ""),
        "user_request": metadata.get("user_request", ""),
        "cot": response.strip(),
        "idx": result["idx"],
    }


def load_input_data(input_file: str) -> List[Dict[str, Any]]:
    """Load input JSONL file."""
    samples = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


async def main_async(args):
    """Main async function."""
    prompts = get_prompts(args.version)
    print(f"Using prompt version: {args.version}")

    # Load input data (stage 2 output)
    print(f"Loading input data from {args.input_file}...")
    samples = load_input_data(args.input_file)
    print(f"Loaded {len(samples)} samples")

    # Limit samples if specified
    if args.num_samples > 0:
        samples = samples[: args.num_samples]
        print(f"Limited to {len(samples)} samples")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "cot.jsonl")

    # Check for existing progress
    processed_ids = set()
    if os.path.exists(output_file) and args.resume:
        processed_ids = get_processed_indices(output_file, idx_key="utt_id")
        print(f"Resuming: {len(processed_ids)} already processed")
    else:
        open(output_file, "w").close()

    # Parse URLs and initialize processor
    base_urls = parse_vllm_urls(args.vllm_url)
    model = args.model or DEFAULT_MODEL
    num_servers = len(base_urls)
    workers_per_server = max(1, args.num_workers // num_servers)

    # Reset validation statistics
    for key in _stats:
        _stats[key] = 0

    # Build queries
    queries = []
    for sample in samples:
        utt_id = sample.get("utt_id", "")
        if utt_id in processed_ids:
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

    # Final summary
    final_count = len(get_processed_indices(output_file, idx_key="utt_id"))
    print(f"\nDone! Total success: {final_count}/{len(samples)}")
    print(f"Output saved to: {output_file}")

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


def main():
    parser = argparse.ArgumentParser(
        description="Generate CoT reasoning traces for TTS examples."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input JSONL (stage 2 user requests output).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for CoT traces.",
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
