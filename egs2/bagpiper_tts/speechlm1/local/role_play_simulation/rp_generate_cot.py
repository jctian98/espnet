#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Stage 3: Generate chain-of-thought with voice inference from persona.

3-part CoT where the model infers voice features from the persona:
- Part A: Persona Analysis (who is this character?)
- Part B: Voice Feature Inference (derive voice from persona, explain WHY)
- Part C: Speech Delivery Planning (how to deliver the specific text)
"""

import argparse
import asyncio
import json
import os
import re
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
            "You are a role-play text-to-speech system that thinks before "
            "generating speech. Given a user's persona-based TTS request "
            "and the audio description that was produced, explain the "
            "reasoning process that led from the persona to that audio "
            "output.\n\n"
            "Your reasoning MUST follow this 3-part structure:\n\n"
            "Part A - Persona Analysis:\n"
            "Who is this character? Analyze their background, occupation, "
            "age, personality, regional origins, and speaking habits as "
            "described or implied in the persona.\n\n"
            "Part B - Voice Feature Inference:\n"
            "Derive specific voice characteristics from Part A's analysis. "
            "For each feature (gender, age range, accent, emotion, pace, "
            "timbre, breathiness, resonance), explain WHY it follows from "
            "the persona. Connect character traits to vocal qualities.\n\n"
            "Part C - Speech Delivery Planning:\n"
            "How should the specific text be delivered given the inferred "
            "voice? Consider:\n"
            "- Prosody: which words to emphasize, where to place pauses\n"
            "- Emotional arc: does the emotion shift within the text?\n"
            "- Character-consistent delivery: how this persona would "
            "naturally say these words\n"
            "- Pacing and rhythm for the specific content\n\n"
            "RULES:\n"
            "- Output ONLY the reasoning trace\n"
            "- Do NOT copy or repeat the audio description\n"
            "- Keep the reasoning concise and focused\n"
            "- In Part B, always explain the REASONING behind each "
            "voice feature inference"
        ),
        "user": (
            "User request (persona + text):\n{user_request}\n\n"
            "Audio description that was produced:\n{caption}\n\n"
            "Explain the reasoning that led from the persona to this "
            "audio output. Use Parts A, B, and C:"
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
    "non_latin_chars": 0,
}

REQUIRED_MARKERS = ["Part A", "Part B", "Part C"]
MIN_COT_LENGTH = 80

# Regex to detect CJK and other non-Latin script characters
_NON_LATIN_RE = re.compile(
    r"[\u4e00-\u9fff"  # CJK Unified Ideographs (Chinese)
    r"\u3040-\u309f"  # Hiragana
    r"\u30a0-\u30ff"  # Katakana
    r"\uac00-\ud7af"  # Hangul (Korean)
    r"\u0600-\u06ff"  # Arabic
    r"\u0900-\u097f"  # Devanagari
    r"]"
)


def validate_cot(
    trace: str, example_id: str = ""
) -> bool:
    """Validate that CoT contains required sections."""
    if trace is None:
        _stats["no_response"] += 1
        print(f"[DISCARD] example_id={example_id} reason=no_response")
        return False

    if len(trace.strip()) < MIN_COT_LENGTH:
        _stats["too_short"] += 1
        print(
            f"[DISCARD] example_id={example_id} reason=too_short "
            f"len={len(trace.strip())}"
        )
        return False

    # Reject CoTs containing non-Latin characters (e.g., Chinese from Qwen)
    if _NON_LATIN_RE.search(trace):
        _stats["non_latin_chars"] += 1
        print(
            f"[DISCARD] example_id={example_id} reason=non_latin_chars"
        )
        return False

    trace_lower = trace.lower()
    found = [m for m in REQUIRED_MARKERS if m.lower() in trace_lower]

    # Require at least 2 of 3 parts
    if len(found) < 2:
        _stats["missing_parts"] += 1
        missing = [m for m in REQUIRED_MARKERS if m.lower() not in trace_lower]
        print(
            f"[DISCARD] example_id={example_id} reason=missing_parts "
            f"found={len(found)}/3 missing={missing}"
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


def process_cot_result(
    result: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
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
        "wer": metadata.get("wer", 0.0),
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
        processed_ids = get_processed_indices(
            output_file, idx_key="example_id"
        )
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
        example_id = sample.get("example_id", "")
        if example_id in processed_ids:
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
    final_count = len(
        get_processed_indices(output_file, idx_key="example_id")
    )
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
        description="Generate CoT with voice inference from persona."
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
