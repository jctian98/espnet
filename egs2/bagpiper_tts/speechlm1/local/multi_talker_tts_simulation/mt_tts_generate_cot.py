#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Stage 3: Generate chain-of-thought reasoning traces for multi-talker TTS."""

import argparse
import asyncio
import json
import os
import random
import re
from typing import Any, Dict, List, Optional

from sft_vllm_client import (
    DEFAULT_MODEL,
    MassiveQueryProcessor,
    get_processed_indices,
    parse_vllm_urls,
)

# =============================================================================
# Version-specific prompts for multi-talker CoT generation
# =============================================================================
PROMPTS_BY_VERSION = {
    "v1": {
        "system": (
            "You are a multi-talker text-to-speech system that thinks before "
            "generating speech. Given a user's TTS request for multiple "
            "speakers and the audio description that was produced, explain "
            "the reasoning process that led from the request to that audio "
            "output.\n\n"
            "Your reasoning MUST follow this 3-part structure:\n\n"
            "Part A - Intent Understanding:\n"
            "What voice characteristics is the user asking for EACH speaker? "
            "Identify the explicitly requested attributes (gender, age, "
            "emotion, accent, pace, etc.) for each speaker and note what is "
            "left unspecified that you need to decide.\n\n"
            "Part B - Speech Planning:\n"
            "How should the multi-speaker dialogue be delivered? Consider:\n"
            "- Turn-taking: speaker transitions and interaction dynamics\n"
            "- Which words to emphasize, where to place pauses\n"
            "- Emotional arc: does the emotion shift within or across turns?\n"
            "- Challenging elements: overlapping speech, interruptions, "
            "back-channels\n"
            "- Pacing and rhythm for the conversational flow\n\n"
            "Part C - Voice & Environment Characterization:\n"
            "What specific voice qualities for EACH speaker match the "
            "request? Consider per-speaker vocal warmth, brightness, "
            "roughness, breathiness, pitch range, and speaking style.\n\n"
            "RULES:\n"
            "- Output ONLY the reasoning trace as plain text\n"
            "- Do NOT copy or repeat the audio description\n"
            "- Keep the reasoning concise and focused\n"
            "- Describe voices in everyday language (e.g., 'warm and deep', "
            "'bright and energetic', 'soft and breathy') — do NOT use audio "
            "engineering jargon (e.g., 'reverb', 'reverberation', "
            "'close-miked', 'fundamental frequency', 'stereo field', "
            "'panning', 'sibilance', 'formant', 'compression', 'EQ', "
            "'proximity effect', 'high-fidelity', 'vocal fry', 'timbre', "
            "'resonance', 'subglottal', 'glottal')\n"
            "- Focus ONLY on speech characteristics — ignore any background "
            "music, electronic hums, hisses, sound effects, or recording "
            "artifacts mentioned in the audio description\n"
            "- Do NOT use markdown formatting (no **bold**, ## headers, "
            "> blockquotes, or --- dividers)"
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
# _stats tracks pass/fail counts; _stats_info tracks cleaning actions
# (separate denominators to avoid inflating pass/fail percentages).
_stats = {
    "success": 0,
    "no_response": 0,
    "too_short": 0,
    "missing_parts": 0,
}
_stats_info = {
    "has_markdown": 0,
}

REQUIRED_MARKERS = ["Part A", "Part B", "Part C"]

# Patterns to strip from responses
_MARKDOWN_RE = re.compile(r"\*\*|##+ |^>+ |^---+$", re.MULTILINE)


def validate_cot(trace: str, utt_id: str = "") -> Optional[str]:
    """Validate that CoT contains required sections.

    Returns cleaned trace on success, None on failure.
    """
    if trace is None:
        _stats["no_response"] += 1
        print(f"[DISCARD] utt_id={utt_id} reason=no_response")
        return None

    trace = trace.strip()

    # Strip markdown formatting (clean rather than reject)
    if _MARKDOWN_RE.search(trace):
        _stats_info["has_markdown"] += 1
        trace = _MARKDOWN_RE.sub("", trace)
        # Clean up residual blank lines from removed headers/dividers
        trace = re.sub(r"\n{3,}", "\n\n", trace).strip()

    if len(trace) < 50:
        _stats["too_short"] += 1
        print(
            f"[DISCARD] utt_id={utt_id} reason=too_short "
            f"len={len(trace)}"
        )
        return None

    trace_lower = trace.lower()
    missing = [m for m in REQUIRED_MARKERS if m.lower() not in trace_lower]

    if missing:
        _stats["missing_parts"] += 1
        print(
            f"[DISCARD] utt_id={utt_id} reason=missing_parts "
            f"missing={missing}"
        )
        return None

    _stats["success"] += 1
    return trace


def build_cot_query(
    sample: Dict[str, Any],
    prompts: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    """Build a query for CoT generation."""
    user_request = sample.get("user_request", "")
    caption = sample.get("rich_caption", "")
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
        "max_tokens": 2048,
        "metadata": sample,
    }


def process_cot_result(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process a CoT generation result."""
    response = result.get("response")
    metadata = result.get("metadata", {})
    utt_id = metadata.get("utt_id", "")

    cleaned = validate_cot(response, utt_id)
    if cleaned is None:
        return None

    return {
        "utt_id": metadata.get("utt_id", ""),
        "text": metadata.get("text", ""),
        "rich_caption": metadata.get("rich_caption", ""),
        "audio_path": metadata.get("audio_path", ""),
        "user_request": metadata.get("user_request", ""),
        "cot": cleaned,
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

    # Set random seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Random seed: {args.seed}")

    # Reset validation statistics
    for key in _stats:
        _stats[key] = 0
    for key in _stats_info:
        _stats_info[key] = 0

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
        # Info stats (not pass/fail, just informational)
        if any(_stats_info.values()):
            print()
            print("Cleaning actions:")
            for reason, count in _stats_info.items():
                print(f"  {reason:<25} {count:>8}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate CoT reasoning traces for multi-talker TTS examples."
        )
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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
