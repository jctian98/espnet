#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Stage 2: Generate multi-talker user requests from rich captions.

Unlike single-speaker which uses a [TEXT] placeholder pattern, multi-talker
requests have the LLM read the caption directly and generate a request that
attributes each speaker's transcription clearly. Validation uses word overlap
(>= 80%) instead of exact placeholder matching.
"""

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
# Version-specific prompts for multi-talker TTS user request generation
# =============================================================================
PROMPTS_BY_VERSION = {
    "v1": {
        "system": (
            "You are a creative assistant that generates diverse user "
            "requests for a multi-talker text-to-speech system. Given a "
            "rich audio caption describing a multi-speaker recording, "
            "generate a user request that:\n\n"
            "1. Describes each speaker's voice characteristics in "
            "everyday language\n"
            "2. Includes the EXACT quoted speech for each speaker, "
            "clearly attributed\n"
            "3. Follows the specified style and structure\n\n"
            "RULES:\n"
            "- Output ONLY the user request as plain text\n"
            "- Include ALL quoted spoken text from the caption verbatim "
            "— every word must appear exactly as quoted\n"
            "- Clearly attribute which speaker says what\n"
            "- Describe voices in everyday terms (e.g., 'warm, deep "
            "voice', 'cheerful and bright') — do NOT use audio "
            "engineering jargon (e.g., 'reverb', 'close-miked', "
            "'reverberation', 'frequency', 'high-fidelity', 'EQ', "
            "'low-frequency hum', 'room tone')\n"
            "- Do NOT request background music, environmental sounds, "
            "sound effects, or any non-speech audio — focus only on "
            "the speakers and their words\n"
            "- Do NOT use markdown formatting (no **bold**, ## headers, "
            "> blockquotes, or --- dividers)\n"
            "- Pick the most relevant voice traits from the caption, "
            "do NOT always mention the same ones\n"
            "- Sound like a real person would talk to a TTS system\n"
            "- Follow both the style and structure instructions closely"
        ),
        "user": (
            "Rich audio caption:\n{caption}\n\n"
            "Style: {style}\n"
            "Structure: {structure}\n"
            "Length: {length}\n\n"
            "Generate a multi-talker TTS user request:"
        ),
        "styles": [
            # Request-style
            "polite request — ask nicely for the multi-speaker speech",
            "need statement — express what you need produced",
            "want statement — state what you want to hear",
            "help request — ask for help generating this conversation",
            # Instruction-style
            "imperative — directly command the system",
            "speaking command — tell the system to have each speaker "
            "say their lines",
            "make command — tell the system to make it sound a "
            "certain way",
            # Description-style
            "descriptive — paint the scene and voices vividly",
            "contextual — add situational context around the dialogue",
            "scenario — set up an imaginary situation for the speakers",
            # Format-style (multi-talker specific)
            "dialogue script — format as a dialogue with stage "
            "directions",
            "production script — format as a production brief with "
            "speaker cues",
            "podcast — frame as a podcast episode production request",
            "interview — frame as an interview or panel setup",
            # Minimal-style
            "direct label — brief and minimal, just speaker tags and "
            "their lines",
            "annotation — short voice notes attached to each "
            "speaker's text",
            # Question-style
            "questioning — ask how this conversation would sound",
            "what-if — pose a hypothetical: what if these speakers "
            "had this exchange",
        ],
        "structures": [
            "speakers-first — describe ALL speakers' voices first, "
            "then list their lines in order",
            "interleaved — for each speaker in turn, describe their "
            "voice then give their text",
            "scene-setup — set the scene and context first, then "
            "introduce each speaker and their dialogue",
            "script-format — format like a screenplay with character "
            "names and stage directions",
        ],
        "lengths": [
            "short — keep voice descriptions very brief (a few words "
            "per speaker), just enough to distinguish them",
            "medium — moderate voice descriptions (1-2 sentences per "
            "speaker)",
            "detailed — rich voice descriptions (2-3 sentences per "
            "speaker) with personality and emotion",
        ],
        # Styles that should always use "short" length
        "short_styles": {"direct label", "annotation"},
    },
}


def get_prompts(version: str) -> Dict[str, Any]:
    """Get prompts for a specific version."""
    if version not in PROMPTS_BY_VERSION:
        available = list(PROMPTS_BY_VERSION.keys())
        raise ValueError(
            f"Version '{version}' not found. Available versions: {available}"
        )
    return PROMPTS_BY_VERSION[version]


# Validation statistics (reset per run in main_async)
# "has_markdown" is tracked separately — it's a cleaning action, not a
# discard reason, so it must not inflate the pass/fail denominator.
_stats = {
    "success": 0,
    "no_response": 0,
    "low_word_overlap": 0,
    "too_short": 0,
}
_stats_info = {
    "has_markdown": 0,
}

# Patterns to strip from responses
_MARKDOWN_RE = re.compile(r"\*\*|##+ |^>+ |^---+$", re.MULTILINE)


def compute_word_overlap(
    ground_truth: str, user_request: str
) -> float:
    """Compute fraction of ground truth words found in user request."""
    gt_words = set(re.sub(r"[^a-z0-9\s]", " ", ground_truth.lower()).split())
    req_words = set(
        re.sub(r"[^a-z0-9\s]", " ", user_request.lower()).split()
    )

    if not gt_words:
        return 1.0

    overlap = gt_words & req_words
    return len(overlap) / len(gt_words)


def validate_user_request(
    response: str, ground_truth: str, utt_id: str = ""
) -> Optional[tuple]:
    """Validate multi-talker user request.

    Returns (cleaned_response, word_overlap) on success, None on failure.
    """
    if response is None:
        _stats["no_response"] += 1
        print(f"[DISCARD] utt_id={utt_id} reason=no_response")
        return None

    response = response.strip()

    # Remove surrounding quotes if present
    if response.startswith('"') and response.endswith('"'):
        response = response[1:-1].strip()

    # Strip markdown formatting (clean rather than reject)
    if _MARKDOWN_RE.search(response):
        _stats_info["has_markdown"] += 1
        response = _MARKDOWN_RE.sub("", response)
        # Clean up residual blank lines from removed headers/dividers
        response = re.sub(r"\n{3,}", "\n\n", response).strip()

    if len(response) < 20:
        _stats["too_short"] += 1
        print(
            f"[DISCARD] utt_id={utt_id} reason=too_short "
            f"len={len(response)}"
        )
        return None

    # Check word overlap with ground truth (>= 80%)
    overlap = compute_word_overlap(ground_truth, response)
    if overlap < 0.8:
        _stats["low_word_overlap"] += 1
        print(
            f"[DISCARD] utt_id={utt_id} reason=low_word_overlap "
            f"overlap={overlap:.2f}"
        )
        return None

    _stats["success"] += 1
    return response, overlap


def build_request_query(
    sample: Dict[str, Any],
    prompts: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Build a query for multi-talker user request generation."""
    caption = sample.get("rich_caption", "")
    text = sample.get("text", "")
    utt_id = sample.get("utt_id", "")

    if not caption or not text:
        return None

    style = random.choice(prompts["styles"])
    structure = random.choice(prompts["structures"])

    # Force short length for minimal styles, random otherwise
    style_name = style.split(" — ")[0]
    short_styles = prompts.get("short_styles", set())
    if style_name in short_styles:
        length = prompts["lengths"][0]  # "short"
    else:
        length = random.choice(prompts["lengths"])

    messages = [
        {"role": "system", "content": prompts["system"]},
        {
            "role": "user",
            "content": prompts["user"].format(
                caption=caption,
                style=style,
                structure=structure,
                length=length,
            ),
        },
    ]

    # Store style/structure/length metadata for output
    sample = dict(sample)
    sample["prompt_style"] = style
    sample["prompt_structure"] = structure
    sample["prompt_length"] = length

    return {
        "idx": utt_id,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 2048,
        "metadata": sample,
    }


def process_request_result(
    result: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Process a multi-talker user request generation result."""
    response = result.get("response")
    metadata = result.get("metadata", {})
    ground_truth = metadata.get("text", "")
    utt_id = metadata.get("utt_id", "")

    validated = validate_user_request(response, ground_truth, utt_id)
    if validated is None:
        return None

    user_request, word_overlap = validated

    return {
        "utt_id": utt_id,
        "text": ground_truth,
        "rich_caption": metadata.get("rich_caption", ""),
        "audio_path": metadata.get("audio_path", ""),
        "user_request": user_request,
        "word_overlap": round(word_overlap, 4),
        "prompt_style": metadata.get("prompt_style", ""),
        "prompt_structure": metadata.get("prompt_structure", ""),
        "prompt_length": metadata.get("prompt_length", ""),
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

    # Load input data (stage 1 filtered output)
    print(f"Loading input data from {args.input_file}...")
    samples = load_input_data(args.input_file)
    print(f"Loaded {len(samples)} samples")

    # Limit samples if specified
    if args.num_samples > 0:
        samples = samples[: args.num_samples]
        print(f"Limited to {len(samples)} samples")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "user_requests.jsonl")

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
        query = build_request_query(sample, prompts)
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
        process_fn=process_request_result,
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
            "Generate multi-talker TTS user requests from rich captions."
        )
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input JSONL (stage 1 filtered output).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for user requests.",
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
