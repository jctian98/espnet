#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Stage 2: Generate user requests for instruct TTS from rich captions."""

import argparse
import asyncio
import json
import os
import random
from typing import Any, Dict, List, Optional

from sft_vllm_client import (
    DEFAULT_MODEL,
    MassiveQueryProcessor,
    get_processed_indices,
    parse_vllm_urls,
)

# =============================================================================
# Version-specific prompts for TTS user request generation
# =============================================================================
PROMPTS_BY_VERSION = {
    "v1": {
        "system": (
            "You are a creative assistant that generates diverse user "
            "requests for a text-to-speech system. Given a rich audio "
            "caption and specific style instructions, generate a user "
            "request that follows those instructions.\n\n"
            "Use [TEXT] as a placeholder for the transcription. Include "
            "[TEXT] exactly once.\n\n"
            "RULES:\n"
            "- Output ONLY the user request\n"
            "- Include [TEXT] exactly once\n"
            "- Keep the description part SHORT (5-25 words, not counting "
            "[TEXT])\n"
            "- Do NOT use technical audio jargon (e.g., 'reverb', 'EQ', "
            "'compression', 'frequency')\n"
            "- Sound like a real person would talk to a TTS system\n"
            "- Follow the style and position instructions closely"
        ),
        "user": (
            "Rich audio caption:\n{caption}\n\n"
            "Style instructions:\n"
            "- Position: {position}\n"
            "- Wording style: {style}\n\n"
            "Pick the most natural and relevant voice attributes from the "
            "caption. Do NOT always mention the same ones.\n\n"
            "Generate the user request:"
        ),
        "positions": [
            "Place [TEXT] at the BEGINNING of the request",
            "Place [TEXT] in the MIDDLE of the request",
            "Place [TEXT] at the END of the request",
        ],
        "styles": [
            # Request-style
            "polite request — ask nicely for the speech",
            "need statement — express what you need",
            "want statement — state what you want",
            "help request — ask for help generating speech",
            # Instruction-style
            "imperative — directly command the system",
            "speaking command — tell the system to speak or say",
            "make command — tell the system to make it sound a certain way",
            # Description-style
            "descriptive — describe the voice as a scene",
            "contextual — add context around the text",
            "scenario — set up an imaginary situation",
            # Minimal-style
            "direct label — brief, tag-like, minimal words",
            "annotation — attach voice traits as a short note",
            # Question-style
            "questioning — ask how it would sound",
            "what-if — pose a hypothetical",
        ],
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


PLACEHOLDER = "[TEXT]"

# Validation statistics (reset per run in main_async)
_stats = {
    "success": 0,
    "no_response": 0,
    "bad_placeholder": 0,
    "too_short": 0,
    "too_long": 0,
}


def validate_request_template(
    response: str, utt_id: str = ""
) -> Optional[str]:
    """Validate the LLM-generated request template containing [TEXT]."""
    if response is None:
        _stats["no_response"] += 1
        print(f"[DISCARD] utt_id={utt_id} reason=no_response")
        return None

    response = response.strip()

    # Remove surrounding quotes if present
    if response.startswith('"') and response.endswith('"'):
        response = response[1:-1].strip()

    # Check that [TEXT] appears exactly once
    count = response.count(PLACEHOLDER)
    if count != 1:
        _stats["bad_placeholder"] += 1
        print(
            f"[DISCARD] utt_id={utt_id} reason=bad_placeholder "
            f"count={count}\n  RESPONSE: {response}"
        )
        return None

    # Check length of the description part (excluding placeholder)
    desc_len = len(response) - len(PLACEHOLDER)
    if desc_len < 5:
        _stats["too_short"] += 1
        print(
            f"[DISCARD] utt_id={utt_id} reason=too_short "
            f"desc_len={desc_len}\n  RESPONSE: {response}"
        )
        return None

    if desc_len > 500:
        _stats["too_long"] += 1
        print(
            f"[DISCARD] utt_id={utt_id} reason=too_long "
            f"desc_len={desc_len}\n  RESPONSE: {response}"
        )
        return None

    _stats["success"] += 1
    return response


def build_request_query(
    sample: Dict[str, Any],
    prompts: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Build a query for user request generation."""
    caption = sample.get("caption", "")
    text = sample.get("text", "")
    utt_id = sample.get("utt_id", "")

    if not caption or not text:
        return None

    # Randomly select one from each slot
    position = random.choice(prompts["positions"])
    style = random.choice(prompts["styles"])

    messages = [
        {"role": "system", "content": prompts["system"]},
        {
            "role": "user",
            "content": prompts["user"].format(
                caption=caption,
                position=position,
                style=style,
            ),
        },
    ]

    # Store style metadata for output
    sample = dict(sample)
    sample["prompt_position"] = position
    sample["prompt_style"] = style

    return {
        "idx": utt_id,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 512,
        "metadata": sample,
    }


def process_request_result(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process a user request generation result."""
    response = result.get("response")
    metadata = result.get("metadata", {})
    original_text = metadata.get("text", "")
    utt_id = metadata.get("utt_id", "")

    template = validate_request_template(response, utt_id)
    if template is None:
        return None

    # Substitute placeholder with the exact transcription
    # Strip edge quotes from original text to avoid double-quoting
    # (e.g., original text '"Very good.' would become '""Very good."')
    clean_text = original_text.strip('"')
    user_request = template.replace(PLACEHOLDER, f'"{clean_text}"')
    # Collapse any remaining "" from LLM template already quoting [TEXT]
    user_request = user_request.replace('""', '"')

    return {
        "utt_id": utt_id,
        "text": original_text,
        "caption": metadata.get("caption", ""),
        "audio_path": metadata.get("audio_path", ""),
        "speaker_id": metadata.get("speaker_id", ""),
        "duration": metadata.get("duration", 0.0),
        "subset": metadata.get("subset", ""),
        "user_request": user_request,
        "prompt_position": metadata.get("prompt_position", ""),
        "prompt_style": metadata.get("prompt_style", ""),
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

    # Reset validation statistics
    for key in _stats:
        _stats[key] = 0

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
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate TTS user requests from rich captions."
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
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
