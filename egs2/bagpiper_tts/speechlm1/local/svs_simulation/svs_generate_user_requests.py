#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Stage 2: Generate user requests for SVS from rich singing captions."""

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
# Version-specific prompts for SVS user request generation
# =============================================================================
PROMPTS_BY_VERSION = {
    "v1": {
        "system": (
            "You are a creative assistant that generates diverse user "
            "requests for a singing voice synthesis (SVS) system. Given a "
            "rich audio caption describing a singing recording, generate a "
            "natural user request asking the SVS system to synthesize that "
            "performance.\n\n"
            "Use [LYRICS] as a placeholder for the song lyrics. Include "
            "[LYRICS] exactly once.\n\n"
            "RULES:\n"
            "- Output ONLY the user request\n"
            "- Include [LYRICS] exactly once\n"
            "- Keep the description part concise (5-30 words, not counting "
            "[LYRICS])\n"
            "- Focus on SINGING attributes: vocal style, genre, emotion, "
            "timbre, energy, performance character\n"
            "- If the caption mentions non-singing elements (background "
            "music, instrumental accompaniment, intro/outro without "
            "vocals), you may SOMETIMES include them in the request to "
            "add realism. Not every request needs this — vary it.\n"
            "- Do NOT use audio production jargon (e.g., 'reverb', 'EQ', "
            "'compression', 'frequency response'). Singing technique "
            "terms like vibrato, falsetto, belt, head voice are fine.\n"
            "- Sound like a real person asking an SVS system to sing\n"
            "- Follow the style and position instructions closely"
        ),
        "user": (
            "Rich singing audio caption:\n{caption}\n\n"
            "Style instructions:\n"
            "- Position: {position}\n"
            "- Wording style: {style}\n\n"
            "Pick the most natural and relevant attributes from the "
            "caption (e.g. gender, vocal range, genre, emotion, energy, "
            "background music, instrumental intro/outro). "
            "Do NOT always mention the same ones.\n\n"
            "Generate the user request:"
        ),
        "positions": [
            "Place [LYRICS] at the BEGINNING of the request",
            "Place [LYRICS] in the MIDDLE of the request",
            "Place [LYRICS] at the END of the request",
        ],
        "styles": [
            # Request-style
            "polite request — ask nicely for the singing",
            "need statement — express what you need",
            "want statement — state what you want",
            "help request — ask for help generating the singing",
            # Instruction-style
            "imperative — directly command the system to sing",
            "singing command — tell the system to perform or render",
            "make command — tell the system to make it sound a certain way",
            # Description-style
            "descriptive — describe the voice and style as a scene",
            "contextual — add context around the performance",
            "scenario — set up an imaginary performance situation",
            # Minimal-style
            "direct label — brief, tag-like, minimal words",
            "annotation — attach voice and style traits as a short note",
            # Question-style
            "questioning — ask how it would sound if sung",
            "what-if — pose a hypothetical singing scenario",
        ],
    },
}


def get_prompts(version: str) -> Dict[str, Any]:
    if version not in PROMPTS_BY_VERSION:
        available = list(PROMPTS_BY_VERSION.keys())
        raise ValueError(
            f"Version '{version}' not found. Available: {available}"
        )
    return PROMPTS_BY_VERSION[version]


PLACEHOLDER = "[LYRICS]"

MIN_DESC_WORDS = 5
MAX_DESC_WORDS = 45


def _new_stats() -> Dict[str, int]:
    return {
        "success": 0,
        "no_response": 0,
        "bad_placeholder": 0,
        "too_short": 0,
        "too_long": 0,
    }


_stats = _new_stats()


def validate_request_template(
    response: str, example_id: str = ""
) -> Optional[str]:
    """Validate the LLM-generated request template containing [LYRICS]."""
    if response is None:
        _stats["no_response"] += 1
        # print(f"[DISCARD] id={example_id} reason=no_response")
        return None

    response = response.strip()

    # Remove surrounding quotes if any
    if response.startswith('"') and response.endswith('"'):
        response = response[1:-1].strip()

    count = response.count(PLACEHOLDER)
    if count != 1:
        _stats["bad_placeholder"] += 1
        # print(
        #     f"[DISCARD] id={example_id} reason=bad_placeholder "
        #     f"count={count}\n  RESPONSE: {response}"
        # )
        return None

    # Count words in the description (everything except the placeholder)
    desc_text = response.replace(PLACEHOLDER, "")
    desc_words = len(desc_text.split())
    if desc_words < MIN_DESC_WORDS:
        _stats["too_short"] += 1
        # print(
        #     f"[DISCARD] id={example_id} reason=too_short "
        #     f"desc_words={desc_words}\n  RESPONSE: {response}"
        # )
        return None

    if desc_words > MAX_DESC_WORDS:
        _stats["too_long"] += 1
        # print(
        #     f"[DISCARD] id={example_id} reason=too_long "
        #     f"desc_words={desc_words}\n  RESPONSE: {response}"
        # )
        return None

    _stats["success"] += 1
    return response


def build_request_query(
    sample: Dict[str, Any],
    prompts: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Build a query for SVS user request generation."""
    caption = sample.get("caption", "")
    example_id = sample.get("example_id", "")

    if not caption:
        return None

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

    sample = dict(sample)
    sample["prompt_position"] = position
    sample["prompt_style"] = style

    return {
        "idx": example_id,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 512,
        "metadata": sample,
    }


def process_request_result(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process a user request generation result."""
    response = result.get("response")
    metadata = result.get("metadata", {})
    lyrics = metadata.get("whisper_lyrics", metadata.get("extracted_lyrics", ""))
    example_id = metadata.get("example_id", "")

    template = validate_request_template(response, example_id)
    if template is None:
        return None

    # Substitute placeholder with the extracted lyrics (quoted)
    clean_lyrics = lyrics.strip().strip('"')
    user_request = template.replace(PLACEHOLDER, f'"{clean_lyrics}"')

    return {
        "example_id": example_id,
        "dataset": metadata.get("dataset", ""),
        "audio_path": metadata.get("audio_path", ""),
        "duration": metadata.get("duration", 0.0),
        "caption": metadata.get("caption", ""),
        "whisper_lyrics": metadata.get("whisper_lyrics", ""),
        "extracted_lyrics": metadata.get("extracted_lyrics", ""),
        "user_request": user_request,
        "prompt_position": metadata.get("prompt_position", ""),
        "prompt_style": metadata.get("prompt_style", ""),
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
    output_file = os.path.join(args.output_dir, "user_requests.jsonl")

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
        description="Stage 2: Generate SVS user requests from singing captions."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Stage 1 filtered JSONL.",
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
