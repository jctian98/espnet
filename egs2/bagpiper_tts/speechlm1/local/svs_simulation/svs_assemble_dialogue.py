#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Stage 5: Assemble filtered SVS examples into dialogue format."""

import argparse
import json
import os
from typing import Any, Dict, List

# =============================================================================
# Version-specific system prompts for SVS
# =============================================================================
SYSTEM_PROMPTS_BY_VERSION = {
    "v1": (
        "You are an advanced singing voice synthesis (SVS) system. "
        "Given a user request that specifies the desired singing style, "
        "vocal character, and lyrics, you synthesize a high-quality "
        "vocal performance that faithfully renders the requested lyrics "
        "with the appropriate voice, emotion, and artistic expression."
    ),
}


def get_system_prompt(version: str) -> str:
    if version not in SYSTEM_PROMPTS_BY_VERSION:
        available = list(SYSTEM_PROMPTS_BY_VERSION.keys())
        raise ValueError(
            f"Version '{version}' not found. Available: {available}"
        )
    return SYSTEM_PROMPTS_BY_VERSION[version]


def format_assistant_text(cot: str, caption: str) -> str:
    """Wrap CoT in <think> tags followed by the rich caption."""
    return f"<think>\n{cot.strip()}\n</think>\n\n{caption.strip()}"


def create_example_id(sample: Dict[str, Any]) -> str:
    dataset = sample.get("dataset", "svs")
    example_id = sample.get("example_id", "unknown")
    return f"svs_{dataset}_{example_id}"


def create_dialogue(
    sample: Dict[str, Any],
    system_prompt: str,
) -> Dict[str, Any]:
    """Build canonical dialogue format for SVS.

    Structure:
        [system, text, <system prompt>]
        [user,   text, <user SVS request with lyrics>]
        [assistant, text, <think>CoT</think>\\n\\n<rich caption>]
        [assistant, audio, <path to singing audio>]
    """
    user_request = sample.get("user_request", "")
    cot = sample.get("cot", "")
    caption = sample.get("caption", "")
    audio_path = sample.get("audio_path", "")

    messages = [
        ["system", "text", system_prompt],
        ["user", "text", user_request],
        ["assistant", "text", format_assistant_text(cot, caption)],
        ["assistant", "audio", audio_path],
    ]

    return {
        "example_id": create_example_id(sample),
        "messages": messages,
        "metadata": {
            "original_id": sample.get("example_id", ""),
            "dataset": sample.get("dataset", ""),
            "duration": sample.get("duration", 0.0),
            "whisper_lyrics": sample.get("whisper_lyrics", ""),
            "extracted_lyrics": sample.get("extracted_lyrics", ""),
        },
    }


def validate_sample(sample: Dict[str, Any]) -> bool:
    """Check that all required fields are non-empty."""
    required = ["user_request", "cot", "caption", "audio_path"]
    return all(
        sample.get(f) and len(str(sample.get(f)).strip()) > 0
        for f in required
    )


def load_input_data(input_file: str) -> List[Dict[str, Any]]:
    samples = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 5: Assemble SVS dialogues from filtered examples."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Stage 4 filtered JSONL.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for assembled dialogues.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="System prompt version.",
    )
    args = parser.parse_args()

    system_prompt = get_system_prompt(args.version)
    print(f"Using system prompt version: {args.version}")

    print(f"Loading input data from: {args.input_file}")
    samples = load_input_data(args.input_file)
    print(f"Loaded {len(samples)} samples")

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "dialogues.jsonl")

    valid_count = 0
    invalid_count = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for sample in samples:
            if not validate_sample(sample):
                invalid_count += 1
                continue
            dialogue = create_dialogue(sample, system_prompt)
            f.write(json.dumps(dialogue, ensure_ascii=False) + "\n")
            valid_count += 1

    print(f"Done! Valid: {valid_count}, Invalid: {invalid_count}")
    print(f"Output: {output_file}")

    summary = {
        "total_input": len(samples),
        "valid_assembled": valid_count,
        "invalid_skipped": invalid_count,
    }
    summary_file = os.path.join(args.output_dir, "assembly_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()
