#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Stage 5: Assemble filtered TTS examples into dialogue format."""

import argparse
import json
import os
from typing import Any, Dict, List

# =============================================================================
# Version-specific system prompts
# =============================================================================
SYSTEM_PROMPTS_BY_VERSION = {
    "v1": "You are a text-to-speech system.",
}


def get_system_prompt(version: str) -> str:
    """Get system prompt for a specific version."""
    if version not in SYSTEM_PROMPTS_BY_VERSION:
        available = list(SYSTEM_PROMPTS_BY_VERSION.keys())
        raise ValueError(
            f"Version '{version}' not found. Available versions: {available}"
        )
    return SYSTEM_PROMPTS_BY_VERSION[version]


def format_assistant_text(cot: str, caption: str) -> str:
    """Format assistant text with thinking tokens."""
    return f"<think>\n{cot.strip()}\n</think>\n\n{caption.strip()}"


def create_example_id(sample: Dict[str, Any]) -> str:
    """Create a unique example ID."""
    utt_id = sample.get("utt_id", "unknown")
    subset = sample.get("subset", "unknown")
    return f"instruct_tts_{subset}_{utt_id}"


def create_dialogue(
    sample: Dict[str, Any],
    system_prompt: str,
) -> Dict[str, Any]:
    """Create dialogue format from sample.

    Output format follows the canonical dialogue format:
    [(role, modality, content), ...]
    """
    user_request = sample.get("user_request", "")
    cot = sample.get("cot", "")
    caption = sample.get("caption", "")
    audio_path = sample.get("audio_path", "")

    messages = [
        ["system", "text", system_prompt],
        ["user", "text", user_request],
        [
            "assistant",
            "text",
            format_assistant_text(cot, caption),
        ],
        ["assistant", "audio", audio_path],
    ]

    return {
        "example_id": create_example_id(sample),
        "messages": messages,
        "metadata": {
            "utt_id": sample.get("utt_id", ""),
            "subset": sample.get("subset", ""),
            "speaker_id": sample.get("speaker_id", ""),
            "duration": sample.get("duration", 0.0),
        },
    }


def validate_sample(sample: Dict[str, Any]) -> bool:
    """Validate that sample has all required fields."""
    required_fields = [
        "user_request",
        "cot",
        "caption",
        "audio_path",
    ]
    return all(
        sample.get(field) and len(str(sample.get(field)).strip()) > 0
        for field in required_fields
    )


def load_input_data(input_file: str) -> List[Dict[str, Any]]:
    """Load input JSONL file."""
    samples = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Assemble TTS dialogue from filtered examples."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input JSONL (stage 4 filtered output).",
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
        help="System prompt version to use.",
    )
    args = parser.parse_args()

    # Get version-specific system prompt
    system_prompt = get_system_prompt(args.version)
    print(f"Using system prompt version: {args.version}")

    # Load input data
    print(f"Loading input data from {args.input_file}...")
    samples = load_input_data(args.input_file)
    print(f"Loaded {len(samples)} samples")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "dialogues.jsonl")

    # Process samples
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
    print(f"Output saved to: {output_file}")

    # Write assembly summary
    summary = {
        "total_input": len(samples),
        "valid_assembled": valid_count,
        "invalid_skipped": invalid_count,
    }
    summary_file = os.path.join(args.output_dir, "assembly_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
