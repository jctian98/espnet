#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Stage 2: Generate intention-only user requests for ITS.

Unlike instruct-TTS, these requests convey communicative PURPOSE + voice
characteristics with ZERO transcription content. The model must later infer
the exact wording from the intention alone.
"""

import argparse
import asyncio
import json
import os
import random
import re
import string
from typing import Any, Dict, List, Optional, Set

from sft_vllm_client import (
    DEFAULT_MODEL,
    MassiveQueryProcessor,
    get_processed_indices,
    parse_vllm_urls,
)

# =============================================================================
# Stopwords for leakage detection (common English words to ignore)
# =============================================================================
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "as", "be", "was", "are",
    "were", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "can", "shall",
    "not", "no", "so", "if", "then", "that", "this", "these", "those",
    "i", "me", "my", "you", "your", "he", "she", "we", "they", "them",
    "his", "her", "its", "our", "their", "what", "which", "who", "whom",
    "how", "when", "where", "why", "all", "each", "every", "both",
    "few", "more", "most", "other", "some", "such", "than", "too",
    "very", "just", "about", "up", "out", "into", "over", "after",
    "before", "between", "through", "during", "without", "again",
    "also", "here", "there", "once", "while", "because", "until",
    "like", "make", "say", "said", "want", "need", "help", "please",
    "voice", "sound", "speak", "generate", "create", "produce",
}

# =============================================================================
# Version-specific prompts for ITS user request generation
# =============================================================================
PROMPTS_BY_VERSION = {
    "v1": {
        "system": (
            "You are a creative assistant that generates user requests for an "
            "intention-to-speech system. A real person is typing a request to "
            "a speech AI — they describe WHAT they want to communicate and HOW "
            "it should sound, but they do NOT provide the exact words.\n\n"
            "CRITICAL RULES:\n"
            "- Write as a REAL PERSON talking to an AI assistant, using first "
            "person (\"I need...\", \"Help me...\", \"Can you...\")\n"
            "- Do NOT write like a stage direction or task description\n"
            "- Do NOT include any specific words, phrases, or sentences from "
            "the transcription\n"
            "- Do NOT describe the content so precisely that only one phrasing "
            "is possible — keep the intention BROAD enough that multiple "
            "wordings could fulfill it\n"
            "- Keep it natural and conversational, 10-30 words\n"
            "- Output ONLY the user request\n\n"
            "BAD examples (too formal/descriptive):\n"
            "  'Convey practical advice for improving organization in a clear "
            "authoritative manner'\n"
            "  'Express awe and excitement about transformative progress'\n\n"
            "GOOD examples (natural, first-person):\n"
            "  'Help me give someone friendly tips about staying organized, "
            "in a warm casual voice'\n"
            "  'I need to sound amazed about something exciting that happened, "
            "like I can barely believe it'"
        ),
        "user": (
            "Rich audio caption:\n{caption}\n\n"
            "The transcription is: \"{transcription}\"\n"
            "(Do NOT include any of these words in your request! Keep the "
            "intention broad enough that the AI could phrase it many ways.)\n\n"
            "Style: {style}\n\n"
            "Write a natural user request:"
        ),
        "styles": [
            # First-person request styles
            "\"I need to...\" — state what you need to communicate",
            "\"Help me...\" — ask for help with a communication goal",
            "\"Can you...\" — ask the system to do something",
            "\"I want to...\" — express what you want to say",
            "\"I'd like...\" — politely request speech for a purpose",
            "\"Make me sound...\" — focus on how you want to come across",
            # Situation-based (still first-person)
            "\"I'm about to...\" — describe a situation you need speech for",
            "\"I have to...\" — describe an obligation to communicate",
            "\"I'm trying to...\" — describe a communication challenge",
            # Casual/direct
            "casual and direct — short, informal request",
            "enthusiastic — excited about what you want to say",
            "urgent — you need this speech quickly for something important",
        ],
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
_stats = {
    "success": 0,
    "no_response": 0,
    "too_short": 0,
    "too_long": 0,
    "transcription_leakage": 0,
}


def normalize_for_leakage(text: str) -> str:
    """Normalize text for leakage comparison."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_content_words(text: str) -> Set[str]:
    """Extract content words (non-stopwords) from normalized text."""
    normalized = normalize_for_leakage(text)
    words = set(normalized.split())
    return words - STOPWORDS


def check_transcription_leakage(
    request: str, transcription: str, threshold: float = 0.4
) -> bool:
    """Check if request leaks transcription content.

    Returns True if leakage detected (request should be rejected).
    """
    trans_content = get_content_words(transcription)
    if not trans_content:
        return False

    request_content = get_content_words(request)
    overlap = trans_content & request_content
    overlap_ratio = len(overlap) / len(trans_content)

    return overlap_ratio >= threshold


def validate_request(
    response: str, transcription: str, example_id: str = ""
) -> Optional[str]:
    """Validate the LLM-generated intention request."""
    if response is None:
        _stats["no_response"] += 1
        print(f"[DISCARD] example_id={example_id} reason=no_response")
        return None

    response = response.strip()

    # Remove surrounding quotes if present
    if response.startswith('"') and response.endswith('"'):
        response = response[1:-1].strip()

    # Check word count (10-50 words; prompt says 10-30 but allow some slack)
    word_count = len(response.split())
    if word_count < 10:
        _stats["too_short"] += 1
        print(
            f"[DISCARD] example_id={example_id} reason=too_short "
            f"words={word_count}\n  RESPONSE: {response}"
        )
        return None

    if word_count > 50:
        _stats["too_long"] += 1
        print(
            f"[DISCARD] example_id={example_id} reason=too_long "
            f"words={word_count}\n  RESPONSE: {response[:100]}..."
        )
        return None

    # Check transcription leakage
    if check_transcription_leakage(response, transcription):
        _stats["transcription_leakage"] += 1
        trans_content = get_content_words(transcription)
        req_content = get_content_words(response)
        overlap = trans_content & req_content
        print(
            f"[DISCARD] example_id={example_id} reason=transcription_leakage "
            f"overlap={overlap}\n  RESPONSE: {response[:100]}..."
        )
        return None

    _stats["success"] += 1
    return response


def build_request_query(
    sample: Dict[str, Any],
    prompts: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Build a query for intention request generation."""
    caption = sample.get("caption", "")
    transcription = sample.get("whisper_lyrics", "")
    example_id = sample.get("example_id", "")

    if not caption or not transcription:
        return None

    # Randomly select a style
    style = random.choice(prompts["styles"])

    messages = [
        {"role": "system", "content": prompts["system"]},
        {
            "role": "user",
            "content": prompts["user"].format(
                caption=caption,
                transcription=transcription,
                style=style,
            ),
        },
    ]

    # Store style metadata for output
    sample = dict(sample)
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
    transcription = metadata.get("whisper_lyrics", "")
    example_id = metadata.get("example_id", "")

    request = validate_request(response, transcription, example_id)
    if request is None:
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
        "user_request": request,
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
        description="Generate intention-only user requests for ITS."
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
