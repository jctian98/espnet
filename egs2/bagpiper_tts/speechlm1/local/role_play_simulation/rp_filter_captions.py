#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Stage 1: Score captions for persona suitability.

LLM scores each rich caption on a 1-5 scale for how well it provides
diverse speaker metadata to derive an effective fictional persona.
Multi-speaker captions get score=1. All scored results are saved;
post-filtering by --min_score threshold produces the filtered output.
"""

import argparse
import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional, Set

from sft_vllm_client import (
    DEFAULT_MODEL,
    MassiveQueryProcessor,
    get_processed_indices,
    parse_vllm_urls,
)

# =============================================================================
# Prompts for persona suitability scoring
# =============================================================================
PROMPTS_BY_VERSION = {
    "v1": {
        "system": (
            "You are a text classification assistant. Given a rich audio "
            "caption describing a speech recording, score how well it "
            "provides diverse speaker metadata to derive a fictional "
            "character persona.\n\n"
            "A good caption for persona derivation should describe a SINGLE "
            "speaker with multiple character-relevant traits such as gender, "
            "age, accent, emotion, personality, speaking style, or vocal "
            "quality.\n\n"
            "Scoring scale:\n"
            "- 5 (Excellent): Rich single-speaker caption with multiple "
            "character traits (gender, age, accent, emotion, personality, "
            "style). Easy to build a vivid fictional persona from.\n"
            "- 4 (Good): Single speaker with several useful traits. Enough "
            "to create a believable persona with some creative inference.\n"
            "- 3 (Acceptable): Single speaker but sparse or generic traits. "
            "Only basic attributes like gender or general tone.\n"
            "- 2 (Poor): Mostly technical or recording description with "
            "minimal speaker characterization. Hard to derive a persona.\n"
            "- 1 (Unsuitable): Multiple speakers, no speech content, "
            "non-informative caption, OR the caption says the audio ENDS "
            "abruptly / cuts off mid-word / is truncated at the END. "
            "NOTE: it is FINE if the caption says the audio BEGINS "
            "mid-sentence or starts abruptly — only penalize if the "
            "ENDING is incomplete or cut off.\n\n"
            "Output ONLY valid JSON: "
            '{"score": <1-5>, "reason": "brief explanation"}'
        ),
        "user": (
            "Score this audio caption for persona derivation suitability "
            "(1-5):\n\n"
            "Caption: \"{caption}\""
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


# Words that almost never end a complete sentence — strong truncation signal.
# Excludes words like "do", "about", "too", "so" that CAN end sentences.
# Regex patterns that indicate the speech content is genuinely INCOMPLETE —
# the speaker was cut off mid-utterance, leaving words/sentences unfinished.
# We do NOT match "ends abruptly" alone (which often just means a clean
# digital edit with no fade-out — perfectly fine for TTS training).
_CAPTION_CUTOFF_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        # Speech stops mid-word/sentence/thought
        r"(?:ends?|stops?|cuts?)\s+(?:off\s+)?mid[\s-]"
        r"(?:word|sentence|thought|phrase|syllable|utterance)",
        # Explicit truncation language
        r"(?:sentence|thought|speech|word|utterance)\s+"
        r"(?:left\s+)?(?:unfinished|incomplete)",
        r"(?:leaving|left)\s+(?:the\s+)?(?:thought|sentence|speech|word)\s+"
        r"(?:unfinished|incomplete|hanging)",
        # Cut off specific words
        r"cutting\s+off\s+(?:the\s+)?(?:final|last)",
        r"(?:syllable|word)\s+is\s+cut\s+off",
        # Breaks off mid-speech
        r"(?:breaks?\s+off|cut\s+short)\s+(?:mid[\s-]|before\s+(?:finish|complet))",
        # Truncation
        r"truncat(?:ed|es|ion)\b",
    ]
]


def has_caption_cutoff(caption: str) -> bool:
    """Check if caption describes audio that ENDS abruptly or is incomplete.

    Does NOT match captions that merely describe audio BEGINNING abruptly
    (which is fine — it just means a clean start without fade-in).
    """
    for pattern in _CAPTION_CUTOFF_PATTERNS:
        if pattern.search(caption):
            return True
    return False


_TRAILING_FUNCTION_WORDS = frozenset({
    "the", "a", "an", "of", "to", "in", "for", "and", "but", "or", "that",
    "which", "is", "are", "was", "were", "be", "being", "with", "as", "at",
    "by", "from", "on", "into", "than", "if", "its", "this",
    "these", "those", "their", "our", "your", "my", "his", "her", "who",
    "whom", "whose", "where", "when", "because", "since",
    "between", "through", "during", "above", "below",
    "can", "could", "would", "should", "will", "shall", "may", "might",
    "must", "has", "have", "had", "does", "did", "very",
    "just", "then", "some", "any", "each", "every",
    "such", "nor", "only", "own", "same", "another",
})


def is_complete_transcription(text: str) -> bool:
    """Check if a transcription is a complete sentence (not truncated).

    Returns False if the text ends with a function word (preposition,
    article, conjunction, etc.) indicating it was cut off mid-sentence.
    """
    if not text or not text.strip():
        return False

    stripped = text.strip()

    # Strip trailing punctuation to get the last word
    last_word = re.sub(r'[.!?,;:\'")\]\-]+$', "", stripped).split()
    if not last_word:
        return False

    return last_word[-1].lower() not in _TRAILING_FUNCTION_WORDS


# Validation statistics (reset per run in main_async)
_stats = {
    "success": 0,
    "no_response": 0,
    "bad_json": 0,
    "bad_score_range": 0,
    "incomplete_transcription": 0,
}


def parse_score_response(response: str, example_id: str = "") -> Optional[Dict]:
    """Parse JSON score response from LLM."""
    if response is None:
        _stats["no_response"] += 1
        return None

    data = None
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
        except json.JSONDecodeError:
            pass

    if data is None or "score" not in data:
        _stats["bad_json"] += 1
        print(
            f"[DISCARD] example_id={example_id} reason=bad_json "
            f"response={response[:100]}"
        )
        return None

    score = data["score"]
    if not isinstance(score, (int, float)) or score < 1 or score > 5:
        _stats["bad_score_range"] += 1
        print(
            f"[DISCARD] example_id={example_id} reason=bad_score_range "
            f"score={score}"
        )
        return None

    return data


def build_filter_query(
    sample: Dict[str, Any],
    prompts: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    """Build a query for persona suitability scoring."""
    caption = sample.get("caption", "")
    example_id = sample.get("example_id", "")

    if not caption or len(caption.strip()) < 10:
        return None

    # Pre-filter: reject truncated transcriptions
    whisper_lyrics = sample.get("whisper_lyrics", "")
    if not is_complete_transcription(whisper_lyrics):
        _stats["incomplete_transcription"] += 1
        return None

    messages = [
        {"role": "system", "content": prompts["system"]},
        {
            "role": "user",
            "content": prompts["user"].format(caption=caption),
        },
    ]

    return {
        "idx": example_id,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 256,
        "json_mode": True,
        "metadata": sample,
    }


def process_filter_result(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process a scoring result."""
    response = result.get("response")
    metadata = result.get("metadata", {})
    example_id = metadata.get("example_id", "")

    parsed = parse_score_response(response, example_id)
    if parsed is None:
        return None

    _stats["success"] += 1

    return {
        "example_id": example_id,
        "dataset": metadata.get("dataset", ""),
        "audio_path": metadata.get("audio_path", ""),
        "duration": metadata.get("duration", 0.0),
        "caption": metadata.get("caption", ""),
        "whisper_lyrics": metadata.get("whisper_lyrics", ""),
        "extracted_lyrics": metadata.get("extracted_lyrics", ""),
        "wer": metadata.get("wer", 0.0),
        "persona_score": int(parsed["score"]),
        "reason": parsed.get("reason", ""),
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

    # Load input data
    print(f"Loading input data from {args.input_file}...")
    samples = load_input_data(args.input_file)
    print(f"Loaded {len(samples)} samples")

    # Limit samples if specified
    if args.num_samples > 0:
        samples = samples[: args.num_samples]
        print(f"Limited to {len(samples)} samples")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    all_results_file = os.path.join(args.output_dir, "stage1_all.jsonl")
    filtered_file = os.path.join(args.output_dir, "stage1_filtered.jsonl")

    # Check for existing progress
    processed_indices: Set[Any] = set()
    if os.path.exists(all_results_file) and args.resume:
        processed_indices = get_processed_indices(
            all_results_file, idx_key="example_id"
        )
        print(f"Resuming: {len(processed_indices)} already processed")
    else:
        open(all_results_file, "w").close()

    # Reset validation statistics
    for key in _stats:
        _stats[key] = 0

    # Build queries
    queries = []
    for sample in samples:
        example_id = sample.get("example_id", "")
        if example_id in processed_indices:
            continue
        query = build_filter_query(sample, prompts)
        if query is not None:
            queries.append(query)

    print(f"Queries to process: {len(queries)}")

    if queries:
        # Parse URLs and initialize processor
        base_urls = parse_vllm_urls(args.vllm_url)
        model = args.model or DEFAULT_MODEL
        num_servers = len(base_urls)
        workers_per_server = max(1, args.num_workers // num_servers)

        processor = MassiveQueryProcessor(
            base_urls=base_urls,
            model=model,
            workers_per_server=workers_per_server,
            timeout=args.timeout,
            checkpoint_interval=10000,
        )

        await processor.process_all(
            queries=queries,
            output_file=all_results_file,
            process_fn=process_filter_result,
        )

    # Filter by min_score threshold
    print(f"\nFiltering with min_score >= {args.min_score}...")
    total = 0
    kept = 0
    cutoff_dropped = 0
    score_distribution = {s: 0 for s in range(1, 6)}

    with open(filtered_file, "w", encoding="utf-8") as out_f:
        with open(all_results_file, "r", encoding="utf-8") as in_f:
            for line in in_f:
                if not line.strip():
                    continue
                record = json.loads(line)
                total += 1
                score = record.get("persona_score", 0)
                if 1 <= score <= 5:
                    score_distribution[score] += 1

                if score >= args.min_score:
                    # Post-filter: reject captions describing truncated audio
                    caption = record.get("caption", "")
                    if has_caption_cutoff(caption):
                        cutoff_dropped += 1
                        continue

                    output = {
                        k: v
                        for k, v in record.items()
                        if k not in ("reason", "idx")
                    }
                    out_f.write(
                        json.dumps(output, ensure_ascii=False) + "\n"
                    )
                    kept += 1

    # Write summary
    summary = {
        "total_scored": total,
        "min_score_threshold": args.min_score,
        "kept": kept,
        "cutoff_dropped": cutoff_dropped,
        "dropped": total - kept,
        "filter_rate": round(kept / total, 4) if total > 0 else 0,
        "score_distribution": score_distribution,
    }
    summary_file = os.path.join(args.output_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Print score distribution
    print("\n" + "=" * 60)
    print("Score Distribution:")
    print(f"{'Score':<10} {'Count':>8} {'Percent':>8}  Bar")
    print("-" * 60)
    for score in range(1, 6):
        count = score_distribution[score]
        pct = count / total * 100 if total > 0 else 0
        bar = "#" * int(pct / 2)
        marker = " <-- threshold" if score == args.min_score else ""
        print(f"  {score:<8} {count:>8} {pct:>7.1f}%  {bar}{marker}")
    print("-" * 60)
    print(f"  Total    {total:>8}")
    print(f"  Kept (>={args.min_score})  {kept:>8} ({summary['filter_rate']:.1%})")
    print(f"  Dropped  {total - kept:>8}")
    print("=" * 60)

    # Print validation statistics
    total_validated = sum(_stats.values())
    if total_validated > 0:
        print("\nValidation Summary:")
        print(f"  {'Reason':<25} {'Count':>8} {'Percent':>8}")
        print("-" * 60)
        for reason, count in _stats.items():
            pct = count / total_validated * 100
            print(f"  {reason:<25} {count:>8} {pct:>7.1f}%")
        print("-" * 60)
        print(f"  {'Total':<25} {total_validated:>8}")

    print(f"\nOutput: {filtered_file}")
    print(f"Summary: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Score and filter captions for persona suitability."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input JSONL from general TTS stage1 filtered output.",
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
        "--min_score",
        type=int,
        default=4,
        help="Minimum persona score to keep (1-5, default: 4).",
    )
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
