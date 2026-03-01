#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Stage 0: Transcribe speech audio using Qwen3-ASR via vLLM.

Sends audio files as base64-encoded data URLs to the Qwen3-ASR vLLM server's
``/v1/chat/completions`` endpoint and saves transcriptions as JSONL.  The
resulting ``whisper_lyrics`` field is used by Stage 1 as the reference
transcript for WER filtering.
"""

import argparse
import asyncio
import base64
import json
import logging
import multiprocessing
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from sft_vllm_client import (
    MassiveQueryProcessor,
    get_processed_indices,
    parse_vllm_urls,
)

logger = logging.getLogger(__name__)

# Module-level counters for drop tracking across chunked processing
_drop_stats = {"no_response": 0, "empty_lyrics": 0}
_skip_stats = {"no_audio_path": 0, "file_not_found": 0, "duration": 0, "encode_fail": 0}

# MIME types for supported audio formats
_MIME_TYPES = {
    ".flac": "audio/flac",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
}

# Chunk size for processing to bound memory usage.
# Each chunk holds base64-encoded audio in memory (~10 MB per 5-min FLAC).
# 5000 chunks ~ 50 GB peak, well within 2 TB server memory.
_CHUNK_SIZE = 50000


# =============================================================================
# Helper functions
# =============================================================================


def parse_asr_text(raw: str) -> str:
    """Extract transcribed text from Qwen3-ASR output format.

    Qwen3-ASR returns text like ``"language English<asr_text>actual text"``.
    This function extracts the text after the ``<asr_text>`` tag.
    """
    match = re.search(r"<asr_text>(.*)", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: return the raw text stripped
    return raw.strip()


def encode_audio_base64(audio_path: str) -> Optional[str]:
    """Read an audio file and return a base64-encoded data URL.

    Returns ``"data:<mime>;base64,..."`` or ``None`` on error.
    """
    ext = os.path.splitext(audio_path)[1].lower()
    mime = _MIME_TYPES.get(ext)
    if mime is None:
        logger.warning("Unsupported audio extension %s: %s", ext, audio_path)
        return None

    try:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
    except (OSError, IOError) as e:
        logger.warning("Cannot read audio file %s: %s", audio_path, e)
        return None

    b64 = base64.b64encode(audio_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_transcription_query(
    args: Tuple[int, Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Build a multimodal query for Qwen3-ASR transcription.

    Returns (query, None) on success or (None, skip_reason) on skip.
    Accepts a single tuple argument for compatibility with multiprocessing.
    """
    idx, sample = args
    audio_path = sample.get("audio_path", "")
    if not audio_path:
        return None, "no_audio_path"
    if not os.path.isfile(audio_path):
        return None, "file_not_found"

    duration = sample.get("duration", 0.0)
    if duration < 0.5 or duration > 1200:
        return None, "duration"

    data_url = encode_audio_base64(audio_path)
    if data_url is None:
        return None, "encode_fail"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": data_url}},
            ],
        }
    ]

    query = {
        "idx": idx,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 1024,
        "metadata": sample,
    }
    return query, None


def process_transcription_result(
    result: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Parse ASR response and return output record with whisper_lyrics.

    Samples where transcription fails are still saved with
    ``has_whisper_transcription=False`` and an empty ``whisper_lyrics`` field.
    """
    response = result.get("response")
    metadata = result.get("metadata", {})

    has_whisper_transcription = True
    lyrics = ""

    if response is None:
        _drop_stats["no_response"] += 1
        has_whisper_transcription = False
    else:
        lyrics = parse_asr_text(response)
        if not lyrics:
            _drop_stats["empty_lyrics"] += 1
            has_whisper_transcription = False

    output = {
        "example_id": metadata.get("example_id", ""),
        "dataset": metadata.get("dataset", ""),
        "audio_path": metadata.get("audio_path", ""),
        "duration": metadata.get("duration", 0.0),
        "sample_rate": metadata.get("sample_rate", 0),
        "caption": metadata.get("caption", ""),
        "whisper_lyrics": lyrics,
        "has_whisper_transcription": has_whisper_transcription,
        "idx": result["idx"],
    }
    return output


# =============================================================================
# Main
# =============================================================================


async def main_async(args) -> None:
    # Reset module-level counters
    for k in _drop_stats:
        _drop_stats[k] = 0
    for k in _skip_stats:
        _skip_stats[k] = 0

    print(f"Loading input from: {args.input_file}")
    samples: List[Dict[str, Any]] = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    print(f"Total input samples: {len(samples)}")

    if args.num_samples > 0:
        samples = samples[: args.num_samples]
        print(f"Limited to {len(samples)} samples")

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "transcriptions.jsonl")

    # Resume support
    processed_indices: Set[Any] = set()
    if os.path.exists(output_file) and args.resume:
        processed_indices = get_processed_indices(output_file)
        print(f"Resuming: {len(processed_indices)} already processed")
    elif not args.resume:
        open(output_file, "w").close()

    # Set up processor
    base_urls = parse_vllm_urls(args.whisper_url)
    workers_per_server = max(1, args.num_workers // len(base_urls))

    processor = MassiveQueryProcessor(
        base_urls=base_urls,
        model=args.model,
        workers_per_server=workers_per_server,
        timeout=args.timeout,
        checkpoint_interval=10000,
    )

    # Process in chunks to bound memory (base64 audio is large)
    total_transcribed = 0
    total_duration = 0.0
    total_words = 0

    remaining = [
        (idx, s)
        for idx, s in enumerate(samples)
        if idx not in processed_indices
    ]
    print(f"Samples remaining: {len(remaining)}")

    num_encode_workers = min(args.num_encode_workers, multiprocessing.cpu_count())

    for chunk_start in range(0, len(remaining), _CHUNK_SIZE):
        chunk = remaining[chunk_start : chunk_start + _CHUNK_SIZE]
        chunk_num = chunk_start // _CHUNK_SIZE + 1
        total_chunks = (len(remaining) + _CHUNK_SIZE - 1) // _CHUNK_SIZE
        print(
            f"\nChunk {chunk_num}/{total_chunks}: "
            f"encoding {len(chunk)} audio files "
            f"with {num_encode_workers} workers..."
        )

        # Parallelize audio encoding across the chunk
        with multiprocessing.Pool(processes=num_encode_workers) as pool:
            results = pool.map(build_transcription_query, chunk)

        queries = []
        skipped_in_chunk = 0
        for query, skip_reason in results:
            if query is not None:
                queries.append(query)
            else:
                skipped_in_chunk += 1
                if skip_reason:
                    _skip_stats[skip_reason] += 1

        if skipped_in_chunk > 0:
            print(f"  Skipped {skipped_in_chunk} samples (query build)")

        if not queries:
            print(f"  No valid queries in chunk {chunk_num}, skipping")
            continue

        print(f"  Processing {len(queries)} queries...")
        await processor.process_all(
            queries=queries,
            output_file=output_file,
            process_fn=process_transcription_result,
        )

    # Compute summary statistics
    print("\nComputing summary statistics...")
    total_transcribed = 0
    total_duration = 0.0
    total_words = 0

    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                total_transcribed += 1
                total_duration += record.get("duration", 0.0)
                lyrics = record.get("whisper_lyrics", "")
                total_words += len(lyrics.split())

    summary = {
        "total_input": len(samples),
        "total_transcribed": total_transcribed,
        "transcription_rate": (
            round(total_transcribed / len(samples), 4)
            if len(samples) > 0
            else 0
        ),
        "avg_duration": (
            round(total_duration / total_transcribed, 2)
            if total_transcribed > 0
            else 0
        ),
        "avg_word_count": (
            round(total_words / total_transcribed, 1)
            if total_transcribed > 0
            else 0
        ),
        "skipped": dict(_skip_stats),
        "dropped": dict(_drop_stats),
    }

    summary_file = os.path.join(args.output_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    total_skipped = sum(_skip_stats.values())
    total_dropped = sum(_drop_stats.values())

    print("\n" + "=" * 60)
    print(f"Total input       : {summary['total_input']}")
    print(f"Total transcribed : {summary['total_transcribed']}")
    print(f"Transcription rate: {summary['transcription_rate']:.1%}")
    print(f"Avg duration (s)  : {summary['avg_duration']}")
    print(f"Avg word count    : {summary['avg_word_count']}")
    if total_skipped > 0:
        print(f"\nSkipped (pre-query): {total_skipped}")
        for reason, count in _skip_stats.items():
            if count > 0:
                print(f"  {reason}: {count}")
    if total_dropped > 0:
        print(f"\nDropped (post-ASR) : {total_dropped}")
        for reason, count in _drop_stats.items():
            if count > 0:
                print(f"  {reason}: {count}")
    print("=" * 60)
    print(f"\nOutput : {output_file}")
    print(f"Summary: {summary_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 0: Transcribe speech audio with Qwen3-ASR."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input JSONL with audio_path field.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--whisper_url",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM API base URL(s), colon-separated.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-ASR-1.7B",
        help="Qwen3-ASR model name.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=128,
        help="Total concurrent workers across all servers.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Request timeout in seconds.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=-1,
        help="Number of samples to process (-1 for all).",
    )
    parser.add_argument(
        "--num_encode_workers",
        type=int,
        default=64,
        help="Number of multiprocessing workers for audio base64 encoding.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing progress.",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
