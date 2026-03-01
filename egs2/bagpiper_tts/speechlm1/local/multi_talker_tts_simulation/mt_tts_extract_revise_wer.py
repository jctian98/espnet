#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Stage 1: Extract transcription from multi-talker captions with revision loop.

Unlike the single-speaker pipeline which filters WER=0 only, this stage uses
a multi-pass extract-revise-inspect loop:
  Pass 0: Extract transcription from all captions (temp=0.0)
           -> WER < 2%: done (keep)
           -> 2% <= WER <= 10%: pending (send to revision)
           -> WER > 10%: discard
  Pass 1-3: For pending samples:
           -> Revise caption to fix transcription (temp=0.3)
           -> Extract transcription from revised caption (temp=0.0)
           -> Apply same WER thresholds
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
# Prompts for transcription extraction and revision
# =============================================================================

# Extract prompts vary per pass to avoid identical retries producing
# the same output.  Pass 0 is the initial attempt; passes 1-3 are
# re-extractions after a revision.
_EXTRACT_SYSTEM_BASE = (
    "You are a precise speech reconstruction assistant. Given a "
    "rich audio caption that describes a multi-speaker recording, "
    "reconstruct the COMPLETE spoken script — every word said by "
    "ALL speakers in order.\n\n"
    "The caption may describe speech in two ways:\n"
    "1. Direct quotes (e.g., He says, \"I love this place.\")\n"
    "2. Indirect/reported speech (e.g., She explains that the "
    "community has felt isolated.)\n\n"
    "You MUST reconstruct the spoken words for BOTH cases. For "
    "indirect speech, convert it back to the actual words the "
    "speaker would have said.\n\n"
    "Output ONLY the full reconstructed transcription with no "
    "quotes, no speaker labels, no explanation."
)

PROMPTS_BY_VERSION = {
    "v1": {
        # --- extract prompts (one per pass) ---
        "extract_system": [
            # Pass 0: baseline
            _EXTRACT_SYSTEM_BASE,
            # Pass 1: emphasise completeness
            (
                _EXTRACT_SYSTEM_BASE + "\n\n"
                "IMPORTANT: Make sure you capture EVERY segment of speech "
                "described in the caption, including narration, interviews, "
                "and any filler words or hesitations. Do NOT skip anything."
            ),
            # Pass 2: emphasise indirect speech
            (
                _EXTRACT_SYSTEM_BASE + "\n\n"
                "Pay SPECIAL attention to indirect/reported speech. When "
                "the caption says something like 'she recounts that X "
                "happened', you must reconstruct the narrator's actual "
                "words, not just summarize them."
            ),
            # Pass 3: emphasise word-level precision
            (
                _EXTRACT_SYSTEM_BASE + "\n\n"
                "Focus on WORD-LEVEL precision. Include filler words "
                "(uh, um, I mean), false starts, self-corrections, and "
                "every conversational detail described in the caption."
            ),
        ],
        "extract_user": [
            # Pass 0
            (
                "Reconstruct the complete spoken script from this audio "
                "caption. Include ALL speech — both directly quoted and "
                "indirectly described:\n\n{caption}"
            ),
            # Pass 1
            (
                "Read the following audio caption carefully. Reconstruct "
                "every word spoken by every speaker, in order. Do not "
                "leave out any segment:\n\n{caption}"
            ),
            # Pass 2
            (
                "The caption below describes a multi-speaker recording. "
                "Some speech is quoted directly, some is described "
                "indirectly. Reconstruct ALL of it as a continuous "
                "transcript:\n\n{caption}"
            ),
            # Pass 3
            (
                "Reconstruct the full verbatim transcript from this "
                "caption. Include every filler word, hesitation, and "
                "self-correction mentioned:\n\n{caption}"
            ),
        ],
        # --- revise prompts ---
        "revise_system": (
            "You are a precise caption editor. You will be given an audio "
            "caption, the text currently extracted from it, and the ground "
            "truth transcription.\n\n"
            "Your task: revise ONLY the spoken/quoted text portions of the "
            "caption so that the words match the ground truth EXACTLY. "
            "Keep ALL other descriptive content (voice descriptions, "
            "recording quality notes, music descriptions, etc.) unchanged."
            "\n\n"
            "IMPORTANT: The ground truth transcription is in ALL UPPER "
            "CASE for technical reasons. Do NOT paste it as-is. Adapt the "
            "casing to match the caption's natural style (typically normal "
            "sentence case with proper capitalization).\n\n"
            "Output ONLY the revised caption, nothing else."
        ),
        "revise_user": (
            "The following audio caption contains spoken text that does "
            "not exactly match the ground truth transcription. Revise "
            "ONLY the spoken/quoted text portions to match the ground "
            "truth, keeping all descriptive content intact.\n\n"
            "REMEMBER: The ground truth is in ALL CAPS. Convert to "
            "natural sentence case when inserting into the caption.\n\n"
            "--- Current Caption ---\n{caption}\n\n"
            "--- Extracted Text (from caption) ---\n{extracted_text}\n\n"
            "--- Ground Truth Transcription ---\n{ground_truth}\n\n"
            "Output the revised caption:"
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


# Number word mappings for normalization
_ONES = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_SCALES = {
    "hundred": 100,
    "thousand": 1000,
    "million": 1000000,
    "billion": 1000000000,
}
_ALL_NUMBER_WORDS = set(_ONES) | set(_TENS) | set(_SCALES)


def _collapse_single_letters(text: str) -> str:
    """Collapse sequences of single letters into one token.

    E.g., "u s a" -> "usa", "l a" -> "la".
    This handles abbreviations that may be spaced out in transcriptions.
    """

    def _replace_match(m):
        return m.group(0).replace(" ", "")

    # Match 2+ single letters separated by spaces
    return re.sub(r"\b([a-z])(?: ([a-z]))+\b", _replace_match, text)


def _normalize_numbers(text: str) -> str:
    """Normalize number words to digit strings.

    Handles compound numbers: "twenty one" -> "21",
    "two thousand" -> "2000", "twenty" -> "20".
    Also handles digit strings: "21" stays "21".
    """
    words = text.split()
    result = []
    i = 0
    while i < len(words):
        if words[i] not in _ALL_NUMBER_WORDS:
            result.append(words[i])
            i += 1
            continue

        # Accumulate number words
        num_val = 0
        current = 0
        while i < len(words) and words[i] in _ALL_NUMBER_WORDS:
            w = words[i]
            if w in _ONES:
                current += _ONES[w]
            elif w in _TENS:
                current += _TENS[w]
            elif w in _SCALES:
                if current == 0:
                    current = 1
                current *= _SCALES[w]
                num_val += current
                current = 0
            i += 1

        num_val += current
        result.append(str(num_val))

    return " ".join(result)


def normalize_text(text: str) -> str:
    """Normalize text for WER comparison.

    Steps: strip possessives, lowercase, remove punctuation,
    collapse single-letter sequences (e.g., "U S A" -> "usa"),
    normalize number words to digits.
    """
    # Strip possessive 's before punctuation removal so "A'S" -> "A"
    text = re.sub(r"'[sS]\b", "", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = _collapse_single_letters(text)
    text = _normalize_numbers(text)
    return text


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate between reference and hypothesis.

    Uses edit distance at word level.
    """
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()

    if not ref_words and not hyp_words:
        return 0.0
    if not ref_words:
        return float(len(hyp_words))
    if not hyp_words:
        return 1.0

    # Dynamic programming for edit distance
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,  # deletion
                    d[i][j - 1] + 1,  # insertion
                    d[i - 1][j - 1] + 1,  # substitution
                )

    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def load_kaldi_file(filepath: str) -> Dict[str, str]:
    """Load a Kaldi-format file (utt_id SPACE content)."""
    data = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                data[parts[0]] = parts[1]
    return data


def load_captions(filepath: str) -> Dict[str, Dict[str, Any]]:
    """Load captions JSONL file, keyed by utt_id."""
    captions = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                utt_id = record.get("utt_id", "")
                if utt_id:
                    captions[utt_id] = record
    return captions


def load_data(data_root: str) -> List[Dict[str, Any]]:
    """Load and join data from flat directory.

    Expects: wav.scp, text, rich_captions.jsonl directly under data_root.
    Returns list of sample dicts with utt_id, text, audio_path, rich_caption.
    """
    samples = []

    # Load all source files
    text_data = load_kaldi_file(os.path.join(data_root, "text"))
    wavscp_data = load_kaldi_file(os.path.join(data_root, "wav.scp"))
    caption_data = load_captions(
        os.path.join(data_root, "rich_captions.jsonl")
    )

    print(
        f"  Loaded: {len(text_data)} text, "
        f"{len(wavscp_data)} wav.scp, "
        f"{len(caption_data)} captions"
    )

    # Join on utt_id (only keep samples with all fields)
    common_ids = (
        set(text_data.keys())
        & set(wavscp_data.keys())
        & set(caption_data.keys())
    )

    for utt_id in sorted(common_ids):
        audio_path = wavscp_data[utt_id]

        samples.append(
            {
                "utt_id": utt_id,
                "text": text_data[utt_id],
                "audio_path": audio_path,
                "rich_caption": caption_data[utt_id].get(
                    "rich_caption", ""
                ),
            }
        )

    print(f"  {len(common_ids)} samples after join")
    return samples


def build_extraction_query(
    sample: Dict[str, Any],
    prompts: Dict[str, Any],
    pass_idx: int = 0,
) -> Optional[Dict[str, Any]]:
    """Build a query for transcription extraction.

    Args:
        pass_idx: Which pass (0-3). Selects prompt variant for diversity.
    """
    caption = sample.get("rich_caption", "")
    utt_id = sample.get("utt_id", "")

    if not caption or len(caption.strip()) < 20:
        return None

    # Select prompt variant, cycling if pass_idx exceeds available
    sys_prompts = prompts["extract_system"]
    usr_prompts = prompts["extract_user"]
    idx = pass_idx % len(sys_prompts)

    messages = [
        {"role": "system", "content": sys_prompts[idx]},
        {
            "role": "user",
            "content": usr_prompts[idx].format(caption=caption),
        },
    ]

    # Use small non-zero temperature for retry passes
    temperature = 0.0 if pass_idx == 0 else 0.1

    return {
        "idx": utt_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 1024,
        "metadata": sample,
    }


def build_revision_query(
    sample: Dict[str, Any],
    extracted_text: str,
    prompts: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    """Build a query for caption revision."""
    caption = sample.get("rich_caption", "")
    ground_truth = sample.get("text", "")
    utt_id = sample.get("utt_id", "")

    if not caption or not ground_truth:
        return None

    messages = [
        {"role": "system", "content": prompts["revise_system"]},
        {
            "role": "user",
            "content": prompts["revise_user"].format(
                caption=caption,
                extracted_text=extracted_text,
                ground_truth=ground_truth,
            ),
        },
    ]

    return {
        "idx": utt_id,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 4096,
        "metadata": {**sample, "extracted_text": extracted_text},
    }


def process_extraction_result(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process extraction result: compute WER and build output record."""
    response = result.get("response")
    metadata = result.get("metadata", {})

    if response is None:
        return None

    extracted_text = response.strip()

    if not extracted_text:
        return None

    # Compute WER against ground truth
    ground_truth = metadata.get("text", "")
    wer = compute_wer(ground_truth, extracted_text)

    return {
        "utt_id": metadata.get("utt_id", ""),
        "text": ground_truth,
        "rich_caption": metadata.get("rich_caption", ""),
        "audio_path": metadata.get("audio_path", ""),
        "extracted_text": extracted_text,
        "wer": round(wer, 4),
        "idx": result["idx"],
    }


def _word_edit_distance(words_a: List[str], words_b: List[str]) -> int:
    """Compute word-level edit distance (Levenshtein) between two word lists."""
    m, n = len(words_a), len(words_b)
    d = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if words_a[i - 1] == words_b[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,
                    d[i][j - 1] + 1,
                    d[i - 1][j - 1] + 1,
                )
    return d[m][n]


def _simple_tokenize(text: str) -> List[str]:
    """Lowercase and split into words, keeping punctuation attached."""
    return text.lower().split()


def _validate_revision(
    original_caption: str,
    revised_caption: str,
    extracted_text: str,
    ground_truth: str,
    max_edit_ratio: float = 3.0,
) -> float:
    """Validate that a revision mostly changed transcription, not description.

    Computes:
      caption_edits  = edit_distance(original_caption, revised_caption)
      transcription_edits = edit_distance(extracted_text, ground_truth)

    Most caption edits should come from fixing the transcription.
    Returns the ratio caption_edits / max(transcription_edits, 1).
    A low ratio (close to 1.0) means the revision only changed what
    was needed; a high ratio means the LLM rewrote descriptive content.
    """
    orig_words = _simple_tokenize(original_caption)
    rev_words = _simple_tokenize(revised_caption)
    ext_words = _simple_tokenize(extracted_text)
    gt_words = _simple_tokenize(ground_truth)

    caption_edits = _word_edit_distance(orig_words, rev_words)
    transcription_edits = _word_edit_distance(ext_words, gt_words)

    ratio = caption_edits / max(transcription_edits, 1)

    # Also compute word-level overlap ratio between original and revised
    orig_set = set(orig_words)
    rev_set = set(rev_words)
    if orig_set:
        overlap = len(orig_set & rev_set) / len(orig_set)
    else:
        overlap = 1.0

    return ratio, caption_edits, transcription_edits, overlap


def process_revision_result(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process revision result: return revised caption.

    Validates that the revision mostly changed the transcription
    portions, not the descriptive content, by comparing edit distances.
    """
    response = result.get("response")
    metadata = result.get("metadata", {})
    utt_id = metadata.get("utt_id", "")

    if response is None:
        return None

    revised_caption = response.strip()
    if not revised_caption or len(revised_caption) < 20:
        return None

    original_caption = metadata.get("rich_caption", "")
    extracted_text = metadata.get("extracted_text", "")
    ground_truth = metadata.get("text", "")

    # Validate: caption edits should be proportional to transcription edits
    ratio, cap_edits, trans_edits, overlap = _validate_revision(
        original_caption, revised_caption, extracted_text, ground_truth
    )
    if ratio > 1.5:
        print(
            f"[DISCARD] utt_id={utt_id} "
            f"reason=revision_changed_too_much "
            f"caption_edits={cap_edits} "
            f"transcription_edits={trans_edits} "
            f"ratio={ratio:.2f} overlap={overlap:.2f}"
        )
        return None

    return {
        "utt_id": utt_id,
        "text": ground_truth,
        "audio_path": metadata.get("audio_path", ""),
        "revised_caption": revised_caption,
        "original_caption": original_caption,
        "edit_ratio": round(ratio, 2),
        "overlap": round(overlap, 4),
        "idx": result["idx"],
    }


def classify_by_wer(
    results_file: str,
    wer_threshold: float,
    discard_threshold: float,
) -> Dict[str, List[Dict[str, Any]]]:
    """Classify extraction results by WER into done/pending/discard."""
    done = []
    pending = []
    discard = []

    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            wer = record.get("wer", 1.0)

            if wer < wer_threshold:
                done.append(record)
            elif wer <= discard_threshold:
                pending.append(record)
            else:
                discard.append(record)

    return {"done": done, "pending": pending, "discard": discard}


async def run_batch(
    queries: List[Dict[str, Any]],
    output_file: str,
    process_fn,
    base_urls: List[str],
    model: str,
    num_workers: int,
    timeout: int,
    resume: bool,
):
    """Run a batch of queries through MassiveQueryProcessor."""
    # Check for existing progress
    processed_ids: Set[Any] = set()
    if os.path.exists(output_file) and resume:
        processed_ids = get_processed_indices(output_file, idx_key="utt_id")
        print(f"  Resuming: {len(processed_ids)} already processed")
    else:
        open(output_file, "w").close()

    # Filter out already processed
    remaining = [q for q in queries if q["idx"] not in processed_ids]
    print(f"  Queries to process: {len(remaining)}")

    if not remaining:
        print("  All already processed!")
        return

    num_servers = len(base_urls)
    workers_per_server = max(1, num_workers // num_servers)

    processor = MassiveQueryProcessor(
        base_urls=base_urls,
        model=model,
        workers_per_server=workers_per_server,
        timeout=timeout,
        checkpoint_interval=10000,
    )

    await processor.process_all(
        queries=remaining,
        output_file=output_file,
        process_fn=process_fn,
    )


async def main_async(args):
    """Main async function with multi-pass extract-revise loop."""
    prompts = get_prompts(args.version)
    print(f"Using prompt version: {args.version}")

    # Load data
    print(f"Loading data from {args.data_root}")
    samples = load_data(args.data_root)
    print(f"Total samples: {len(samples)}")

    # Limit samples if specified
    if args.num_samples > 0:
        samples = samples[: args.num_samples]
        print(f"Limited to {len(samples)} samples")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse URLs
    base_urls = parse_vllm_urls(args.vllm_url)
    model = args.model or DEFAULT_MODEL

    # Track per-pass statistics
    pass_stats = []

    # =========================================================================
    # Pass 0: Initial extraction
    # =========================================================================
    print("\n" + "=" * 60)
    print("Pass 0: Initial extraction")
    print("=" * 60)

    extract0_file = os.path.join(args.output_dir, "stage1_extract0.jsonl")

    # Build extraction queries (pass 0)
    extract_queries = []
    for sample in samples:
        query = build_extraction_query(sample, prompts, pass_idx=0)
        if query is not None:
            extract_queries.append(query)

    print(f"Built {len(extract_queries)} extraction queries")

    await run_batch(
        queries=extract_queries,
        output_file=extract0_file,
        process_fn=process_extraction_result,
        base_urls=base_urls,
        model=model,
        num_workers=args.num_workers,
        timeout=args.timeout,
        resume=args.resume,
    )

    # Classify results
    classification = classify_by_wer(
        extract0_file, args.wer_threshold, args.discard_threshold
    )
    pass_stats.append(
        {
            "pass": 0,
            "type": "extract",
            "done": len(classification["done"]),
            "pending": len(classification["pending"]),
            "discard": len(classification["discard"]),
        }
    )
    print(
        f"\nPass 0 results: "
        f"done={len(classification['done'])}, "
        f"pending={len(classification['pending'])}, "
        f"discard={len(classification['discard'])}"
    )

    # Accumulate final results
    all_done = list(classification["done"])
    all_discard = list(classification["discard"])
    current_pending = classification["pending"]

    # =========================================================================
    # Revision passes (up to max_revision_passes)
    # =========================================================================
    for rev_pass in range(1, args.max_revision_passes + 1):
        if not current_pending:
            print(f"\nNo pending samples left. Stopping at pass {rev_pass}.")
            break

        print("\n" + "=" * 60)
        print(
            f"Pass {rev_pass}: Revise + Extract "
            f"({len(current_pending)} pending)"
        )
        print("=" * 60)

        # --- Revise step ---
        revise_file = os.path.join(
            args.output_dir, f"stage1_revise{rev_pass}.jsonl"
        )

        # Build sample lookup from pending
        pending_by_id = {r["utt_id"]: r for r in current_pending}

        # Build original sample lookup
        sample_by_id = {s["utt_id"]: s for s in samples}

        revise_queries = []
        for record in current_pending:
            utt_id = record["utt_id"]
            sample = sample_by_id.get(utt_id, {})
            # Use the current caption (possibly already revised)
            sample_for_revise = dict(sample)
            sample_for_revise["rich_caption"] = record.get(
                "rich_caption", sample.get("rich_caption", "")
            )
            extracted = record.get("extracted_text", "")
            query = build_revision_query(
                sample_for_revise, extracted, prompts
            )
            if query is not None:
                revise_queries.append(query)

        print(f"\n  Revise step: {len(revise_queries)} queries")
        await run_batch(
            queries=revise_queries,
            output_file=revise_file,
            process_fn=process_revision_result,
            base_urls=base_urls,
            model=model,
            num_workers=args.num_workers,
            timeout=args.timeout,
            resume=args.resume,
        )

        # Load revised captions
        revised_by_id = {}
        with open(revise_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    revised_by_id[r["utt_id"]] = r

        # --- Extract step from revised captions ---
        extract_file = os.path.join(
            args.output_dir, f"stage1_extract{rev_pass}.jsonl"
        )

        extract_queries = []
        for utt_id, revised in revised_by_id.items():
            sample = dict(sample_by_id.get(utt_id, {}))
            sample["rich_caption"] = revised["revised_caption"]
            query = build_extraction_query(
                sample, prompts, pass_idx=rev_pass
            )
            if query is not None:
                extract_queries.append(query)

        # Also handle pending samples that failed revision
        failed_revision = set(pending_by_id.keys()) - set(revised_by_id.keys())
        if failed_revision:
            print(f"  {len(failed_revision)} samples failed revision")

        print(f"  Extract step: {len(extract_queries)} queries")
        await run_batch(
            queries=extract_queries,
            output_file=extract_file,
            process_fn=process_extraction_result,
            base_urls=base_urls,
            model=model,
            num_workers=args.num_workers,
            timeout=args.timeout,
            resume=args.resume,
        )

        # Classify this pass's results
        classification = classify_by_wer(
            extract_file, args.wer_threshold, args.discard_threshold
        )

        # Update captions for done samples to use revised version,
        # keeping the original caption and revision metrics
        for record in classification["done"]:
            utt_id = record["utt_id"]
            if utt_id in revised_by_id:
                rev = revised_by_id[utt_id]
                original = sample_by_id.get(utt_id, {})
                record["original_rich_caption"] = original.get(
                    "rich_caption", ""
                )
                record["rich_caption"] = rev["revised_caption"]
                record["revision_pass"] = rev_pass
                record["edit_ratio"] = rev.get("edit_ratio", 0.0)
                record["overlap"] = rev.get("overlap", 0.0)

        for record in classification["pending"]:
            utt_id = record["utt_id"]
            if utt_id in revised_by_id:
                record["rich_caption"] = revised_by_id[utt_id][
                    "revised_caption"
                ]

        pass_stats.append(
            {
                "pass": rev_pass,
                "type": "revise+extract",
                "revised": len(revised_by_id),
                "failed_revision": len(failed_revision),
                "done": len(classification["done"]),
                "pending": len(classification["pending"]),
                "discard": len(classification["discard"]),
            }
        )
        print(
            f"\n  Pass {rev_pass} results: "
            f"done={len(classification['done'])}, "
            f"pending={len(classification['pending'])}, "
            f"discard={len(classification['discard'])}"
        )

        all_done.extend(classification["done"])
        all_discard.extend(classification["discard"])
        current_pending = classification["pending"]

    # Any remaining pending after all passes are discarded
    if current_pending:
        print(
            f"\n{len(current_pending)} samples still pending after "
            f"{args.max_revision_passes} passes - discarding"
        )
        all_discard.extend(current_pending)

    # =========================================================================
    # Write final output
    # =========================================================================
    filtered_file = os.path.join(args.output_dir, "stage1_filtered.jsonl")
    with open(filtered_file, "w", encoding="utf-8") as f:
        for record in all_done:
            if "revision_pass" not in record:
                record["revision_pass"] = 0
            # Pass-0 samples have no revision; set defaults
            if "original_rich_caption" not in record:
                record["original_rich_caption"] = record.get(
                    "rich_caption", ""
                )
            if "edit_ratio" not in record:
                record["edit_ratio"] = 0.0
            if "overlap" not in record:
                record["overlap"] = 1.0
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Write summary
    summary = {
        "total_samples": len(samples),
        "total_done": len(all_done),
        "total_discard": len(all_discard),
        "pass_rate": (
            round(len(all_done) / len(samples), 4) if samples else 0
        ),
        "wer_threshold": args.wer_threshold,
        "discard_threshold": args.discard_threshold,
        "max_revision_passes": args.max_revision_passes,
        "pass_stats": pass_stats,
    }
    summary_file = os.path.join(args.output_dir, "stage1_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("Stage 1 Summary (Extract + Revise Loop)")
    print("=" * 60)
    print(f"Total samples:    {len(samples)}")
    print(f"Total done:       {len(all_done)}")
    print(f"Total discarded:  {len(all_discard)}")
    print(f"Pass rate:        {summary['pass_rate']:.1%}")
    print(f"WER threshold:    {args.wer_threshold}")
    print(f"Discard threshold:{args.discard_threshold}")
    print()
    print("Per-pass breakdown:")
    print(f"  {'Pass':<20} {'Done':>8} {'Pending':>8} {'Discard':>8}")
    print("  " + "-" * 48)
    for ps in pass_stats:
        label = f"Pass {ps['pass']} ({ps['type']})"
        print(
            f"  {label:<20} {ps['done']:>8} "
            f"{ps.get('pending', 0):>8} {ps['discard']:>8}"
        )
    print("=" * 60)

    print(f"\nOutput: {filtered_file}")
    print(f"Summary: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract transcription from multi-talker captions "
            "with revision loop."
        )
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root path to multi-talker data directory.",
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
        "--wer_threshold",
        type=float,
        default=0.02,
        help="WER threshold for acceptance (default: 0.02 = 2%%).",
    )
    parser.add_argument(
        "--discard_threshold",
        type=float,
        default=0.10,
        help="WER threshold for discard (default: 0.10 = 10%%).",
    )
    parser.add_argument(
        "--max_revision_passes",
        type=int,
        default=3,
        help="Maximum number of revision passes (default: 3).",
    )
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
