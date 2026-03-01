#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Stage 2: Generate diverse user requests for TTS across application categories.

Two-pass approach for balanced category distribution:
  Pass 1 (Category Selection): Randomly sample 10 categories from 40, ask LLM
      to pick the 3 most fitting ones, then randomly select 1 of those 3.
  Pass 2 (Request Generation): Feed the chosen category into the prompt to
      generate the actual user request.
"""

import argparse
import asyncio
import json
import os
import random
from collections import Counter
from typing import Any, Dict, List, Optional

from sft_vllm_client import (
    DEFAULT_MODEL,
    MassiveQueryProcessor,
    get_processed_indices,
    parse_vllm_urls,
)

# =============================================================================
# Application category taxonomy (40 categories across 8 tiers)
# =============================================================================
APPLICATION_CATEGORIES = [
    # Tier 1: Content Production
    ("audiobook_narration", "Read a passage as an audiobook narrator"),
    ("podcast_host", "Deliver content as a podcast host talking to listeners"),
    ("news_anchor", "Read content in a news broadcast style"),
    ("documentary_narrator", "Narrate a documentary segment"),
    ("commercial_voiceover", "Record an advertisement or promotional spot"),
    ("movie_trailer", "Deliver dramatic movie/show trailer narration"),
    ("radio_dj", "Announce as an upbeat radio DJ"),
    ("sports_commentary", "Deliver excited sports commentary"),
    # Tier 2: Professional & Business
    ("presentation_narration", "Narrate slides or a business presentation"),
    ("corporate_training", "Deliver corporate training or onboarding content"),
    ("legal_reading", "Read legal documents, contracts, or terms formally"),
    ("medical_dictation", "Dictate a clinical or medical report"),
    ("meeting_summary", "Narrate meeting minutes or a recap"),
    # Tier 3: Education & Learning
    ("lecture_delivery", "Deliver an academic lecture or talk"),
    ("tutorial_explanation", "Explain a how-to, walkthrough, or guide"),
    ("language_teaching", "Model clear pronunciation for language learners"),
    ("childrens_story", "Narrate a children's book or bedtime story warmly"),
    ("exam_reading", "Read exam or test questions neutrally"),
    ("elearning_module", "Narrate an online course module"),
    # Tier 4: Accessibility & Assistive
    ("screen_reader", "Read screen content for visually impaired users"),
    ("audio_description", "Describe visual content (images, video scenes)"),
    ("slow_clear_speech", "Speak slowly and clearly for comprehension"),
    ("aac_voice", "Serve as a personal voice for someone who cannot speak"),
    # Tier 5: Communication Systems
    ("ivr_phone_system", "Automated phone menu or hold message"),
    ("public_announcement", "PA system announcement (airport, train station, store)"),
    ("voice_assistant", "Smart assistant response (like Siri/Alexa)"),
    ("voice_message", "Compose and send a personal voice message"),
    ("gps_navigation", "Give turn-by-turn navigation directions"),
    # Tier 6: Creative & Artistic
    ("poetry_reading", "Read poetry with artistic cadence and emotion"),
    ("dramatic_monologue", "Perform a dramatic theatrical monologue"),
    ("comedy_delivery", "Deliver content with comedic timing and humor"),
    ("spoken_word", "Perform spoken word poetry or slam poetry"),
    ("audio_drama_character", "Voice a character in an audio drama/play"),
    # Tier 7: Emotional & Interpersonal
    ("motivational_speech", "Deliver an inspiring, motivational message"),
    ("comfort_soothing", "Speak softly and soothingly to comfort someone"),
    ("congratulatory_message", "Deliver a warm congratulatory message"),
    ("sympathy_condolence", "Deliver a gentle sympathy or condolence message"),
    # Tier 8: Prosodic & Style Control
    ("whispering_asmr", "Speak in a whisper or intimate ASMR-like style"),
    ("formal_ceremonial", "Deliver text in a formal or ceremonial manner"),
    ("casual_conversational", "Speak casually as in everyday conversation"),
]

CATEGORY_DICT = {cat_id: desc for cat_id, desc in APPLICATION_CATEGORIES}
VALID_CATEGORIES = set(CATEGORY_DICT.keys())
NUM_CANDIDATE_CATEGORIES = 10
NUM_LLM_PICKS = 3

# =============================================================================
# Pass 1: Category selection prompts
# =============================================================================
CATEGORY_SELECTION_PROMPTS = {
    "v1": {
        "system": (
            "You are a classification assistant. Given a rich audio caption "
            "describing a speech recording, pick the 3 MOST FITTING "
            "application categories from the provided list.\n\n"
            "OUTPUT FORMAT (3 lines only):\n"
            "1: <category_id>\n"
            "2: <category_id>\n"
            "3: <category_id>\n\n"
            "RULES:\n"
            "- Output ONLY the 3 lines above, nothing else\n"
            "- Rank from most fitting (1) to least fitting (3)\n"
            "- Use the exact category_id from the list\n"
            "- All 3 must be different categories\n"
            "- Consider the speaker's tone, emotion, pace, setting, and "
            "content to determine fit"
        ),
        "user": (
            "Rich speech audio caption:\n{caption}\n\n"
            "Transcription of the spoken text:\n{extracted_lyrics}\n\n"
            "CANDIDATE CATEGORIES (pick 3 from these):\n{category_list}\n\n"
            "Pick the 3 most fitting categories:"
        ),
    },
}

# =============================================================================
# Pass 2: Request generation prompts
# =============================================================================
REQUEST_GENERATION_PROMPTS = {
    "v1": {
        "system": (
            "You are a creative assistant that generates diverse user "
            "requests for a text-to-speech (TTS) system. You are given:\n"
            "- A rich audio caption describing a speech recording\n"
            "- A transcription of the spoken text\n"
            "- An assigned application category\n\n"
            "Generate a natural user request that a real person would write "
            "when using TTS for that specific application.\n\n"
            "Use [TEXT] as a placeholder for the spoken text. Include "
            "[TEXT] exactly once.\n\n"
            "OUTPUT FORMAT (1 line only):\n"
            "REQUEST: <your generated user request containing [TEXT]>\n\n"
            "RULES:\n"
            "- Generate a request that fits the assigned application "
            "category naturally\n"
            "- Output ONLY the single REQUEST line, nothing else\n"
            "- Include [TEXT] exactly once\n"
            "- Keep the description part SHORT (5-30 words, not counting "
            "[TEXT])\n"
            "- Focus on attributes from the caption that are relevant to "
            "the assigned application\n"
            "- Do NOT use technical audio jargon (reverb, EQ, compression)\n"
            "- Sound like a real person requesting TTS for that specific "
            "purpose\n"
            "- Follow the style and position instructions"
        ),
        "user": (
            "Assigned application category: {application_category} "
            "({category_description})\n\n"
            "Rich speech audio caption:\n{caption}\n\n"
            "Transcription of the spoken text:\n{extracted_lyrics}\n\n"
            "Style: {style}\n"
            "Position: {position}\n\n"
            "Generate a user request for this application:"
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
            "help request — ask for help generating the speech",
            # Instruction-style
            "imperative — directly command the system to speak",
            "speech command — tell the system to read or say the text",
            "make command — tell the system to make it sound a certain way",
            # Description-style
            "descriptive — describe the voice and style as a scene",
            "contextual — add context around the speech",
            "scenario — set up an imaginary speaking situation",
            # Minimal-style
            "direct label — brief, tag-like, minimal words",
            "annotation — attach voice and style traits as a short note",
            # Question-style
            "questioning — ask how it would sound if spoken",
            "what-if — pose a hypothetical speaking scenario",
        ],
    },
}

PLACEHOLDER = "[TEXT]"

# =============================================================================
# Pass 1: Category selection stats and logic
# =============================================================================
_pass1_stats = {
    "success": 0,
    "no_response": 0,
    "bad_format": 0,
    "bad_category": 0,
}


def parse_category_selection(
    response: str,
    candidate_ids: List[str],
    example_id: str = "",
) -> Optional[List[str]]:
    """Parse the 3-line category selection output.

    Returns list of up to 3 valid category IDs, or None on failure.
    """
    if response is None:
        _pass1_stats["no_response"] += 1
        return None

    response = response.strip()
    candidate_set = set(candidate_ids)
    picked = []

    for ln in response.split("\n"):
        ln = ln.strip()
        if not ln:
            continue
        # Parse "1: category_id" or just "category_id"
        if ":" in ln:
            cat = ln.split(":", 1)[1].strip()
        else:
            cat = ln.strip()
        if cat in candidate_set and cat not in picked:
            picked.append(cat)

    if len(picked) < 1:
        _pass1_stats["bad_format"] += 1
        # print(
        #     f"[PASS1-DISCARD] id={example_id} reason=no_valid_picks\n"
        #     f"  RESPONSE: {response[:200]}"
        # )
        return None

    _pass1_stats["success"] += 1
    return picked


def build_category_query(
    sample: Dict[str, Any],
    prompts: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Build a Pass 1 query: select top-3 categories from 10 random ones."""
    caption = sample.get("caption", "")
    extracted_lyrics = sample.get(
        "extracted_lyrics", sample.get("whisper_lyrics", "")
    )
    example_id = sample.get("example_id", "")

    if not caption:
        return None

    # Randomly sample 10 categories
    candidates = random.sample(APPLICATION_CATEGORIES, NUM_CANDIDATE_CATEGORIES)
    candidate_ids = [cid for cid, _ in candidates]
    category_list = "\n".join(
        f"- {cid}: {desc}" for cid, desc in candidates
    )

    messages = [
        {"role": "system", "content": prompts["system"]},
        {
            "role": "user",
            "content": prompts["user"].format(
                caption=caption,
                extracted_lyrics=extracted_lyrics,
                category_list=category_list,
            ),
        },
    ]

    metadata = dict(sample)
    metadata["candidate_category_ids"] = candidate_ids

    return {
        "idx": example_id,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 128,
        "metadata": metadata,
    }


def process_category_result(
    result: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Process a Pass 1 category selection result.

    Parses top-3 picks, randomly selects 1, returns sample with
    assigned application_category.
    """
    response = result.get("response")
    metadata = result.get("metadata", {})
    example_id = metadata.get("example_id", "")
    candidate_ids = metadata.get("candidate_category_ids", [])

    picks = parse_category_selection(response, candidate_ids, example_id)
    if picks is None:
        return None

    # Randomly select one from the LLM's top picks
    chosen = random.choice(picks)

    # Remove internal metadata before passing downstream
    out = {k: v for k, v in metadata.items() if k != "candidate_category_ids"}
    out["application_category"] = chosen
    out["category_candidates"] = candidate_ids
    out["category_llm_picks"] = picks
    out["idx"] = result["idx"]
    return out


# =============================================================================
# Pass 2: Request generation stats and logic
# =============================================================================
_pass2_stats = {
    "success": 0,
    "no_response": 0,
    "bad_placeholder": 0,
    "too_short": 0,
    "too_long": 0,
    "bad_format": 0,
}

_category_counts: Counter = Counter()


def parse_request_output(
    response: str, example_id: str = ""
) -> Optional[str]:
    """Parse the single-line REQUEST: output format."""
    if response is None:
        _pass2_stats["no_response"] += 1
        # print(f"[PASS2-DISCARD] id={example_id} reason=no_response")
        return None

    response = response.strip()

    # Remove surrounding quotes if any
    if response.startswith('"') and response.endswith('"'):
        response = response[1:-1].strip()

    # Extract REQUEST: line
    request = None
    for ln in response.split("\n"):
        ln = ln.strip()
        if ln.upper().startswith("REQUEST:"):
            request = ln.split(":", 1)[1].strip()
            break

    # Fallback: if no REQUEST: prefix, treat entire response as request
    if request is None:
        if PLACEHOLDER in response and len(response.split("\n")) <= 2:
            request = response
        else:
            _pass2_stats["bad_format"] += 1
            # print(
            #     f"[PASS2-DISCARD] id={example_id} reason=bad_format\n"
            #     f"  RESPONSE: {response[:200]}"
            # )
            return None

    count = request.count(PLACEHOLDER)
    if count != 1:
        _pass2_stats["bad_placeholder"] += 1
        # print(
        #     f"[PASS2-DISCARD] id={example_id} reason=bad_placeholder "
        #     f"count={count}\n  REQUEST: {request}"
        # )
        return None

    desc_len = len(request) - len(PLACEHOLDER)
    if desc_len < 5:
        _pass2_stats["too_short"] += 1
        # print(
        #     f"[PASS2-DISCARD] id={example_id} reason=too_short "
        #     f"desc_len={desc_len}\n  REQUEST: {request}"
        # )
        return None

    if desc_len > 600:
        _pass2_stats["too_long"] += 1
        # print(
        #     f"[PASS2-DISCARD] id={example_id} reason=too_long "
        #     f"desc_len={desc_len}\n  REQUEST: {request}"
        # )
        return None

    _pass2_stats["success"] += 1
    return request


def build_request_query(
    sample: Dict[str, Any],
    prompts: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Build a Pass 2 query: generate user request for assigned category."""
    caption = sample.get("caption", "")
    example_id = sample.get("example_id", "")
    extracted_lyrics = sample.get(
        "extracted_lyrics", sample.get("whisper_lyrics", "")
    )
    application_category = sample.get("application_category", "")

    if not caption or not application_category:
        return None

    category_description = CATEGORY_DICT.get(application_category, "")
    position = random.choice(prompts["positions"])
    style = random.choice(prompts["styles"])

    messages = [
        {"role": "system", "content": prompts["system"]},
        {
            "role": "user",
            "content": prompts["user"].format(
                caption=caption,
                extracted_lyrics=extracted_lyrics,
                application_category=application_category,
                category_description=category_description,
                position=position,
                style=style,
            ),
        },
    ]

    metadata = dict(sample)
    metadata["prompt_position"] = position
    metadata["prompt_style"] = style

    return {
        "idx": example_id,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 512,
        "metadata": metadata,
    }


def process_request_result(
    result: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Process a Pass 2 request generation result."""
    response = result.get("response")
    metadata = result.get("metadata", {})
    lyrics = metadata.get(
        "extracted_lyrics", metadata.get("whisper_lyrics", "")
    )
    example_id = metadata.get("example_id", "")

    template = parse_request_output(response, example_id)
    if template is None:
        return None

    # Substitute placeholder with the extracted text (quoted)
    clean_lyrics = lyrics.strip('"')
    user_request = template.replace(PLACEHOLDER, f'"{clean_lyrics}"')
    user_request = user_request.replace('""', '"')

    category = metadata.get("application_category", "")
    _category_counts[category] += 1

    return {
        "example_id": example_id,
        "dataset": metadata.get("dataset", ""),
        "audio_path": metadata.get("audio_path", ""),
        "duration": metadata.get("duration", 0.0),
        "caption": metadata.get("caption", ""),
        "whisper_lyrics": metadata.get("whisper_lyrics", ""),
        "extracted_lyrics": metadata.get("extracted_lyrics", ""),
        "user_request": user_request,
        "application_category": category,
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


def _print_stats(title: str, stats: Dict[str, int]) -> None:
    total = sum(stats.values())
    if total == 0:
        return
    print("\n" + "=" * 60)
    print(f"{title}:")
    print(f"  {'Reason':<25} {'Count':>8} {'Percent':>8}")
    print("-" * 60)
    for reason, count in stats.items():
        pct = count / total * 100
        print(f"  {reason:<25} {count:>8} {pct:>7.1f}%")
    print("-" * 60)
    print(f"  {'Total':<25} {total:>8}")
    print("=" * 60)


async def main_async(args) -> None:
    cat_prompts = CATEGORY_SELECTION_PROMPTS.get(args.version)
    req_prompts = REQUEST_GENERATION_PROMPTS.get(args.version)
    if cat_prompts is None or req_prompts is None:
        raise ValueError(
            f"Version '{args.version}' not found. "
            f"Available: {list(CATEGORY_SELECTION_PROMPTS.keys())}"
        )
    print(f"Using prompt version: {args.version}")

    print(f"Loading input data from: {args.input_file}")
    samples = load_input_data(args.input_file)
    print(f"Loaded {len(samples)} samples")

    if args.num_samples > 0:
        samples = samples[: args.num_samples]
        print(f"Limited to {len(samples)} samples")

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "user_requests.jsonl")

    # Check for already-completed samples (final output)
    processed_ids = set()
    if os.path.exists(output_file) and args.resume:
        processed_ids = get_processed_indices(
            output_file, idx_key="example_id"
        )
        print(f"Resuming: {len(processed_ids)} already completed")

    # Determine which samples still need processing
    pending_samples = [
        s for s in samples
        if s.get("example_id", "") not in processed_ids
    ]
    print(f"Pending samples: {len(pending_samples)}")

    if not pending_samples:
        print("All samples already processed!")
        return

    base_urls = parse_vllm_urls(args.vllm_url)
    model = args.model or DEFAULT_MODEL
    workers_per_server = max(1, args.num_workers // len(base_urls))

    # =========================================================================
    # Pass 1: Category selection
    # =========================================================================
    print("\n" + "=" * 60)
    print("PASS 1: Category Selection (10 random -> LLM picks 3 -> random 1)")
    print("=" * 60)

    # Check for Pass 1 intermediate results
    pass1_file = os.path.join(args.output_dir, "pass1_categories.jsonl")
    pass1_done_ids = set()
    if os.path.exists(pass1_file) and args.resume:
        pass1_done_ids = get_processed_indices(
            pass1_file, idx_key="example_id"
        )
        print(f"Pass 1 resuming: {len(pass1_done_ids)} already done")
    else:
        if not args.resume:
            open(pass1_file, "w").close()
            open(output_file, "w").close()

    for key in _pass1_stats:
        _pass1_stats[key] = 0

    pass1_queries = []
    for sample in pending_samples:
        eid = sample.get("example_id", "")
        if eid in pass1_done_ids:
            continue
        query = build_category_query(sample, cat_prompts)
        if query is not None:
            pass1_queries.append(query)

    print(f"Pass 1 queries to process: {len(pass1_queries)}")

    if pass1_queries:
        processor1 = MassiveQueryProcessor(
            base_urls=base_urls,
            model=model,
            workers_per_server=workers_per_server,
            timeout=args.timeout,
            checkpoint_interval=10000,
        )
        await processor1.process_all(
            queries=pass1_queries,
            output_file=pass1_file,
            process_fn=process_category_result,
        )

    _print_stats("Pass 1 Validation Summary", _pass1_stats)

    # Load all Pass 1 results
    pass1_results = load_input_data(pass1_file)
    print(f"Pass 1 total results: {len(pass1_results)}")

    # =========================================================================
    # Pass 2: Request generation
    # =========================================================================
    print("\n" + "=" * 60)
    print("PASS 2: Request Generation (category-conditioned)")
    print("=" * 60)

    for key in _pass2_stats:
        _pass2_stats[key] = 0
    _category_counts.clear()

    pass2_queries = []
    for result in pass1_results:
        eid = result.get("example_id", "")
        if eid in processed_ids:
            continue
        query = build_request_query(result, req_prompts)
        if query is not None:
            pass2_queries.append(query)

    print(f"Pass 2 queries to process: {len(pass2_queries)}")

    if pass2_queries:
        processor2 = MassiveQueryProcessor(
            base_urls=base_urls,
            model=model,
            workers_per_server=workers_per_server,
            timeout=args.timeout,
            checkpoint_interval=10000,
        )
        await processor2.process_all(
            queries=pass2_queries,
            output_file=output_file,
            process_fn=process_request_result,
        )

    _print_stats("Pass 2 Validation Summary", _pass2_stats)

    final_count = len(
        get_processed_indices(output_file, idx_key="example_id")
    )
    print(f"\nDone! Total success: {final_count}/{len(samples)}")
    print(f"Output: {output_file}")

    if _category_counts:
        print("\n" + "=" * 60)
        print("Category Distribution:")
        print(f"  {'Category':<35} {'Count':>8} {'Percent':>8}")
        print("-" * 60)
        total_cat = sum(_category_counts.values())
        for cat, count in _category_counts.most_common():
            pct = count / total_cat * 100
            print(f"  {cat:<35} {count:>8} {pct:>7.1f}%")
        print("-" * 60)
        print(f"  {'Total':<35} {total_cat:>8}")
        print(f"  {'Unique categories':<35} {len(_category_counts):>8}")
        print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2: Generate TTS user requests (two-pass)."
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
