#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Stage 2: Generate persona-based user requests for role-play TTS.

Given a rich audio caption and transcription, generate a user request
containing a fictional character persona + verbatim transcription via
[TEXT] placeholder. The persona is derived from the caption's voice
characteristics but expressed as a character description.
"""

import argparse
import asyncio
import json
import os
import random
import re
import threading
from collections import Counter
from typing import Any, Dict, List, Optional

from sft_vllm_client import (
    DEFAULT_MODEL,
    MassiveQueryProcessor,
    get_processed_indices,
    parse_vllm_urls,
)

# =============================================================================
# Version-specific prompts for persona-based user request generation
# =============================================================================
PROMPTS_BY_VERSION = {
    "v1": {
        "system": (
            "You are a creative assistant that generates diverse user "
            "requests for a role-play text-to-speech system. Given a rich "
            "audio caption and specific style instructions, generate a user "
            "request that provides a FICTIONAL CHARACTER PERSONA along with "
            "text to speak.\n\n"
            "Use [TEXT] as a placeholder for the transcription. Include "
            "[TEXT] exactly once.\n\n"
            "CRITICAL RULES:\n"
            "- The persona describes a CHARACTER, not voice features\n"
            "- BAD: 'a warm female voice with a British accent, age 50'\n"
            "- GOOD: 'Margaret, a retired schoolteacher from Cornwall who "
            "loves telling stories to her grandchildren'\n"
            "- The persona should be a brief character description that "
            "IMPLIES the voice characteristics found in the caption\n"
            "- Output ONLY the user request\n"
            "- Include [TEXT] exactly once\n"
            "- Keep the persona description part between 20-100 words\n"
            "- Do NOT use technical audio jargon\n"
            "- Follow the style and diversity instructions closely\n\n"
            "NAME DIVERSITY (IMPORTANT):\n"
            "- NEVER reuse common default names. Specifically AVOID: "
            "Elias, Elara, Elena, Eleanor, Clara, Daniel, Amir, Amara, "
            "Kai, Luna, Maya, Liam, Noah, Sofia, Omar\n"
            "- Each persona MUST have a UNIQUE, uncommon name\n"
            "- Draw from the FULL global spectrum of names — do not "
            "cluster around any single culture or region\n"
            "- Vary titles: not everyone is 'Dr.' or 'Professor'\n\n"
            "OCCUPATION DIVERSITY (IMPORTANT):\n"
            "- Do NOT default to professors, doctors, archivists, "
            "historians, theologians, or clergy\n"
            "- Use diverse occupations: chef, pilot, firefighter, farmer, "
            "mechanic, nurse, athlete, musician, artist, carpenter, "
            "fisherman, taxi driver, florist, barber, welder, etc.\n\n"
            "FRAMING RULES:\n"
            "- Always frame the request as a DIRECT INSTRUCTION to a TTS "
            "system (2nd person), not a 3rd-person narration\n"
            "- BAD: 'Nikko is a tech forager. He responds: [TEXT]'\n"
            "- GOOD: 'As Nikko, a tech forager, say: [TEXT]'\n"
            "- The transcription [TEXT] should come LAST in the request; "
            "do not add persona description after [TEXT]"
        ),
        "user": (
            "Rich audio caption:\n{caption}\n\n"
            "Style instructions:\n"
            "- Framing: {style}\n"
            "- Diversity nudge: {diversity}\n\n"
            "Create a fictional character persona that would naturally have "
            "the voice described in the caption. Then combine it with [TEXT] "
            "as the words to speak.\n\n"
            "Generate the user request:"
        ),
        "styles": [
            # Framing styles
            "Roleplay intro -- introduce the character and ask them to say "
            "the text (e.g., 'Imagine you are [persona]. Say: [TEXT]')",
            "Direct persona assignment -- assign a character directly "
            "(e.g., 'As [persona], speak the following: [TEXT]')",
            "Casting direction -- frame as a casting call or audition "
            "(e.g., 'For the role of [persona], deliver this line: [TEXT]')",
            "Scene-setting -- set up a scene where the character speaks "
            "(e.g., '[Persona] is sitting by the fire. They say: [TEXT]')",
            "Character name only -- just name and minimal context "
            "(e.g., '[Name], a [brief descriptor]: [TEXT]')",
            "Storytelling frame -- narrate a moment with the character "
            "(e.g., '[Persona] takes a deep breath and says [TEXT]')",
            "Voice acting direction -- direct a voice actor as the character "
            "(e.g., 'Voice this as [persona] would say it: [TEXT]')",
            "Dialogue script -- format as a script with character name "
            "(e.g., '[CHARACTER NAME] ([brief description]): [TEXT]')",
            "Interview setup -- character being interviewed "
            "(e.g., '[Persona] is asked a question. They respond: [TEXT]')",
            "Monologue frame -- character giving a monologue or speech "
            "(e.g., '[Persona] stands before the crowd and declares: "
            "[TEXT]')",
            "Letter or message -- character writing/recording a message "
            "(e.g., '[Persona] records a voice message: [TEXT]')",
            "Memory or flashback -- character recalling something "
            "(e.g., '[Persona] remembers and whispers: [TEXT]')",
        ],
        "diversity_nudges": [
            "Create a persona with a non-Western cultural background "
            "(e.g., South Asian, East Asian, African, Middle Eastern, "
            "Latin American)",
            "Focus on the character's professional identity and how it "
            "shapes their speech (e.g., doctor, chef, mechanic, teacher, "
            "artist)",
            "Emphasize age and life stage in the persona (e.g., teenager, "
            "young adult, middle-aged, elderly)",
            "Highlight a distinctive personality trait that affects speech "
            "(e.g., nervous, confident, sarcastic, gentle, enthusiastic)",
            "Set the character in a specific regional or geographic context "
            "(e.g., rural village, coastal town, mountain community, big "
            "city)",
            "Give the character a formal or ceremonial speaking context "
            "(e.g., giving a toast, reading an announcement, teaching a "
            "class)",
            "Make the persona someone in an emotional moment (e.g., "
            "reuniting with family, receiving good news, feeling nostalgic)",
            "Create a character defined by a hobby or passion that colors "
            "their speech (e.g., poet, sports fan, gardener, musician)",
            "Focus on a relationship dynamic (e.g., parent to child, "
            "mentor to student, old friends meeting again)",
            "Give the character an unusual or creative occupation "
            "(e.g., lighthouse keeper, street performer, radio host, "
            "archaeologist)",
            "Set the character in a historical or period-specific context "
            "(e.g., 1920s jazz singer, Victorian scholar, frontier "
            "settler)",
            "Create a character with a contrasting personality and "
            "profession (e.g., a tough construction worker who writes "
            "poetry, a shy librarian who does stand-up comedy)",
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


PLACEHOLDER = "[TEXT]"

# Pattern to detect third-person framing (narration about a character rather
# than a direct instruction to the TTS system).  Matches sentences like:
#   "Nikko is a tech forager. He responds: [TEXT]"
#   "Maria, a florist, says: [TEXT]"
# but NOT direct instructions like:
#   "As Nikko, say: [TEXT]"
#   "Imagine you are Maria. Say: [TEXT]"
_THIRD_PERSON_RE = re.compile(
    r"^[A-Z][a-z]+(?:\s[A-Z][a-z]+)*"  # Name at start
    r"(?:,\s|\s+)"  # separator
    r"(?:is |was |has |had |sits |stands |walks |looks |takes |turns |"
    r"leans |pauses |glances |speaks |replies |responds |recalls |"
    r"remembers |whispers |mutters |sighs |smiles |laughs |nods )",
    re.MULTILINE,
)

# --- Name frequency cap ---
# Reject results if the character name already appears too many times.
# This prevents the LLM from latching onto a single "favorite" name.
_NAME_RE = re.compile(
    r"^(?:As |Imagine (?:you are |yourself as )?|For the role of |"
    r"Voice this as |Speak as |Embody |Channel |Playing (?:as )?|"
    r"\[?)([A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?)",
)
_name_counts: Counter = Counter()
_name_lock = threading.Lock()
# Max times any single first name can appear (0.5% of dataset as a cap,
# with a minimum floor of 3 to avoid over-filtering on small runs)
NAME_MAX_COUNT = 3  # will be updated in main_async based on dataset size


def _extract_first_name(request: str) -> Optional[str]:
    """Extract the character's first name from a user request."""
    m = _NAME_RE.match(request.strip())
    if m:
        return m.group(1).lower()
    return None


# Validation statistics (reset per run in main_async)
_stats = {
    "success": 0,
    "no_response": 0,
    "bad_placeholder": 0,
    "too_short": 0,
    "too_long": 0,
    "transcription_mismatch": 0,
    "trailing_narration": 0,
    "third_person": 0,
    "name_overused": 0,
}


def validate_request_template(
    response: str, transcription: str, example_id: str = ""
) -> Optional[str]:
    """Validate the LLM-generated request template containing [TEXT]."""
    if response is None:
        _stats["no_response"] += 1
        # print(f"[DISCARD] example_id={example_id} reason=no_response")
        return None

    response = response.strip()

    # Remove surrounding quotes if present
    if response.startswith('"') and response.endswith('"'):
        response = response[1:-1].strip()

    # Check that [TEXT] appears exactly once
    count = response.count(PLACEHOLDER)
    if count != 1:
        _stats["bad_placeholder"] += 1
        # print(
        #     f"[DISCARD] example_id={example_id} reason=bad_placeholder "
        #     f"count={count}\n  RESPONSE: {response}"
        # )
        return None

    # Check length of the non-TEXT portion (20-500 chars)
    desc_len = len(response) - len(PLACEHOLDER)
    if desc_len < 20:
        _stats["too_short"] += 1
        # print(
        #     f"[DISCARD] example_id={example_id} reason=too_short "
        #     f"desc_len={desc_len}\n  RESPONSE: {response}"
        # )
        return None

    if desc_len > 500:
        _stats["too_long"] += 1
        # print(
        #     f"[DISCARD] example_id={example_id} reason=too_long "
        #     f"desc_len={desc_len}\n  RESPONSE: {response}"
        # )
        return None

    # After substitution, verify transcription appears verbatim
    clean_text = transcription.strip('"')
    substituted = response.replace(PLACEHOLDER, clean_text)
    if clean_text.lower() not in substituted.lower():
        _stats["transcription_mismatch"] += 1
        return None

    # Reject if significant narration appears after [TEXT]
    text_pos = response.find(PLACEHOLDER)
    after_text = response[text_pos + len(PLACEHOLDER) :].strip().strip('".,;:')
    if len(after_text) > 30:
        _stats["trailing_narration"] += 1
        return None

    # Reject third-person narration framing
    # Get the part before [TEXT] to check framing style
    before_text = response[:text_pos].strip()
    if _THIRD_PERSON_RE.match(before_text):
        _stats["third_person"] += 1
        return None

    _stats["success"] += 1
    return response


def build_request_query(
    sample: Dict[str, Any],
    prompts: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Build a query for persona-based user request generation."""
    caption = sample.get("caption", "")
    whisper_lyrics = sample.get("whisper_lyrics", "")
    example_id = sample.get("example_id", "")

    if not caption or not whisper_lyrics:
        return None

    # Randomly select style and diversity nudge
    style = random.choice(prompts["styles"])
    diversity = random.choice(prompts["diversity_nudges"])

    messages = [
        {"role": "system", "content": prompts["system"]},
        {
            "role": "user",
            "content": prompts["user"].format(
                caption=caption,
                style=style,
                diversity=diversity,
            ),
        },
    ]

    # Store style metadata for output
    sample = dict(sample)
    sample["prompt_style"] = style
    sample["prompt_diversity"] = diversity

    return {
        "idx": example_id,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 512,
        "metadata": sample,
    }


def process_request_result(
    result: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Process a user request generation result."""
    response = result.get("response")
    metadata = result.get("metadata", {})
    whisper_lyrics = metadata.get("whisper_lyrics", "")
    example_id = metadata.get("example_id", "")

    template = validate_request_template(
        response, whisper_lyrics, example_id
    )
    if template is None:
        return None

    # Substitute placeholder with the exact transcription
    clean_text = whisper_lyrics.strip('"')
    user_request = template.replace(PLACEHOLDER, f'"{clean_text}"')
    # Collapse any remaining "" from LLM template already quoting [TEXT]
    user_request = user_request.replace('""', '"')


    return {
        "example_id": example_id,
        "dataset": metadata.get("dataset", ""),
        "audio_path": metadata.get("audio_path", ""),
        "duration": metadata.get("duration", 0.0),
        "caption": metadata.get("caption", ""),
        "whisper_lyrics": metadata.get("whisper_lyrics", ""),
        "extracted_lyrics": metadata.get("extracted_lyrics", ""),
        "wer": metadata.get("wer", 0.0),
        "user_request": user_request,
        "prompt_style": metadata.get("prompt_style", ""),
        "prompt_diversity": metadata.get("prompt_diversity", ""),
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

    # Reset name frequency counter and set cap
    # Cap each name to at most 0.5% of total samples (min 3)
    global NAME_MAX_COUNT
    NAME_MAX_COUNT = max(3, len(samples) // 200)
    _name_counts.clear()
    print(f"Name frequency cap: {NAME_MAX_COUNT} per name")

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
        description="Generate persona-based user requests for role-play TTS."
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
