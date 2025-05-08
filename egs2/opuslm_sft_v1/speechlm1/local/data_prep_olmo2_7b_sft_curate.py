#!/usr/bin/env python3

# Copyright 2025 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import random
import re

from langdetect import detect as lang_detect
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from espnet2.speechlm.dialogue.dialogue_format import Dialogue, DialogueDataset

random.seed(42)

all_subsets = set([
    'ai2-adapt-dev/numinamath_tir_math_decontaminated', 
    'ai2-adapt-dev/tulu_v3.9_sciriff_10k', 
    'allenai/tulu-3-sft-personas-math-grade', 
    'ai2-adapt-dev/tulu_v3.9_wildjailbreak_decontaminated_50k', 
    'ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980', 
    None, 
    'ai2-adapt-dev/tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k', 
    'ai2-adapt-dev/no_robots_converted', 
    'ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k', 
    'ai2-adapt-dev/oasst1_converted', 
    'ai2-adapt-dev/flan_v2_converted', 
    'ai2-adapt-dev/tulu_v3.9_personahub_math_interm_algebra_20k', 
    'ai2-adapt-dev/evol_codealpaca_heval_decontaminated', 
    'ai2-adapt-dev/personahub_code_v2_34999', 
    'ai2-adapt-dev/coconot_converted', 
    'ai2-adapt-dev/tulu_v3.9_wildchat_100k', 
    'ai2-adapt-dev/personahub_math_v5_regen_149960'
])

removed_subsets = set([
    'ai2-adapt-dev/tulu_v3.9_aya_100k', # multilingual
    'ai2-adapt-dev/tulu_v3.9_table_gpt_5k', # table
])

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download_dir", 
        type=Path, 
        help="Download datadir for OLMO2-SFT"
    )
    parser.add_argument(
        "--output_dir", 
        type=Path, 
        help="Output data folder for training"
    )
    parser.add_argument(
        "--task_type", 
        type=str,
        choices=["audio_text", "audio"],
        default="audio_text",
        help="Audio format"
    )
    parser.add_argument(
        "--max_user_len", 
        type=int,
        default=70,
        help="Maximum user instruction length in token"
    )
    parser.add_argument(
        "--max_assistant_len", 
        type=int,
        default=70,
        help="Maximum assistant response length in token, effective to audio dialogue"
    )
    parser.add_argument(
        "--max_overall_len", 
        type=int,
        default=2048,
        help="Maximum overall length in token, effective to audio dialogue"
    )

    return parser

desired_subsets = []

def is_english_or_digit(text):
    if text.strip().isdigit():
        return True

    try:
        return lang_detect(text) == "en"
    except:
        return False

def is_code_or_math(text):
    # Look for code patterns
    code_indicators = [
        r'def\s+\w+\s*\(', r'class\s+\w+', r'import\s+\w+',
        r'for\s+\w+\s+in\s+', r'if\s+.*:', r'while\s+.*:',
        r'<.*>', r'\{.*\}', r'\$.*\$',  # Math in LaTeX often uses $ delimiters
        r'\\begin\{.*\}', r'\\end\{.*\}'  # LaTeX environments
    ]
    
    for pattern in code_indicators:
        if re.search(pattern, text):
            return True
    
    # Check for high density of special characters common in code/math
    special_chars = sum(c in '{}[]()$\\=+*/<>:;._%#@!' for c in text)
    if len(text) > 0 and special_chars / len(text) > 0.1:  # Arbitrary threshold
        return True
        
    return False


def validate(
    dialogue, 
    tokenizer,
    args,
):
    # (1) First, check the length
    segment_lengths = []
    for segment in dialogue.segments:
        role, modality, target, content = segment

        segment_len = len(tokenizer(content)['input_ids'])
        
        # check the length
        if role == "user":
            if segment_len > args.max_user_len:
                return False
        elif role == "assistant":
            if segment_len > args.max_assistant_len and args.task_type == "audio":
                return False

        segment_lengths.append(segment_len)
    
    if sum(segment_lengths) > args.max_overall_len:
        return False
    
    # (2) Only keep the data: either English or code/math. A.k.a., remove multilingual
    for segment in dialogue.segments:
        role, modality, target, content = segment

        if not is_code_or_math(content) and not is_english_or_digit(content):
            return False

        if is_code_or_math(content) and role == "user":
            print('kill bad user input: ', content)
            return False
    
    return True

def main():
    parser = get_parser()
    args = parser.parse_args()

    download_dir=args.download_dir
    output_dir=args.output_dir

    # (1) create dataset objects
    olmo2_sft = load_dataset(
        "allenai/tulu-3-sft-olmo-2-mixture", 
        cache_dir=download_dir, 
        streaming=True,
        keep_in_memory=True,
    )['train']
    train_dataset = DialogueDataset(task="text_dialogue")
    valid_dataset = DialogueDataset(task="text_dialogue")

    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B", use_fast=True)

    for idx, example in enumerate(olmo2_sft):
        if idx % 1000 == 0:
            print(f'Done {idx} examples', flush=True)

        example_id = example['id']
        messages = example['messages']
        
        # Remove some subset
        source = example['source']
        if source in removed_subsets:
            continue
        
        dialogue = Dialogue(task="text_dialogue")

        for msg in messages:
            content = msg['content']
            role = msg['role']
            modality = "text_bpe"
            target = role == "assistant"

            assert role in ["user", "assistant", "system"], f"Invalid role: {role}"
            dialogue.add_segment(role, modality, target, content)
        
        # Not a valid example
        if not validate(dialogue, tokenizer, args):
            continue

        example_id = example_id.replace("/", "_")
        if not example_id in train_dataset:
            train_dataset.add_dialogue(example_id, dialogue)
            if random.random() < 0.005:
                valid_dataset.add_dialogue(example_id, dialogue)
        else:
            logging.info(f"Find duplicated example: {example_id}. Skip it.")

    # (2) save dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving {len(train_dataset)} examples.")
    train_dataset.dump_dataset(output_dir / 'train')
    valid_dataset.dump_dataset(output_dir / 'valid')
    
if __name__ == "__main__":
    main()


