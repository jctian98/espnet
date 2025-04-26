#!/usr/bin/env python3

# Copyright 2025 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import random

from pathlib import Path
from datasets import load_dataset
from espnet2.speechlm.dialogue.dialogue_format import Dialogue, DialogueDataset


random.seed(42)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download_dir", 
        type=Path, 
        help="Download datadir for TULU3"
    )
    parser.add_argument(
        "--output_dir", 
        type=Path, 
        help="Output data folder for training"
    )

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    download_dir=args.download_dir
    output_dir=args.output_dir

    # (1) create dataset objects
    olmo2_dpo = load_dataset(
        "allenai/olmo-2-1124-7b-preference-mix", 
        cache_dir=download_dir, 
        num_proc=6,
    )['train']
    train_dataset = DialogueDataset(task="text_dialogue")
    valid_dataset = DialogueDataset(task="text_dialogue")

    for example in olmo2_dpo:
        example_id = example['id'].replace("/", "_")
        dialogue = Dialogue(task="text_dialogue")
        is_valid_data = random.random() < 0.005
        for option in ['chosen', 'rejected']:
            
            messages = example[option]

            for idx, msg in enumerate(messages):
                dialogue.add_segment(
                    role=msg['role'],
                    modality='text_bpe',
                    target=(idx == len(messages) - 1),
                    content=msg['content']
                )
            
        train_dataset.add_dialogue(example_id, dialogue)
        if is_valid_data:
            valid_dataset.add_dialogue(example_id, dialogue)
        
    # (2) save dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset.dump_dataset(output_dir / 'train')
    valid_dataset.dump_dataset(output_dir / 'valid')
    
if __name__ == "__main__":
    main()


