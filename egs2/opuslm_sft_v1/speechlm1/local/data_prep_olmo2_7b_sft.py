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
        help="Download datadir for OLMO2-SFT"
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
    tulu3 = load_dataset(
        "allenai/tulu-3-sft-olmo-2-mixture", 
        cache_dir=download_dir, 
        streaming=True,
    )['train']
    train_dataset = DialogueDataset(task="text_dialogue")
    valid_dataset = DialogueDataset(task="text_dialogue")

    for example in tulu3:
        example_id = example['id']
        print('processing id: ', example_id)
        messages = example['messages']
        
        dialogue = Dialogue(task="text_dialogue")

        for msg in messages:
            content = msg['content']
            role = msg['role']
            modality = "text_bpe"
            target = role == "assistant"

            assert role in ["user", "assistant", "system"], f"Invalid role: {role}"
            dialogue.add_segment(role, modality, target, content)

        if not example_id in train_dataset:
            train_dataset.add_dialogue(example_id, dialogue)
            if random.random() < 0.005:
                valid_dataset.add_dialogue(example_id, dialogue)
        else:
            logging.info(f"Find duplicated example: {example_id}. Skip it.")
        
    # (2) save dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset.dump_dataset(output_dir / 'train')
    valid_dataset.dump_dataset(output_dir / 'valid')
    
if __name__ == "__main__":
    main()


