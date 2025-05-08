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

    # Currently, we only find the everyday-conversations are suitable for 
    # spoken-based SFT
    smoltalk = load_dataset(
        "HuggingFaceTB/smoltalk",
        "everyday-conversations",
        streaming=True
    )['train']

    train_dataset = DialogueDataset(task="text_dialogue")
    valid_dataset = DialogueDataset(task="text_dialogue")

    for idx, example in enumerate(smoltalk):
        example_id = f"smoltalk_everyday_conversation_{idx}"
        dialogue = Dialogue(task="text_dialogue")

        # NOTE(Jinchuan): these conversations always start from meaningless words like
        # "Hi", "Hi How I can help you today?", which is not helpful to the system.
        for msg in example['messages'][2:]:
            dialogue.add_segment(
                role=msg['role'],
                modality="text_bpe",
                target=msg['role'] == "assistant",
                content=msg['content']
            )

        train_dataset.add_dialogue(example_id, dialogue)
        if random.random() < 0.005:
            valid_dataset.add_dialogue(example_id, dialogue)
        
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset.dump_dataset(output_dir / 'train')
    valid_dataset.dump_dataset(output_dir / 'valid')

if __name__ == "__main__":
    main()




    