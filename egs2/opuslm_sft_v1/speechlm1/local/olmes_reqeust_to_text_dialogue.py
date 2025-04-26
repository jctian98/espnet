#!/usr/bin/env python3

# Copyright 2025 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import json
import logging

from pathlib import Path
from espnet2.speechlm.dialogue.dialogue_format import Dialogue, DialogueDataset

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_jsonl", 
        type=Path, 
        help="input_jsonl file from olmes requests"
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
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = DialogueDataset(task="text_dialogue")

    args.input_jsonl = "model_text_evaluation_v2/olmo-2-7b-1124-instruct/main/gsm8k__olmes/task-000-gsm8k-requests.jsonl"

    for line in open(args.input_jsonl):
        line_dict = json.loads(line)
        dialogue = Dialogue(task="text_dialogue")

        for msg in line_dict['request']['context']['messages']:
            dialogue.add_segment(
                role=msg["role"],
                modality="text_bpe",
                target=msg["role"] == "assistant",
                content=msg["content"],
            )
        dialogue.add_segment(
            role="assistant",
            modality="text_bpe",
            target=True,
            content=line_dict['doc']['answer'],
        )

        dataset.add_dialogue(line_dict['doc_id'], dialogue)
        
    dataset.dump_dataset(args.output_dir)

if __name__ == "__main__":
    main()
