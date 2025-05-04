#!/usr/bin/env python3

# Copyright 2025 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import csv
import json

from pathlib import Path
from espnet2.speechlm.dialogue.dialogue_format import Dialogue, DialogueDataset


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
    

    subsets = ["alpaca_eval", "llama_questions", "trivia_qa",  "web_questions"]
    csv_files = [
        download_dir / subset / f"{subset}.csv"
        for subset in subsets
    ]

    for subset, csv_file in zip(subsets, csv_files):
        reader = csv.DictReader(open(csv_file))
        
        write_dir = output_dir / subset
        write_dir.mkdir(exist_ok=True, parents=True)

        dataset = DialogueDataset(task="text_dialogue")
        data_dict = {}

        wav_writer = open(write_dir / 'wav.scp', 'w')

        if subset == "alpaca_eval":
            que_key = "instruction"
            ans_key = "output"
            aud_key = "audio_filename"
        
        elif subset == "llama_questions":
            que_key = "Questions"
            ans_key = "Answer"
            aud_key = "audio_filename"

        elif subset == "trivia_qa":
            que_key = "question"
            ans_key = "answer_normalized_value"
            aud_key = "audio_filename"
        
        elif subset == "web_questions":
            que_key = "question"
            ans_key = "answers"
            aud_key = "audio_filename"

        for row in reader:

            # Keep as a normal dict
            key = f"{subset}_{row[aud_key]}"
            question = row[que_key]

            answer = row[ans_key]
            assert isinstance(answer, str)

            audio = str((download_dir / subset / 'audios' / row[aud_key]).resolve())

            data_dict[key] = {
                "question": question,
                "answer": answer,
                "audio": audio
            }
            if subset == "trivia_qa":
                data_dict[key]["answer_normalized_aliases"] = row["answer_normalized_aliases"]

            dialogue = Dialogue(task="text_dialogue")
            dialogue.add_segment("user", "text_bpe", False, question)
            dialogue.add_segment("assistant", "text_bpe", True, answer)
            dataset.add_dialogue(key, dialogue)

            # Audio of the first turn.
            wav_writer.write(f"{key}_turn0_speech {audio}\n")
        
        dataset.dump_dataset(write_dir)

        json_writer = open(write_dir / "qa.json", 'wb')
        json_writer.write(
            json.dumps(data_dict, indent=4, ensure_ascii=False, sort_keys=False).encode(
                "utf_8"
            )
        )

if __name__ == "__main__":
    main()


