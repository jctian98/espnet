#!/usr/bin/env python3

# Copyright 2025 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import json

from pathlib import Path
from espnet2.speechlm.dialogue.dialogue_format import Dialogue, DialogueDataset
from espnet2.fileio.read_text import read_2columns_text

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", 
        type=Path, 
        help="Download datadir for TULU3"
    )
    parser.add_argument(
        "--output_dir", 
        type=Path, 
        help="Output data folder for training"
    )
    parser.add_argument(
        "--task", 
        type=str, 
        choices=['audio_dialogue', 'audio_text_dialogue'],
        help="speech-to-speech dialogue / speech-to-text dialogue",
    )
    parser.add_argument(
        "--ready_audio", 
        type=Path,
        action="append",
        help="The prepared audio list",
    )

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    # (1) load all files
    all_json_files = set(
        line.strip().split()[1] for line in open(args.input_dir / 'dialogue')
    )
    raw_examples = dict()
    for json_file in all_json_files:
        raw_examples.update(json.load(open(json_file)))
    
    ready_audios = dict()
    for file in args.ready_audio:
        logging.info(f"loading ready audio file: {file}")
        ready_audios.update(read_2columns_text(file))
    
    # (2) process one-by-one
    dataset = DialogueDataset(task=args.task)

    for uid, segments in raw_examples.items():
        
        dialogue = Dialogue(task=args.task)

        if args.task == "audio_dialogue":
            prompt = ready_audios[f"{uid}_assistant_prompt"]
            dialogue.add_segment("system", "codec_ssl", False, prompt)
        
        if segments[0][0] == "system":
            dialogue.add_segment(*segments[0])
            segments = segments[1:]
        
        for s_idx, segment in enumerate(segments):
            role, modality, target, content = segment

            # audio first, text second
            audio = ready_audios[f"{uid}_turn{s_idx}_speech"]
            if role == "user":
                dialogue.add_segment("user", "codec_ssl", False, audio)
                dialogue.add_segment(f"user", "text_bpe", True, content)
            
            # text first, audio second
            elif role == "assistant":
                dialogue.add_segment(f"assistant", "text_bpe", True, content)
                dialogue.add_segment(f"assistant", "codec_ssl", True, audio)
            
        dataset.add_dialogue(uid, dialogue)
    
    dataset.dump_dataset(args.output_dir)

if __name__ == "__main__":
    main()



    
