#!/usr/bin/env python3

# Copyright 2025 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os
import argparse
import logging
import json

from pathlib import Path
from espnet2.speechlm.dialogue.dialogue_format import Dialogue, DialogueDataset
from espnet2.fileio.read_text import read_2columns_text
from espnet2.utils.types import str2bool

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
    parser.add_argument(
        "--think_mode", 
        type=str2bool,
        default=False,
        help="whether to use think mode",
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
        if os.path.exists(file):
            logging.info(f"loading ready audio file: {file}")
            ready_audios.update(read_2columns_text(file))
        else:
            logging.info(f'Cannot find {file}. Skip')
    
    # (2) process one-by-one
    dataset = DialogueDataset(task=args.task)

    for uid, segments in raw_examples.items():
        
        dialogue = Dialogue(task=args.task)
        
        if segments[0][0] == "system":
            dialogue.add_segment(*segments[0])
            segments = segments[1:]
        
        loop_length = 3 if args.think_mode else 2
        if not len(segments) % loop_length == 0:
            print(f"segments of {uid} is not valid in length: {segments}")
            continue
        
        for s_idx, segment in enumerate(segments):
            role, modality, target, content = segment

            # user_text -> assistant_text
            if args.task == "audio_text_dialogue":
                turn = (s_idx // 2) * 2 

                if s_idx % 2 == 0:
                    audio = ready_audios[f"{uid}_turn{turn}_speech"]
                    dialogue.add_segment("user", "codec_ssl", False, audio)
                    dialogue.add_segment("assistant", "text_bpe", True, content)
                
                elif s_idx % 2 == 1:
                    dialogue.add_segment("assistant", "text_bpe", True, content)
            
            # user_text -> assistant_text
            elif args.task == "audio_dialogue" and not args.think_mode:
                turn = (s_idx // 2) * 2 

                if s_idx % 2 == 0:
                    audio = ready_audios[f"{uid}_turn{turn}_speech"]
                    dialogue.add_segment("user", "codec_ssl", False, audio)
                    dialogue.add_segment("assistant", "text_bpe", True, content)
                
                elif s_idx % 2 == 1:
                    dialogue.add_segment("assistant", "text_bpe", True, content)
                    audio = ready_audios[f"{uid}_turn{turn+1}_speech"]
                    dialogue.add_segment("assistant", "codec_ssl", True, audio)
            
            # user_text -> assistant_think -> assistant_text
            elif args.task == "audio_dialogue" and args.think_mode:
                turn = (s_idx // 3) * 2

                if s_idx % 3 == 0:
                    audio = ready_audios[f"{uid}_turn{turn}_speech"]
                    dialogue.add_segment("user", "codec_ssl", False, audio)
                    dialogue.add_segment("assistant", "text_bpe", True, content)
                
                elif s_idx % 3 == 1:
                    dialogue.add_segment("assistant", "text_bpe", True, content)
                
                elif s_idx % 3 == 2:
                    dialogue.add_segment("assistant", "text_bpe", True, content)
                    audio = ready_audios[f"{uid}_turn{turn+1}_speech"]
                    dialogue.add_segment("assistant", "codec_ssl", True, audio)
            
            else:
                raise NotImplementedError
            
        dataset.add_dialogue(uid, dialogue)
    
    dataset.dump_dataset(args.output_dir)

if __name__ == "__main__":
    main()



    
