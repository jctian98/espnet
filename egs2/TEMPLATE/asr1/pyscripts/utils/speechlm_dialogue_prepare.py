#!/usr/bin/env python3

# Copyright 2025 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import json
import re
import random

from pathlib import Path
from functools import partial
from espnet2.fileio.read_text import read_2columns_text

random.seed(42)

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
        "--user_prompt_list", 
        type=Path,
        default=None,
        help="prompt for user speech simulation ",
    )
    parser.add_argument(
        "--assistant_prompt_list", 
        type=Path, 
        default=None,
        help="prompt for user speech simulation ",
    )
    parser.add_argument(
        "--ready_audio_list", 
        type=Path, 
        default=None,
        help="audio that already exists",
    )

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    # (1) load all files
    if args.user_prompt_list is not None:
        user_prompts = read_2columns_text(args.user_prompt_list)
        user_prompts = list(user_prompts.values())
    else:
        user_prompts = None
    
    if args.assistant_prompt_list is not None:
        assistant_prompts = read_2columns_text(args.assistant_prompt_list)
        assistant_prompts = list(assistant_prompts.values())
    else:
        assistant_prompts = None
    
    if args.ready_audio_list is not None:
        ready_audio = read_2columns_text(args.ready_audio_list)
    else:
        ready_audio = None
    
    all_json_files = set(
        line.strip().split()[1] for line in open(args.input_dir / 'dialogue')
    )
    raw_examples = dict()
    for json_file in all_json_files:
        raw_examples.update(json.load(open(json_file)))
    
    # (2) build writer
    # (2.1) gen_dir: directions for TTS generation
    gen_dir = args.output_dir / 'generation'
    gen_dir.mkdir(exist_ok=True, parents=True)
    gen_text_writer = open(gen_dir / 'text', 'w')
    gen_spk_writer = open(gen_dir / 'utt2spk', 'w')
    gen_wav_writer = open(gen_dir / 'wav.scp', 'w')

    # (2.2) tok_dir: directions for tokenization
    tok_dir = args.output_dir / 'tokenization'
    tok_dir.mkdir(exist_ok=True, parents=True)
    tok_wav_writer = open(tok_dir / 'wav.scp', 'w')

    # (2.3) ark_dir: directions for examples that are already in ark format
    ark_dir = args.output_dir / 'ark'
    ark_dir.mkdir(exist_ok=True, parents=True)
    ark_wav_writer = open(ark_dir / 'wav.scp', 'w')

    save_fn = partial(
        save_adaptive,
        gen_text_writer=gen_text_writer,
        gen_spk_writer=gen_spk_writer,
        gen_wav_writer=gen_wav_writer,
        tok_wav_writer=tok_wav_writer,
        ark_wav_writer=ark_wav_writer,
    )

    # (3) process each example
    for uid, segments in raw_examples.items():
        # (3.1) assistant prompt only if speech-to-speech dialogue;
        #       user prompt always exists
        if args.task == "audio_dialogue":
            if f"{uid}_assistant_prompt" in ready_audio:
                assistant_prompt = ready_audio[f"{uid}_assistant_prompt"]
            else:
                assistant_prompt = random.choice(assistant_prompts)
        else:
            assistant_prompt = None

        if f"{uid}_user_prompt" in ready_audio:
            user_prompt = ready_audio[f"{uid}_user_prompt"]
        else:
            user_prompt = random.choice(user_prompts)
        
        # (3.2) text prompt is not relavent to this
        if segments[0][0] == "system":
            segments = segments[1:]
        
        assistant_contents = []
        user_contents = []
        for s_idx, segment in enumerate(segments):
            segment_id = f"{uid}_turn{s_idx}_speech"
            role, modality, _, content = segment

            # NOTE(Jinchuan): content is either an audio path or a text transcription.
            # audio path contains no \n. text transcription should not have \n.
            # This ensures TTS can always work ok.
            content = content.replace("\n", " ")

            if role == "assistant" and args.task == "audio_text_dialogue":
                continue
            
            if role == "assistant":
                prompt = assistant_prompt
            else:
                prompt = user_prompt
            
            if segment_id in ready_audio:
                content = ready_audio[segment_id]
            elif role == "assistant":
                assistant_contents.append(content)
            elif role == "user":
                user_contents.append(content)
            
            save_fn(segment_id, content, prompt)

        if len(assistant_contents) > 0:
            save_fn(f"{uid}_assistant_prompt", assistant_prompt)
        if len(user_contents) > 0:
            save_fn(f"{uid}_user_prompt", user_prompt)


def save_adaptive(
    segment_id,
    content,
    prompt=None,
    gen_text_writer=None,
    gen_spk_writer=None,
    gen_wav_writer=None,
    tok_wav_writer=None,
    ark_wav_writer=None,
):
    # already ark file
    assert isinstance(content, str)

    if re.match(r".*\.ark:\d+$", content):
        ark_wav_writer.write(f"{segment_id} {content}\n")
    
    # audio scp file or pipe scp file
    elif content.endswith(".wav") or content.endswith("|") or content.endswith(".mp3"):
        tok_wav_writer.write(f"{segment_id} {content}\n")
    
    else:
        assert re.match(r".*\.ark:\d+$", prompt), "Prompt should only be kaldi ark"
        gen_text_writer.write(f"{segment_id} {content}\n")
        gen_spk_writer.write(f"{segment_id} {prompt}\n")
        gen_wav_writer.write(f"{segment_id} {prompt}\n")
    

if __name__ == "__main__":
    main()