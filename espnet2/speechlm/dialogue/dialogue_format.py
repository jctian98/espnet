#!/usr/bin/env python3

# Copyright 2025 Jinchuan Tian
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import json
import random
from pathlib import Path
from typing import List

from espnet2.speechlm.definitions import MODALITIES
from espnet2.fileio.read_text import read_2columns_text

class Dialogue:
    def __init__(self, task):
        self.segments = list()
        self.task = task
    
    def add_segment(
        self,
        role: str,
        modality: str,
        target: bool,
        content: str,
    ):
        if role not in [None, "system", "user", "assistant"]:
            raise ValueError(
                f"Role can only be None, system, user or assistant: {role}"
            )
    
        if modality not in MODALITIES:
            raise ValueError(f"unrecognized modality: {modality}")

        self.segments.append([role, modality, target, content])
    
    def to_str(self):
        string = ""
        for segment in self.segments:
            role, modality, target, content = segment
            string += f"role={role}, modality={modality}, target={target}, content={content}\n"

        return string.strip()

class DialogueDataset:
    def __init__(self, task):

        if task not in ["text_dialogue", "audio_dialogue", "audio_text_dialogue", "vision_dialogue"]:
            raise ValueError("dialogue support: text, audio, vision")
        self.task = task

        self.dialogues = dict()
    
    def __len__(self):
        return len(self.dialogues)
    
    def __contains__(self, item):
        return item in self.dialogues

    def add_dialogue(self, name: str, dialogue: Dialogue):
        assert name not in self.dialogues, f"Duplicate dialogue name: {name}"
        self.dialogues[name] = dialogue
    
    def __contains__(self, name):
        return name in self.dialogues

    def dump_dataset(
        self, 
        output_dir, 
        pack_size: int = 20000, 
        rank: int = 1,
    ):
        output_dir = Path(output_dir)

        (output_dir / 'data').mkdir(parents=True, exist_ok=True)
        index_file = str(output_dir / 'data' / f'dialogue.{rank}')
        index_writer = open(index_file, 'w')

        example_ids = list(self.dialogues.keys())
        pack_idx = 0
        all_utt_text_pairs = list()
        while pack_idx * pack_size < len(example_ids):
            start = pack_idx * pack_size
            end = min((pack_idx + 1) * pack_size, len(example_ids))
            pack_idx += 1

            pack = dict()
            for key in example_ids[start: end]:
                pack[key] = self.dialogues[key].segments

            pack_file = str(output_dir / 'data' / f'dialogue_rank{rank}_pack{pack_idx}.json')

            for key in pack:
                index_writer.write(f"{key} {pack_file}\n")

            pack_writer = open(pack_file, 'wb')
            pack_writer.write(
                json.dumps(pack, indent=4, ensure_ascii=False, sort_keys=False).encode(
                    "utf_8"
                )
            )
