#!/usr/bin/env python3

import argparse
import logging
import librosa
import multiprocessing as mp
import torch
import json

from functools import partial
from pathlib import Path

from transformers import (
    Qwen2_5OmniForConditionalGeneration, 
    Qwen2_5OmniProcessor,
)
from qwen_omni_utils import process_mm_info


def get_parser():
    parser = argparse.ArgumentParser(
        description='Use ALM for audio caption',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input_file",
        type=Path,
        help='input scp file',
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        help='output jsonl file',
    )
    parser.add_argument(
        "--nproc",
        type=int,
        help='number of parallel processing',
    )
    parser.add_argument(
        "--shared_size",
        type=int,
        default=100,
        help='The size of each shared, in parallel processing',
    )
    parser.add_argument(
        "--hf_tag",
        type=str,
        help='tag for HF model',
    )
    return parser


def main():
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logfmt)

    parser = get_parser()
    args = parser.parse_args()

    all_shareds = list()
    buffer = []
    for line in open(args.input_file):
        buffer.append(line)
        if len(buffer) == args.shared_size:
            all_shareds.append(buffer)
            buffer = list()
    if len(buffer) > 0:
        all_shareds.append(buffer)
    
    mp.set_start_method('spawn', force=True)
    pool = mp.Pool(args.nproc)
    fn = partial(worker, model_name=args.hf_tag)

    futures = []
    for shared in all_shareds:
        future = pool.apply_async(fn, (shared,))
        futures.append(future)
    
    pool.close()  # Close the pool
    pool.join()  

    results = list()
    for future in futures:
        results.extend(future.get())
    
    writer = open(args.output_file, 'w')
    for line in results:
        writer.write(json.dumps(line) + "\n")

@torch.no_grad()
def worker(lines, model_name):
    if model_name == "Qwen/Qwen2.5-Omni-7B":
        
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).cuda()
        model.disable_talker()
        processor = Qwen2_5OmniProcessor.from_pretrained(model_name)

    else:
        raise NotImplementedError(f"Cannot support model: {model_name}")

    retval = list()
    for line in lines:
        retval.extend(process_one_line(line, model, processor))
    
    print(f'finish one chunck with {len(retval)} effective segments', flush=True)
    return retval

def process_one_line(
    line, 
    model, 
    processor,
):
    # configurations:
    sr = 16000
    chunck_size = 10

    sys_prompt = "You are a sound understanding and classification model."
    prompt = "Generate caption for this audio input."

    # load audio
    name, path = line.strip().split(maxsplit=1)
    try:
        audio, _ = librosa.load(path, sr=sr, mono=True)
    except:
        print(f"Cannot load audio file: {path}. Skip it")
        return list()
    
    retval = list()
    start = 0
    while start < audio.shape[0]:
        # preprocessing
        end = start + chunck_size * sr
        end = min(end, audio.shape[0])
        chunck = audio[start: end]

        # break if shorter than 1s. Should be the last segment
        if (end - start) < 1 * sr:
            break

        messages = [
            {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
            {"role": "user", "content": [
                    {"type": "audio", "audio": chunck},
                    {"type": "text", "text": prompt},
                ]
            },
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
        inputs = processor(
            text=text, 
            audio=audios, 
            images=images, 
            videos=videos, 
            return_tensors="pt", 
            padding=True, 
            use_audio_in_video=True,
        )
        inputs = inputs.to(model.device).to(model.dtype)

        # model inference
        output = model.generate(
            **inputs, 
            use_audio_in_video=True, 
            return_audio=False, 
            thinker_max_new_tokens=256, 
            thinker_do_sample=False
        )
        text = processor.batch_decode(
            output, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        text = text[0].split("assistant\n")[-1]

        ans = {
            "name": name,
            "path": path,
            "start": start / sr,
            "end": end / sr,
            "caption": text,
        }
        retval.append(ans)

        start = end

    return retval

if __name__ == "__main__":
    main()

