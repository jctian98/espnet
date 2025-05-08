#!/usr/bin/env python3

# Copyright 2025 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging
import json
import os
import multiprocessing
import vertexai

from functools import partial
from pathlib import Path
from vertexai.generative_models import GenerativeModel, GenerationConfig

# prompt = """
# You are a text-to-speech formatting assistant. Your task is to convert input text into a natural, conversational style suitable for speech synthesis. Follow these guidelines carefully:

# 1. ASSESSMENT: First determine if the text already sounds natural for speaking. If it does, return it unchanged.

# 2. LENGTH LIMIT: Ensure the output contains no more than 70 tokens. If longer, condense while preserving the core meaning.

# 3. FORMATTING CLEANUP: Remove all XML tags (like <tags>), formatting markers, and metadata. Keep only the essential spoken content.

# 4. SYMBOL CONVERSION: Replace symbolic representations with their spoken equivalents:
#    - "$100" → "one hundred dollars"
#    - "%" → "percent"
#    - "&" → "and"
#    - Numbers, dates, times in natural spoken form

# 5. CONVERSATIONAL STYLE: Make the text sound like natural speech. Use contractions where appropriate and avoid formal/technical language unless necessary.

# Below is the INPUT:
# """

prompt = """
Your task is to revise the following text to make it suitable for Text-to-Speech (TTS) rendering. Focus exclusively on these two aspects:

Remove all formatting symbols

Delete markdown markers (*, _, #, -, etc.)
Remove HTML tags (<p>, <br>, etc.)
Eliminate bullet points, numbered lists formatting
Remove extra whitespace, tabs, or multiple line breaks
Delete any other non-speech symbols


Replace symbolic representations with spoken equivalents

Convert "&" to "and"
Replace "$" with "dollars" (or appropriate currency)
Change "%" to "percent"
Convert "@" to "at"
Transform "+" to "plus"
Change "-" to "minus" when used mathematically
Convert "=" to "equals"
Replace "<" and ">" with "less than" and "greater than"
Convert other symbols like ©, ®, ™ to their spoken form
Spell out emoticons/emojis (":)" becomes "smile" or omit if appropriate)
Convert URLs to spoken form (e.g., "visit our website at example dot com")
Write out date formats consistently (05/06/2025 becomes "May sixth, twenty twenty-five")
Spell out abbreviations and acronyms when first mentioned



Do not alter the core content, meaning, or intent of the text. Maintain the original voice and style. Make only the changes necessary to ensure optimal TTS performance.
"""


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", 
        type=Path, 
        help="Input curated data"
    )
    parser.add_argument(
        "--output_dir", 
        type=Path, 
        help="Output data folder for training"
    )
    parser.add_argument(
        "--chunk_size", 
        type=int,
        default=1,
        help="Number of queries in each chunk"
    )
    parser.add_argument(
        "--num_workers", 
        type=int,
        default=1,
        help="Number of multiprocessing workers"
    )
    parser.add_argument(
        "--gemini_model_id", 
        type=str,
        default='gemini-2.0-flash',
        help="Gemini model id"
    )
    parser.add_argument(
        "--gemini_project_id",
        type=str,
        default="lti-sw-gemini",
        help="The project name of cortex AI to call Gemini"
    )

    return parser

class GeminiAPI:
    def __init__(self, model_id, project_id, prompt):
        self.prompt = prompt
        vertexai.init(project=project_id)
        
        config = GenerationConfig(
            max_output_tokens=512,
            temperature=1.0,
        )
        
        self.model = GenerativeModel(
            model_id,
            generation_config=config,
        )

    def __call__(self, query):
        query = prompt.format(query[1])
        response = self.model.generate_content(query)
        return response.text

def batch_process_queries(
        queries, 
        llm, 
        chunk_size: int = 100, 
        num_workers: int = 64,
        cache_file: str = "llm_cache.json"
    ):
    """Process queries in parallel with chunking and caching."""

    # Load cache if it exists
    cache = {}
    if os.path.exists(cache_file):
        print('loading cache file', cache_file)
        with open(cache_file, 'r') as f:
            cache = json.load(f)
    print('cache size: ', len(cache))
    
    results = []
    
    # Process queries in chunks
    for i in range(0, len(queries), chunk_size):
        chunk = queries[i:i+chunk_size]
        print(f"Processing chunk {i//chunk_size + 1}/{(len(queries)-1)//chunk_size + 1}")

        # Process chunk in parallel
        chunk = [x for x in chunk if x[0] not in cache]
        with multiprocessing.Pool(num_workers) as pool:
            chunk_results = pool.map(llm, chunk)
        for result, query in zip(chunk_results, chunk):
            cache[query[0]] = result
            print("before: ", query[1])
            print("after: ", result, flush=True)
    
        # Save cache after each chunk
        with open(cache_file, 'w') as f:
            json.dump(cache, f)
    
    return results


def main():
    parser = get_parser()
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # (1) load all json files
    all_json_files = set()
    for line in open(args.input_dir / 'dialogue'):
        json_file = line.strip().split()[1]
        all_json_files.add(json_file)
    
    all_examples = dict()
    for json_file in all_json_files:
        all_examples.update(json.load(open(json_file)))
    
    # (2) parse all queries.
    queries = list()
    for example_id, messages in all_examples.items():

        if messages[0][0] == "system":
            messages = messages[1:]

        for idx, msg in enumerate(messages):
            role, modality, target, content = msg

            assert modality == "text_bpe"
            
            # Only revise the user input
            if role == "user":
                assert target == False
                key = f"{example_id}_turn{idx}_text"
                queries.append((key, content))
    
    # (3) init Gemini
    llm = GeminiAPI(
        args.gemini_model_id,
        args.gemini_project_id,
        prompt,
    )

    # (4) processing in batches
    cache_file = args.output_dir / 'cache.json'
    results = batch_process_queries(
        queries,
        llm,
        args.chunk_size,
        args.num_workers,
        cache_file,
    )


    
if __name__ == "__main__":
    main()


