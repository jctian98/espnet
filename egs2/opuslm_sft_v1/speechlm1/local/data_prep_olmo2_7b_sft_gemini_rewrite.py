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
from espnet2.speechlm.dialogue.dialogue_format import Dialogue, DialogueDataset
from espnet2.utils.types import str2bool


prompt = """
LLM Prompt for TTS-Friendly Response Processing
You will receive dialogue between a human user and an AI assistant. Your task is to revise the AI's latest response to be suitable for text-to-speech rendering.

Input Format:
CONTEXT: Previous dialogue history between human and AI
RESPONSE: The AI assistant's latest response that needs revision

Your Task:
Analyze the RESPONSE and determine if revision is needed. If the response is already TTS-friendly, leave it unchanged. Otherwise, revise it to:
1. Contain no more than 60 tokens.
2. Include only plain text and punctuation (no formatting, code, lists, or special characters).
3. Avoid all linebreaks, keeping text in a single continuous paragraph.
4. Replace symbolic representations with their spoken equivalents:
  * Numbers (20 → twenty)
  * Currency ($100 → one hundred dollars)
  * Percentages (7% → seven percent)
  * Mathematical symbols (× → multiplied by, - → minus)
5. Preserve the most important information and intent of the original response.
6. Maintain a natural, conversational tone.


Guidelines:
1. Leave responses untouched if they already meet all TTS requirements
2. Prioritize the direct answer to the user's question
3. Remove explanations, examples, and tangential information when necessary
4. Eliminate references to visual elements like formatting
5. Ensure the response sounds natural when read aloud

### CONTEXT ###
{}

### RESPONSE ###
{}

### REVISED RESPONSE ###
"""

"""
7. For reasoning questions, you need to keep the reasoning procedure.
8. For response about math, ONLY SUMMARIZE THE FINAL OUTPUT IN A SHORT SENTENCE.
9. For response about code, write a narrative description of the code.
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
        default=500,
        help="Number of queries in each chunk"
    )
    parser.add_argument(
        "--num_workers", 
        type=int,
        default=48,
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
    parser.add_argument(
        "--think_mode",
        type=str2bool,
        default=False,
        help="If the data contains thinking procedure. i.e., the original text response"
    )

    return parser

def build_query(messages):
    """ The last message are response, others are context """
    context = ""
    for msg in messages[:-1]:
        role, modality, target, content = msg
        assert modality == "text_bpe"

        context += f"[{role}]:\n{content}\n\n"
    
    role, modality, target, content = messages[-1]
    response = f"[{role}]:\n{content}"

    return context, response

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
        key, context, response = query
        
        query = prompt.format(context, response)

        try:
            llm_output = self.model.generate_content(query).text
        except:
            print(f"fail on query {key}")
            llm_output = ""

        # TTS is not robust to \n; it also cause bugs.
        llm_output = llm_output.replace("\n", " ")

        ans = {
            "key": key,
            "query": query,
            'response': llm_output,
        }


        return ans

def batch_process_queries(
        queries, 
        llm, 
        chunk_size: int = 100, 
        num_workers: int = 64,
        cache_file: str = "llm_cache.json"
    ):
    """Process queries in parallel with chunking and caching."""
    print(f'in total, there are {len(queries)} examples')

    # Load cache if it exists
    cache = {}
    if os.path.exists(cache_file):
        print('loading cache file', cache_file)
        with open(cache_file, 'r') as f:
            cache = json.load(f)
    print('cache size: ', len(cache))
    
    # Process queries in chunks
    for i in range(0, len(queries), chunk_size):
        chunk = queries[i:i+chunk_size]
        print(f"Processing chunk {i//chunk_size + 1}/{(len(queries)-1)//chunk_size + 1}")

        # Process chunk in parallel
        chunk = [x for x in chunk if x[0] not in cache]
        if len(chunk) == 0:
            continue
        else:
            print("size of this chunk: ", len(chunk))
        
        with multiprocessing.Pool(num_workers) as pool:
            chunk_results = pool.map(llm, chunk)
        
        for result, query in zip(chunk_results, chunk):
            key, response = result['key'], result['response']
            cache[key] = {'query': query[1], 'response': query[2], 'revised_response': response}
    
        # Save cache after each chunk
        writer = open(cache_file, 'wb')
        writer.write(
            json.dumps(cache, indent=4, ensure_ascii=False, sort_keys=False).encode(
                "utf_8"
            )
        )
        print(f'save the cache at {cache_file}', flush=True)
    
    return cache
    
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
            
            # only revise the assistant response
            if role == "assistant":
                key = f"{example_id}_turn{idx}_text"
                context, response = build_query(messages[:idx + 1])
                queries.append((key, context, response))
    
    # (3) init Gemini
    llm = GeminiAPI(
        args.gemini_model_id,
        args.gemini_project_id,
        prompt,
    )

    # (4) processing in batches
    cache_file = args.output_dir / 'cache.json'
    cache = batch_process_queries(
        queries,
        llm,
        args.chunk_size,
        args.num_workers,
        cache_file,
    )
    assert len(cache) == len(queries), "Some queries are not fulfilled"

    # (5) Save new dataset
    all_results = dict()
    for query in queries:
        key, _, text_response = query
        result = cache[key]
        speech_response = result['revised_response']
        all_results[key] = {"text": text_response, "speech": speech_response}
    
    dataset = DialogueDataset(task="text_dialogue")
    for example_id, messages in all_examples.items():
        good_example = True
        dialogue = Dialogue(task="text_dialogue")

        if messages[0][0] == "system":
            dialogue.add_segment(*messages[0])
            messages = messages[1:]

        for idx, msg in enumerate(messages):
            role, modality, target, content = msg

            if role != "assistant":
                dialogue.add_segment(role, modality, target, content)
            
            else:
                key = f"{example_id}_turn{idx}_text"
                text_response = all_results[key]['text']
                speech_response = all_results[key]['speech']

                # Fail, since the safety mechanism of Gimini is triggered.
                if speech_response.strip() == "":
                    good_example = False
            
                if args.think_mode:
                    dialogue.add_segment(role, modality, target, content)
                dialogue.add_segment(role, modality, target, speech_response)
        
        if good_example:
            dataset.add_dialogue(example_id, dialogue)
        else:
            print(f'avoid adding {example_id}. So cases failed.')
        
    dataset.dump_dataset(args.output_dir)
    print('done at ', args.output_dir)
                 
    
if __name__ == "__main__":
    main()


