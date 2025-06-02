import argparse
import random
import json

from multiprocessing import Pool
from pathlib import Path
from itertools import chain

from espnet2.speechlm.dialogue.dialogue_format import Dialogue, DialogueDataset

random.seed(42)

def get_parser():
    parser = argparse.ArgumentParser(description='Processing the manifest from ETTA')
    
    # Add arguments
    parser.add_argument(
        '--input_dir',
        type=Path,
        help='Input directory, the HF download dir',
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        help='Output directory',
    )
    parser.add_argument(
        '--n_split',
        type=int,
        default=150,
        help='number of parallel processing',
    )
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pool = Pool()

    for subset in ['code', 'math']:
    # for subset in ['safety', 'chat', 'science', 'code', 'math']:
        all_files = list((input_dir / subset).glob("*.jsonl"))
        futures = []
        for rank in range(args.n_split):
            name = f"ln_{subset}_{rank}"
            future = pool.apply_async(
                process_subset, 
                (all_files, name, rank, args.n_split)
            )
            futures.append(future)

        all_dialogues = []
        for rank, future in enumerate(futures):
            this_subset = future.get()
            print(f'rank: {rank}: find {len(this_subset)} examples', flush=True)
            all_dialogues.extend(this_subset)
        print(f'get all dialogues: {len(all_dialogues)}')
        
        dataset = DialogueDataset(task='audio_dialogue')
        for name, dialogue in all_dialogues:
            dataset.add_dialogue(name, dialogue)
        dataset.dump_dataset(output_dir / subset)
    
    pool.close()
    pool.join()
        
def process_subset(files, name, rank, world_size):
    retval = list()

    iterator = [open(file) for file in files]
    iterator = chain(*iterator)

    for idx, example in enumerate(iterator):
        if idx % world_size != rank:
            continue
        
        try:
            example = json.loads(example)
        except:
            print(f'Failed to parse line: {example}')
            continue

        # Only examples used on small models
        # if not 'Nano' in example['used_in_training']:
        #     continue
        
        dialogue = Dialogue(task='audio_dialogue')
        example_name = f"ln_{name}_{idx}"

        system_prompt = example['system_prompt']
        dialogue.add_segment('system', 'text_bpe', False, system_prompt)

        messages = example['input']
        for message in messages:
            role = message['role']
            message = message['content']
            # QUESTION(Jinchuan): do we always apply loss over assistant?
            target = role == "assistant" 
            dialogue.add_segment(role, 'text_bpe', target, message)

        answer = example['output']
        assert isinstance(answer, str)
        dialogue.add_segment('assistant', 'text_bpe', True, answer)

        retval.append((example_name, dialogue))

        if len(retval) % 1000 == 0:
            print(f"Rank: {rank} | Processed {name} {len(retval)} examples", flush=True)

    print(f"rank {rank} finishes with {len(retval)} examples", flush=True)
    return retval

if __name__ == "__main__":
    main()