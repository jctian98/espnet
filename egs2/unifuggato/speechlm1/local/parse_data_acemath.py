import argparse
import random
from pathlib import Path

from datasets import load_dataset
from espnet2.speechlm.dialogue.dialogue_format import Dialogue, DialogueDataset

random.seed(42)

def get_parser():
    parser = argparse.ArgumentParser(description='Processing the manifest from ETTA')
    
    # Add arguments
    parser.add_argument(
        '--output_dir',
        type=Path,
        help='Output directory',
    )
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    acemath = load_dataset("nvidia/AceMath-Instruct-Training-Data", 'default')
    # (NOTE)Jinchuan: skip the math_sft subset
    for subset in ['general_sft_stage2', 'general_sft_stage1']:
        iterator = acemath[subset]
        dataset = DialogueDataset(task='audio_dialogue')
        valid_dataset = DialogueDataset(task='audio_dialogue')

        for idx, example in enumerate(iterator):
            example_name = f"acemath_{subset}_{idx}"
            dialogue = Dialogue(task="audio_dialogue")

            messages = example['messages']
            for message in messages:
                role = message['role']
                message = message['content']
                # QUESTION(Jinchuan): do we always apply loss over assistant?
                target = role == "assistant" 
                dialogue.add_segment(role, 'text_bpe', target, message)

            answer = example['answer']
            assert isinstance(answer, str)
            dialogue.add_segment('assistant', 'text_bpe', True, answer)

            dataset.add_dialogue(example_name, dialogue)
            if random.random() < 0.0001:
                valid_dataset.add_dialogue(example_name, dialogue)

        dataset.dump_dataset(args.output_dir / f"acemath_{subset}_train")
        valid_dataset.dump_dataset(args.output_dir / f"acemath_{subset}_valid")
    
    llama_nemotron = load_dataset(
        "nvidia/Llama-Nemotron-Post-Training-Dataset", 'SFT', 
        num_proc=8,
    )
    for subset in []:
    # for subset in ['math']:
        iterator = llama_nemotron[subset]
        dataset = DialogueDataset(task='audio_dialogue')
        valid_dataset = DialogueDataset(task='audio_dialogue')

        for idx, example in enumerate(iterator):
            example_name = f"ln_{subset}_{idx}"
            dialogue = Dialogue(task="audio_dialogue")

            # Specifical filtering
            if not 'Nano' in example['used_in_training']:
                continue
            
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

            dataset.add_dialogue(example_name, dialogue)
            if random.random() < 0.00005:
                valid_dataset.add_dialogue(example_name, dialogue)
            
            if idx % 10000 == 0:
                print(f'Finished {idx} examples for {subset}', flush=True)

        dataset.dump_dataset(args.output_dir / f"ln_{subset}_train")
        valid_dataset.dump_dataset(args.output_dir / f"ln_{subset}_valid")

if __name__ == "__main__":
    main()