import argparse
import json
import librosa
import os
import multiprocessing as mp

from pathlib import Path

def get_parser():
    parser = argparse.ArgumentParser(description='Processing the manifest from AF3')
    
    # Add arguments
    parser.add_argument(
        '--input_dir',
        type=Path,
        help='Input path that contains multiple manifest json files',
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        help='Output file for all subset',
    )
    parser.add_argument(
        '--prefix',
        type=str,
        help='The prefix to distinguish all examples and subsets',
    )

    return parser

def get_audio_length(path):
    """Get audio length in seconds using librosa"""
    try:
        return librosa.get_duration(filename=path)
    except Exception as e:
        print(f"Error processing {os.path.basename(path)}: {e}")
        return None

def get_audio_lengths(audio_paths, num_processes=None):
    """Process multiple audio files in parallel to get their lengths"""
    with mp.Pool(processes=num_processes or mp.cpu_count()) as pool:
        audio_lengths = pool.map(get_audio_length, audio_paths)
    
    retval = {
        audio_file: audio_length
        for audio_file, audio_length in zip(audio_paths, audio_lengths)
        if audio_length is not None and audio_length > 0 and " " not in audio_file
    }
    return retval

def dump_json_file(json_file, output_dir):
    # (1) parse json file
    examples = list()
    for line in open(json_file):
        line_dict = json.loads(line)
        examples.append(line_dict)
    print(f'json file {json_file} has {len(examples)} examples')
    
    # (2) find the lengths of all audio files
    all_audio = set([example['location'] for example in examples])
    all_audio_lengths = get_audio_lengths(all_audio)

    # (3) parse each example
    wav_scp_writer = open(output_dir / 'wav.scp', 'w')
    text_writer = open(output_dir / 'text', 'w')
    segments_writer = open(output_dir / 'segments', 'w')
    utt2spk_writer = open(output_dir / 'utt2spk', 'w')

    unique_map = set()
    success = 0
    for example in examples:
        # (3.1) check valid audio
        audio = example['location']
        if audio not in all_audio_lengths:
            print(f"The following example is discarded: {example}")
            continue
        
        # (3.2) check start and end
        if 'start' in example and 'end' in example:
            start, end = float(example['start']), float(example['end'])
        else:
            start, end = 0, all_audio_lengths[audio]
        
        if end > all_audio_lengths[audio]:
            print(f'The audio is overly long: {example}')
            continue
        
        # (3.3) process and write
        dataset_name = example['dataset']
        wav_name = Path(audio).stem
        wav_name = f"{dataset_name}_{wav_name}"

        clip_name = f"{wav_name}_{start}_{end}"
        while clip_name in unique_map:
            clip_name = clip_name + "_repeat"
        unique_map.add(clip_name)

        caption = example['captions']

        wav_scp_writer.write(f"{wav_name} {audio}\n")
        text_writer.write(f"{clip_name} {caption}\n")
        segments_writer.write(f"{clip_name} {wav_name} {start} {end}\n")
        utt2spk_writer.write(f"{clip_name} {clip_name}\n")

        success += 1
    
    print(f"Success: {success}/{len(examples)}")
    

def main():
    parser = get_parser()
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for json_file in list(args.input_dir.glob("*.json")):
        subset = args.prefix + "_" + json_file.stem
        output_dir = args.output_dir / subset
        output_dir.mkdir(parents=True, exist_ok=True)

        dump_json_file(json_file, output_dir)
    
if __name__ == "__main__":
    main()