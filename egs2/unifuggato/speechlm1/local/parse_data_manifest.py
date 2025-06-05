import argparse
import json
import os
import librosa
import shutil
import multiprocessing as mp
from functools import partial
from pathlib import Path

def get_parser():
    parser = argparse.ArgumentParser(description='Processing the manifest from ETTA')
    
    # Add arguments
    parser.add_argument(
        '--manifests',
        nargs="+",
        type=str,
        help="input manifest files, as a list"
    )
    parser.add_argument(
        '--prefix',
        type=str,
        help='Prefix of each dataset',
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        help='Output directory',
    )

    return parser

def main():
    parser = get_parser()
    args =parser.parse_args()

    for manifest in args.manifests:
        process_one_manifest(manifest, args.output_dir, args.prefix)

def process_one_manifest(manifest, output_dir, prefix):
    # (1) subset name
    print(f'processing {manifest}', flush=True)
    subset_name = Path(manifest).stem
    if subset_name == "train":
        subset_name = Path(manifest).parent.stem
    output_dir = output_dir / f"{prefix}_{subset_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if (output_dir / '.done').exists():
        print('This dataset has been processed already')
        return
    
    # (2) load whole dict 
    try:
        json_list = json.load(open(manifest))
    except:
        json_list = [json.loads(line) for line in open(manifest)]
    
    # (3) find all audio
    all_audio = set()
    for example in json_list:
        if 'location' in example:
            audio = example['location']
        elif 'sound' in example:
            audio = example['sound']
        else:
            raise ValueError(f"Cannot find audio in example: {example}")
        
        all_audio.add(audio)
    print(f"Find unique audio files: {len(all_audio)}")

    # (4) Validation: existance and length
    cache = output_dir / 'cache'
    audio_metadata = validate_audios(all_audio, cache)
    print(f'{len(audio_metadata)} audio files are valid')

    # (5) write wave folder
    wav_scp_writer = open(output_dir / 'wav.scp', 'w')
    segments_writer = open(output_dir / 'segments', 'w')
    dedup = set()
    
    for idx, example in enumerate(json_list):
        example_id = f"{prefix}_{subset_name}_{idx}"

        # (5.1) audio file and metadata
        if 'location' in example:
            audio = example['location']
        elif 'sound' in example:
            audio = example['sound']
        else:
            raise ValueError(f"Cannot find audio in example: {example}")
        
        if audio not in audio_metadata:
            print(f"Skip processing example: {example}")
            continue
        real_path, real_length = audio_metadata[audio]

        # (5.2) audio length
        if 'start' in example and 'end' in example:
            start, end = example['start'], example['end']
        elif 'duration' in example and example['duration']:
            start, end = 0, example['duration']
        else:
            start, end = 0, real_length
        

        if (isinstance(start, str) and start.isdigit()) or isinstance(start, int):
            start = float(start)
        if (isinstance(end, str) and end.isdigit()) or isinstance(end, int):
            end = float(end)
        if not (isinstance(start, float) and isinstance(end, float)):
            print(f"example has bad start and/or end time-stamp: {example}", type(start), type(end))
            continue
        
        if end > real_length:
            end =  real_length
        
        if start > end:
            print(f"Example has wrong length: {example}")
            continue

        # (5.3) write segments and wav.scp
        if real_path not in dedup:
            wav_scp_writer.write(f"{real_path} {real_path}\n")
            dedup.add(real_path)
        segments_writer.write(f"{example_id} {real_path} {start} {end}\n")
    
    # (6) mark as finished
    (output_dir / '.done').touch()


def validate_audio(path, cache):
    # (1) If no such file, return None
    if not Path(path).exists():
        print(f"path : {path} doesn't exist.")
        return None
    
    # (2) If path contains space, make a copy of it to cache
    if " " in path:
        example_name = Path(path).stem.replace(" ", "-")
        example_path = (cache / example_name).resolve()
        if not example_path.exists():
            shutil.copy(path, example_path)
            print(f'Copy the file path with space: {path} -> {example_path}')
        path = example_path

    # (3) get the duration
    try:
        audio_len = librosa.get_duration(path=path)
        return path, audio_len
    except:
        return None

def validate_audios(all_audio, cache, num_processes=None):
    cache.mkdir(parents=True, exist_ok=True)
    fn = partial(validate_audio, cache=cache)
    with mp.Pool(processes=num_processes or mp.cpu_count()) as pool:
        metadata = pool.map(fn, all_audio)
    
    metadata = {
        path: m for path, m in zip(all_audio, metadata)
        if m is not None
    }

    return metadata

if __name__ == "__main__":
    main()