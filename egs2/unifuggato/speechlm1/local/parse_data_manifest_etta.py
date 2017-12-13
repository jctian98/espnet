import argparse
import json
import os
import librosa
import multiprocessing as mp
from pathlib import Path


raw_files = [
    "/lustre/fsw/portfolios/adlr/projects/adlr_audio_music/syn-cap-for-TTA-checkpoints/audioset_augmented_caption/AudioFlamingo2-3B-filtered-nvclap-organized/AudioSet_long_nvclap_geq_0.2_top1_AESfiltered.ndjson",
    "/lustre/fsw/portfolios/adlr/projects/adlr_audio_music/syn-cap-for-TTA-checkpoints/audioset_augmented_caption/AudioFlamingo2-3B-filtered-nvclap-organized/AudioSet_short_nvclap_geq_0.2_top1_AESfiltered.ndjson",
    "/lustre/fsw/portfolios/adlr/projects/adlr_audio_music/syn-cap-for-TTA-checkpoints/freesound-clap_augmented_caption/AudioFlamingo2-3B-filtered-nvclap-organized/FreeSound-clap_long_nvclap_geq_0.2_top1_AESfiltered.ndjson",
    "/lustre/fsw/portfolios/adlr/projects/adlr_audio_music/syn-cap-for-TTA-checkpoints/freesound-clap_augmented_caption/AudioFlamingo2-3B-filtered-nvclap-organized/FreeSound-clap_short_nvclap_geq_0.2_top1_AESfiltered.ndjson",
    "/lustre/fsw/portfolios/adlr/projects/adlr_audio_music/syn-cap-for-TTA-checkpoints/freesound-stereo-twins_augmented_caption/AudioFlamingo2-3B-filtered-nvclap-organized/FreeSound-stereo-twins_long_nvclap_geq_0.2_top1_AESfiltered.ndjson",
    "/lustre/fsw/portfolios/adlr/projects/adlr_audio_music/syn-cap-for-TTA-checkpoints/freesound-stereo-twins_augmented_caption/AudioFlamingo2-3B-filtered-nvclap-organized/FreeSound-stereo-twins_short_nvclap_geq_0.2_top1_AESfiltered.ndjson",
    "/lustre/fsw/portfolios/adlr/projects/adlr_audio_music/syn-cap-for-TTA-checkpoints/sounddescs_augmented_caption/AudioFlamingo2-3B-filtered-nvclap-organized/SoundDescs_long_nvclap_geq_0.2_top1_AESfiltered.ndjson",
    "/lustre/fsw/portfolios/adlr/projects/adlr_audio_music/syn-cap-for-TTA-checkpoints/sounddescs_augmented_caption/AudioFlamingo2-3B-filtered-nvclap-organized/SoundDescs_short_nvclap_geq_0.2_top1_AESfiltered.ndjson",
    "/lustre/fsw/portfolios/adlr/projects/adlr_audio_music/syn-cap-for-TTA-checkpoints/vggsound_augmented_caption/AudioFlamingo2-3B-filtered-nvclap-organized/VGGSound_long_nvclap_geq_0.2_top1_AESfiltered.ndjson",
    "/lustre/fsw/portfolios/adlr/projects/adlr_audio_music/syn-cap-for-TTA-checkpoints/vggsound_augmented_caption/AudioFlamingo2-3B-filtered-nvclap-organized/VGGSound_short_nvclap_geq_0.2_top1_AESfiltered.ndjson",
]

def get_parser():
    parser = argparse.ArgumentParser(description='Processing the manifest from ETTA')
    
    # Add arguments
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

def main():
    parser = get_parser()
    args = parser.parse_args()

    (args.output_dir / args.prefix).mkdir(parents=True, exist_ok=True)
    wav_scp_writer = open(args.output_dir / args.prefix / 'wav.scp', 'w')
    segments_writer = open(args.output_dir / args.prefix / 'segments', 'w')
    reco2dur_writer = open(args.output_dir / args.prefix / 'reco2dur', 'w')

    # Find all audio paths
    all_audio = set()
    for json_file in raw_files:
        for line in open(json_file):
            line_dict = json.loads(line)
            audio_path = line_dict['location']
            all_audio.add(audio_path)
    print(f'Find {len(all_audio)} audio in total')
    
    # Find all audio duration
    audio_lengths = get_audio_lengths(all_audio)
    fail_count = 0
    for audio_path in all_audio:
        if audio_path not in audio_lengths:
            print(f"Path {audio_path} is invalid")
            fail_count
    print(f"{fail_count}/{len(all_audio)} audio is invalid")

    # write the file
    clip_id_cache = set()
    for json_file in raw_files:
        for line in open(json_file):
            line_dict = json.loads(line)
            audio_path = line_dict['location']

            if not audio_path in audio_lengths:
                continue
            
            if 'start' in line_dict and 'end' in line_dict:
                start = float(line_dict['start'])
                end = float(line_dict['end'])
                end = min(end, audio_lengths[audio_path])
                assert start < end
            else:
                start, end = 0, audio_lengths[audio_path]
            
            clip_id = f"{audio_path}_{start}_{end}"
            if clip_id not in clip_id_cache:
                segments_writer.write(f"{clip_id} {audio_path} {start} {end}\n")
                clip_id_cache.add(clip_id)

            if audio_path in all_audio:
                wav_scp_writer.write(f"{audio_path} {audio_path}\n")
                reco2dur_writer.write(f"{audio_path} {audio_lengths[audio_path]}\n")
                all_audio.remove(audio_path)

    print('done')

if __name__ == "__main__":
    main()