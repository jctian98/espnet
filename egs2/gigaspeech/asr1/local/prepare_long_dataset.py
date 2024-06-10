from argparse import ArgumentParser
from pathlib import Path
from espnet2.utils.types import str2bool

from utils import (
    Utterance,
    generate_long_utterances,
)

def parse_args():
    parser = ArgumentParser(description="Splice data into long audio format")
    parser.add_argument("--input_dir", type=Path, help="Path to input data directory")
    parser.add_argument("--output_dir", type=Path, help="Path to output data directory")
    parser.add_argument(
        "--force_long_and_continuous", 
        type=str2bool, 
        default=True,
        help="If true, filter out those long utterances that have long gap"
    )
    parser.add_argument(
        "--gap_threshold",
        type=float,
        default=1.0,
        help="utterance that contains gap longer than this is discarded",
    )
    parser.add_argument(
        "--min_threshold",
        type=float,
        default=10.0,
        help="utterance shorter than this threshold is discarded",
    )
    parser.add_argument(
        "--max_threshold",
        type=float,
        default=30.0,
        help="utterance longer than this threshold is discarded",
    )

    args = parser.parse_args()
    return args

def parse_file(file_path, file_name):
    retval = {}
    for line in open(file_path):
        eg_name, content = line.strip().split(maxsplit=1)
        
        if file_name == "segments":
            wavid, start, end = content.strip().split()
            start, end = float(start), float(end)
            content = (wavid, start, end)

        retval[eg_name] = content
    
    return retval


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(exist_ok=True, parents=True)
    
    data_dict = {}
    for file_name in ["text", "wav.scp", "utt2spk", "spk2utt", "segments"]:
        data_dict[file_name] = parse_file(input_dir / file_name, file_name)
    
    wav_scp_writer = open(output_dir / "wav.scp", 'w')
    utt2spk_writer = open(output_dir / "utt2spk", 'w')
    text_writer = open(output_dir / "text", 'w')
    segments_writer = open(output_dir / "segments", 'w')

    for wav_id, wav_path in data_dict["wav.scp"].items():
        utterances = []
        for utt_id, (_wav_id, start, end) in data_dict["segments"].items():
            if _wav_id == wav_id:
                utterances.append(
                    Utterance(
                        utt_id=utt_id,
                        wav_id=wav_id,
                        wav_path=wav_path,
                        start_time=start,
                        end_time=end,
                        text=data_dict["text"][utt_id],
                        speaker=data_dict["utt2spk"][utt_id],
                    )
                )
        
        long_utterances = generate_long_utterances(
            utterances,
            force_long_and_continuous=args.force_long_and_continuous,
            max_thre=args.max_threshold,
            min_thre=args.min_threshold,
            gap_thre=args.gap_threshold,
        )

        for u in long_utterances:
            wav_scp_writer.write(f"{u.wav_id} {u.wav_path}\n")
            utt2spk_writer.write(f"{u.utt_id} {u.speaker}\n")
            text_writer.write(f"{u.utt_id} {u.text}\n")
            segments_writer.write(f"{u.utt_id} {u.wav_id} {u.start_time} {u.end_time}\n")
        
if __name__ == "__main__":
    args = parse_args()
    main(args)


