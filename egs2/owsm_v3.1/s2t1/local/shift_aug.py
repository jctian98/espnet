import argparse
import shutil
import copy
from kaldiio import ReadHelper

from pathlib import Path
from espnet2.utils.types import str2bool
from utils import (
    SYMBOL_NA,
    SYMBOL_NOSPEECH,
    SYMBOLS_TIME,
    SYMBOL_UNTRANSCRIBED,
    SYMBOL_UNTRANSCRIBED_LONG,
    LongUtterance,
    Utterance,
    generate_long_utterances,
)


def main():
    args = get_parser()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # (1) Parse the segment and text files
    text_dict = {}
    for line in open(args.data_dir / "text"):
        line = line.strip().split()
        uttid, content = line[0], " ".join(line[1:])
        text_dict[uttid] = content

    wav_dict = {}
    for line in open(args.data_dir / "wav.scp"):
        line = line.strip().split()
        uttid, content = line[0], " ".join(line[1:])
        wav_dict[uttid] = content

    # This takes a bit long time. So keep a file record
    wavlen_dict = {}
    if (args.data_dir / "reco2dur").is_file():
        for line in open(args.data_dir / "reco2dur"):
            wavid, dur = line.strip().split()
            wavlen_dict[wavid] = float(dur)
    else:
        for wavid, (rate, wav) in ReadHelper(f"scp:{args.data_dir}/wav.scp"):
            wavlen_dict[wavid] = len(wav) / rate
        reco2dur_writer = open(args.data_dir / "reco2dur", "w")
        for wavid, dur in wavlen_dict.items():
            reco2dur_writer.write(f"{wavid} {dur}\n")
    
    # (2) Get all transcribed clips
    clip_dict = {}
    for line in open(args.data_dir / "segments"):
        uttid, wavid, start, end = line.strip().split()
        start, end = float(start), float(end)
        if wavid not in clip_dict:
            clip_dict[wavid] = []
        clip_dict[wavid].append(
            Utterance(
                utt_id=uttid,
                wav_id=wavid,
                wav_path=wav_dict[wavid],
                start_time=start,
                end_time=end,
                lang="<eng>",
                task="<asr>",
                text=text_dict[uttid],
                asr_text=text_dict[uttid],
            )
        )
    for v in clip_dict.values():
        v.sort(key=lambda x: x.start_time)

    # (3) Merge the consecutive transcribed clips:
    for k, v in clip_dict.items():
        clip_dict[k] = merge_consecutive_clips(v, args.merge_thre, args.max_clip_len)

    # (4) Get all untranscribed clips
    for wavid, clips in clip_dict.items():
        clips = copy.deepcopy(clips)
        global_start, global_end = 0.0, wavlen_dict[wavid]
        for idx, clip in enumerate(clips):
            seg_dur = clip.start_time - global_start
            text = (
                SYMBOL_UNTRANSCRIBED_LONG if seg_dur > args.long_noise_tag_thre else SYMBOL_UNTRANSCRIBED
            )
            if seg_dur >= args.include_thre and seg_dur <= args.max_segment_len:
                clip_dict[wavid].append(
                    Utterance(
                        utt_id=f"{wavid}_noise{idx}",
                        wav_id=wavid,
                        wav_path=wav_dict[wavid],
                        start_time=global_start,
                        end_time=clip.start_time,
                        lang="<eng>",
                        task="<asr>",
                        text=text,
                        asr_text=text,
                    )
                )
            global_start = clip.end_time

        # The last clip
        seg_dur = global_end - end
        text = SYMBOL_UNTRANSCRIBED_LONG if seg_dur > args.long_noise_tag_thre else SYMBOL_UNTRANSCRIBED
        if seg_dur >= args.include_thre and seg_dur <= args.max_segment_len:
            clip_dict[wavid].append(
                Utterance(
                    utt_id=f"{wavid}_noise{idx+1}",
                    wav_id=wavid,
                    wav_path=wav_dict[wavid],
                    start_time=global_start,
                    end_time=global_end,
                    lang="<eng>",
                    task="<asr>",
                    text=text,
                    asr_text=text,
                )
            )
        clip_dict[wavid].sort(key=lambda x: x.start_time)

    # (5) Dump to a the output_dir. Note some policy is integrated in this process
    # like apply_overlap & noise_tag threshold
    wavscp_fp = open(args.output_dir / "wav.scp", "w")
    segments_fp = open(args.output_dir / "segments", "w")
    text_fp = open(args.output_dir / "text", "w")
    textprev_fp = open(args.output_dir / "text.prev", "w")
    textctc_fp = open(args.output_dir / "text.ctc", "w")
    utt2spk_fp = open(args.output_dir / "utt2spk", "w")

    for talk in clip_dict.values():
        for u in generate_long_utterances(talk, args.apply_overlap, args.noise_tag_thre, use_time_stamp=args.use_time_stamp):
            if u.start_time >= u.end_time:
                print('get a example with bad time stamp: ', u)
                continue

            wavscp_fp.write(f"{u.wav_id} {u.wav_path}\n")
            segments_fp.write(
                f"{u.utt_id} {u.wav_id} {u.start_time:.2f} {u.end_time:.2f}\n"
            )
            text_fp.write(f"{u.utt_id} {u.lang}{u.task}{u.text_with_time}\n")
            textprev_fp.write(f"{u.utt_id} {u.prev_text}\n")
            textctc_fp.write(f"{u.utt_id} {u.asr_text}\n")
            utt2spk_fp.write(f"{u.utt_id} {u.utt_id}\n")
    
    # (Jinchuan): TODO: add other symbols like different lang / st_lang
    special_tokens = [
        SYMBOL_NA,
        SYMBOL_NOSPEECH,
        SYMBOL_UNTRANSCRIBED,
        SYMBOL_UNTRANSCRIBED_LONG,
        "<asr>",
        "<eng>",
        *SYMBOLS_TIME,
    ]
    with open(args.output_dir / "nlsyms.txt", "w") as fp:
        for tok in special_tokens:
            fp.write(f"{tok}\n")
        
def merge_consecutive_clips(clips, merge_thre, max_clip_len):
    left, right = 0, 0
    new_clips = []
    while left < len(clips):
        if (
            right < len(clips)
            and clips[right].end_time - clips[left].start_time <= max_clip_len
            and (
                right == left
                or clips[right].start_time - clips[right - 1].end_time < merge_thre
            )
        ):
            right += 1

        elif right > left:
            if right - left > 2:
                print('merged: ', clips[left].wav_id, clips[left].start_time)
            new_clips.append(
                Utterance(
                    utt_id=clips[left].utt_id + "_merged",
                    wav_id=clips[left].wav_id,
                    wav_path=clips[left].wav_path,
                    start_time=clips[left].start_time,
                    end_time=clips[right - 1].end_time,
                    lang=clips[left].lang,
                    task=clips[left].task,
                    text=" ".join([c.text.strip() for c in clips[left:right]]),
                    asr_text=" ".join([c.text.strip() for c in clips[left:right]]),
                )
            )
            left = right
        else:
            new_clips.append(clips[left])
            left = right + 1
            right = left

    return new_clips


def get_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--data_dir", type=Path, help="Path to raw data.")
    parser.add_argument("--output_dir", type=Path, help="Path to raw data.")
    parser.add_argument(
        "--include_thre",
        type=float,
        default=0.0,
        help="Untranscribed Clips longer than this will be included in audio",
    )
    parser.add_argument(
        "--noise_tag_thre",
        type=float,
        default=3.0,
        help="Untranscribed Clips longer than this have a <untranscribed> tag in text",
    )
    parser.add_argument(
        "--long_noise_tag_thre",
        type=float,
        default=15.0,
        help="Untranscribed Clips longer than this have a <untranscribed_long> tag in text",
    )
    parser.add_argument(
        "--merge_thre",
        type=float,
        default=1.0,
        help="will merge two transcribed clips when the transcribed clips in the middle is shorter than this",
    )
    parser.add_argument(
        "--max_clip_len",
        type=float,
        default=15.0,
        help="maximum clip lengths for clip merging",
    )
    parser.add_argument(
        "--max_segment_len",
        type=float,
        default=30.0,
        help="maximum segment length. Clip longer than this will be discarded",
    )
    parser.add_argument(
        "--apply_overlap",
        type=str2bool,
        default=False,
        help="If true, apply overlap sliding windows",
    )
    parser.add_argument(
        "--use_time_stamp",
        type=str2bool,
        default=True,
        help="If true, use time-stamp for each clip",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main()
