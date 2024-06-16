from dataclasses import dataclass
from typing import List

@dataclass
class Utterance:
    utt_id: str
    wav_id: str
    wav_path: str
    start_time: float  # in seconds
    end_time: float  # in seconds
    text: str  # target text without timestamps
    speaker: str

def merge_short_utterances(
    utts: List[Utterance],
    force_long_and_continuous: bool = False,
    max_thre: float = 30.0,
    min_thre: float = 10.0,
    gap_thre: float = 1.0,
) -> Utterance:
    """Merge a list of utterances to create a long utterance."""

    wav_id = utts[0].wav_id
    wav_path = utts[0].wav_path
    speaker = utts[0].speaker
    start_time = utts[0].start_time
    end_time = utts[-1].end_time
    utt_id = (
        f"{wav_id}_{round(1000 * start_time):09d}_"
        f"{round(1000 * end_time):09d}"
    )
    text = "<sep> ".join([u.text for u in utts])

    if force_long_and_continuous:
        if end_time - start_time > max_thre or end_time - start_time < min_thre:
            return None
        
        for i in range(len(utts) - 1):
            if utts[i+1].start_time - utts[i].end_time > gap_thre:
                return None

    return Utterance(
        utt_id=utt_id,
        wav_id=wav_id,
        wav_path=wav_path,
        start_time=start_time,
        end_time=end_time,
        text=text,
        speaker=speaker,
    )


def generate_long_utterances(
    utts: List[Utterance],
    force_long_and_continuous: bool = False,
    max_thre: float = 30.0,
    min_thre: float = 10.0,
    gap_thre: float = 1.0,
) -> List[Utterance]:
    """Generate a list of long utterances from a list of short utterances."""

    utts.sort(key=lambda x: x.start_time)

    long_utts = [None]
    left, right = 0, 0
    while left < len(utts):
        if right < len(utts) and (
            utts[right].end_time - utts[left].start_time <= max_thre
        ):
            right += 1
        elif right > left:
            long_utts.append(merge_short_utterances(
                utts[left:right],
                force_long_and_continuous=force_long_and_continuous,
                max_thre=max_thre,
                min_thre=min_thre,
                gap_thre=gap_thre,
            ))
            left = right
        else:  # skip the current utt if its length already exceeds the limit
            long_utts.append(None)
            left = right + 1
            right = left

    long_utts = [u for u in long_utts if u is not None]
    return long_utts
