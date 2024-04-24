from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from functools import partial

# dependency at: https://github.com/noumar/iso639
from iso639 import languages as iso_languages

############################################
# Definitions of shared symbols and values #
############################################
SYMBOL_NA: str = "<na>"  # symbol denoting text is not available
SYMBOL_NOSPEECH: str = "<nospeech>"  # symbol denoting non-speech audio
SYMBOL_UNTRANSCRIBED: str = "<untranscribed>"
SYMBOL_UNTRANSCRIBED_LONG: str = "<untranscribed_long>"
SPEECH_MAX_LEN: float = 30  # max speech length in seconds
SPEECH_RESOLUTION: float = 0.02  # resolution in seconds
# all timestamp symbols
SYMBOLS_TIME: List[str] = [
    "<notimestamps>",
    *[
        f"<{i * SPEECH_RESOLUTION:.2f}>"
        for i in range(round(SPEECH_MAX_LEN / SPEECH_RESOLUTION) + 1)
    ],
]

# Copied from OpenAI Whisper's tokenizer.py
LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
}

# language code lookup by name, with a few language aliases
TO_LANGUAGE_CODE = {
    **{language: code for code, language in LANGUAGES.items()},
    "burmese": "my",
    "valencian": "ca",
    "flemish": "nl",
    "haitian": "ht",
    "letzeburgesch": "lb",
    "pushto": "ps",
    "panjabi": "pa",
    "moldavian": "ro",
    "moldovan": "ro",
    "sinhalese": "si",
    "castilian": "es",
}

TO_ISO_LANGUAGE_CODE = {}
for k, v in LANGUAGES.items():
    if k == "jw":
        TO_ISO_LANGUAGE_CODE[k] = "jav"
    elif k == "my":
        TO_ISO_LANGUAGE_CODE[k] = "ntx"  # (jinchuan) not for sure
    elif len(k) == 2:
        lang = iso_languages.get(alpha2=k)
    elif len(k) == 3:
        lang = iso_languages.get(part3=k)
    TO_ISO_LANGUAGE_CODE[k] = lang.part3


@dataclass
class Utterance:
    utt_id: str
    wav_id: str
    wav_path: str
    start_time: float  # in seconds
    end_time: float  # in seconds
    lang: str  # language token of speech
    task: str  # task token
    text: str  # target text without timestamps
    asr_text: str  # source text for CTC ASR without timestamps


@dataclass
class LongUtterance(Utterance):
    prev_text: str  # previous (target) text as condition
    text_with_time: str  # target text with timestamps


def time2token(x: float) -> str:
    """ "Convert float time to timestamp token."""
    x = round(x / SPEECH_RESOLUTION) * SPEECH_RESOLUTION
    return f"<{x:.2f}>"


def merge_short_utterances(
    utts: List[Utterance], prev: Optional[LongUtterance] = None,
    noise_tag_thre: float = 3.0,
    use_time_stamp: bool = True,
) -> LongUtterance:
    """Merge a list of utterances to create a long utterance."""

    def fn(u, thre):
        if u.end_time - u.start_time < thre and u.text in [SYMBOL_UNTRANSCRIBED, SYMBOL_UNTRANSCRIBED_LONG]:
            return False
        else:
            return True
    fn2 = partial(fn, thre=noise_tag_thre)

    wav_id = utts[0].wav_id
    wav_path = utts[0].wav_path
    start_time = utts[0].start_time
    end_time = utts[-1].end_time
    lang = utts[0].lang
    task = utts[0].task
    utt_id = (
        f"{wav_id}_{round(1000 * start_time):09d}_"
        f"{round(1000 * end_time):09d}_{lang[1:-1]}_{task[1:-1]}"
    )
    text = " ".join([u.text for u in utts if fn2(u)])
    asr_text = " ".join([u.asr_text for u in utts if fn2(u)])
    prev_text = prev.text if prev is not None else SYMBOL_NA

    text_with_time = ""
    for u in utts:
        if not fn2(u):
            continue
        if use_time_stamp:
            text_with_time += (
                f"{time2token(u.start_time - start_time)} "
                f"{u.text.strip()}{time2token(u.end_time - start_time)}"
            )
        else:
            text_with_time += f"{u.text.strip()}"

    return LongUtterance(
        utt_id=utt_id,
        wav_id=wav_id,
        wav_path=wav_path,
        start_time=start_time,
        end_time=end_time,
        lang=lang,
        task=task,
        text=text,
        asr_text=asr_text,
        prev_text=prev_text,
        text_with_time=text_with_time,
    )


def generate_long_utterances(
    utts: List[Utterance],  # list of short utterances in the same long talk/speech
    apply_overlap: bool = False,
    noise_tag_thre: float = 3.0,
    use_time_stamp: bool = True,
) -> List[LongUtterance]:
    """Generate a list of long utterances from a list of short utterances."""

    utts.sort(key=lambda x: x.start_time)

    long_utts = [None]
    left, right = 0, 0
    while left < len(utts):
        if right < len(utts) and (
            utts[right].end_time - utts[left].start_time <= SPEECH_MAX_LEN
        ):
            right += 1
        elif right > left:
            long_utts.append(
                merge_short_utterances(utts[left:right], long_utts[-1], noise_tag_thre, use_time_stamp=use_time_stamp)
            )
            left = right if not apply_overlap else left + 1
        else:  # skip the current utt if its length already exceeds the limit
            long_utts.append(None)
            left = right + 1 if not apply_overlap else left + 1
            right = left

    long_utts = [u for u in long_utts if u is not None]
    return long_utts
