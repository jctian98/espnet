#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from dataclasses import dataclass
from typing import List, Tuple

# Users are encouraged to read our document to understand the definitions in this file:
# https://github.com/jctian98/espnet/tree/speechlm3/egs2/TEMPLATE/README.md


# (1) Modality definition
# a. discrete means the discrete / continuous format in final sequences.
#    e.g., the LLM output embeddings (textemb) is originally read as text
#    then tokenized into BPE, and finally converted into continuous
#    embeddings before being processed by the SpeechLM. So it's continouos
#    This is to determine if a placeholder should be adopted in the spliced
#    sequence in preprocess_fn
# b. data_type: how the original data file is loaded. This is exactly follows
#    the definitions in espent2.train.dataset
# c. For discrete modality, we usually have a modality-specific vocabulary
#    an exception is "spk"
@dataclass
class Modality:
    discrete: bool = True


MODALITIES = {}
# Discrete
MODALITIES["codec"] = Modality()
MODALITIES["ssl"] = Modality()
MODALITIES["codec_ssl"] = Modality()
MODALITIES["text_bpe"] = Modality()
MODALITIES["g2p"] = Modality()
MODALITIES["spk"] = Modality()
MODALITIES["class"] = Modality()
MODALITIES["bool"] = Modality()
MODALITIES["video_ssl"] = Modality()
MODALITIES["svs_lb"] = Modality()
MODALITIES["image"] = Modality()

# continuous
MODALITIES["wav"] = Modality(discrete=False)
MODALITIES["text_emb"] = Modality(discrete=False)
MODALITIES["ssl_feat"] = Modality(discrete=False)

# dialogue
MODALITIES["dialogue"] = Modality()

# END OF MODALITY DEFINITION #


# (2) Task Definition
# a. usually we will place a task identifier in the begining.
#    however, when we want to specify the task by natual langauge,
#    we don't use that identifier.
# b. encoder_entries: the entires that should feed to encoder, which
#    is a list of tuple (file_name, entry_modality, data_type).
# c. decoder_entries: similar to encoder_entries, but is fed to decoder.
# d. in decoder-only format, encoder_entries and decoder_entries are merged
# e. target_entries: entries that the loss computed on. Usually same as
#    the decoder_entries.
# f. file_name, the expected file name in original data folder. e.g., wav.scp
#    entry_modality: the modality defined above, which will be used to determine
#      how this data will be pre-processed before training. e.g., codec tokenization
#    data_type: it determines how the data will be loaded during training.
#    E.g., in TTS, the wave files are indexed with wav.scp, it will experence codec
#      tokenization and then loaded as kaldi_ark -> (wav.scp, codec, kaldi_ark)
@dataclass
class SpeechLMTaskTemplate:
    conditions: List[Tuple[str, str, str]]
    targets: List[Tuple[str, str, str]]
    use_task_identifier: bool = True
    fixed_length_key: str = ""

    @property
    def data_triplets(self):
        all_entries = self.conditions + self.targets
        return all_entries

    @property
    def n_conditions(self):
        return len(self.conditions)

    @property
    def n_targets(self):
        return len(self.targets)

    @property
    def data_triplets_string(self):
        ans = ""
        for entry in self.data_triplets:
            ans = ans + ",".join(entry) + " "
        return ans

    @property
    def condition_string(self):
        ans = ""
        for entry in self.conditions:
            ans = ans + ",".join(entry) + " "
        return ans

    @property
    def target_string(self):
        ans = ""
        for entry in self.targets:
            ans = ans + ",".join(entry) + " "
        return ans


SPEECHLM_TASKS = dict()

SPEECHLM_TASKS["textlm"] = SpeechLMTaskTemplate(
    conditions=[],
    targets=[("text", "text_bpe", "text")],
)

SPEECHLM_TASKS["audiolm"] = SpeechLMTaskTemplate(
    conditions=[],
    targets=[("wav.scp", "codec", "kaldi_ark")],
)

SPEECHLM_TASKS["ssl_audiolm"] = SpeechLMTaskTemplate(
    conditions=[],
    targets=[("wav.scp", "ssl", "kaldi_ark")],
)

SPEECHLM_TASKS["tts"] = SpeechLMTaskTemplate(
    conditions=[("text", "g2p", "text"), ("utt2spk", "spk", "text")],
    targets=[("wav.scp", "codec", "kaldi_ark")],
)

SPEECHLM_TASKS["ssl_tts"] = SpeechLMTaskTemplate(
    conditions=[("text", "text_bpe", "text")],
    targets=[("wav.scp", "ssl", "kaldi_ark")],
)

SPEECHLM_TASKS["bpe_tts"] = SpeechLMTaskTemplate(
    conditions=[("text", "text_bpe", "text"), ("utt2spk", "spk", "text")],
    targets=[("wav.scp", "codec", "kaldi_ark")],
)

SPEECHLM_TASKS["asr"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec", "kaldi_ark")],
    targets=[("text", "text_bpe", "text")],
)

SPEECHLM_TASKS["ssl_asr"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "ssl", "kaldi_ark")],
    targets=[("text", "text_bpe", "text")],
)

SPEECHLM_TASKS["mt"] = SpeechLMTaskTemplate(
    conditions=[("src_text", "text_bpe", "text")],
    targets=[("text", "text_bpe", "text")],
)

SPEECHLM_TASKS["text2audio"] = SpeechLMTaskTemplate(
    conditions=[("text", "text_emb", "kaldi_ark")],
    targets=[("wav.scp", "codec", "kaldi_ark")],
)

SPEECHLM_TASKS["visual_tts"] = SpeechLMTaskTemplate(
    conditions=[
        ("text", "g2p", "text"),
        ("utt2spk", "spk", "text"),
        ("video.scp", "video_ssl", "kaldi_ark"),
    ],
    targets=[("wav.scp", "codec", "kaldi_ark")],
)

SPEECHLM_TASKS["vc"] = SpeechLMTaskTemplate(
    conditions=[("src_wav.scp", "codec", "kaldi_ark"), ("utt2spk", "spk", "text")],
    targets=[("wav.scp", "codec", "kaldi_ark")],
)

SPEECHLM_TASKS["ssl2codec"] = SpeechLMTaskTemplate(
    conditions=[("ssl_wav.scp", "ssl", "kaldi_ark"), ("utt2spk", "spk", "text")],
    targets=[("wav.scp", "codec", "kaldi_ark")],
)

SPEECHLM_TASKS["svs"] = SpeechLMTaskTemplate(
    conditions=[("label", "svs_lb", "text")],
    targets=[("wav.scp", "codec", "kaldi_ark")],
)

SPEECHLM_TASKS["mt"] = SpeechLMTaskTemplate(
    conditions=[("src_text", "text_bpe", "text")],
    targets=[("text", "text_bpe", "text")],
)

SPEECHLM_TASKS["st"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "ssl", "kaldi_ark")],
    targets=[("src_text", "text_bpe", "text"), ("text", "text_bpe", "text")],
)

SPEECHLM_TASKS["se"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec", "kaldi_ark")],
    targets=[("spk1.scp", "codec", "kaldi_ark")],
)

# codec_ssl tasks:
SPEECHLM_TASKS["codec_ssl_asr"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("text", "text_bpe", "text")],
)

SPEECHLM_TASKS["codec_ssl_tts"] = SpeechLMTaskTemplate(
    conditions=[("text", "text_bpe", "text"), ("utt2spk", "spk", "text")],
    targets=[("wav.scp", "codec_ssl", "kaldi_ark")],
)

SPEECHLM_TASKS["codec_ssl_plain_tts"] = SpeechLMTaskTemplate(
    conditions=[("text", "text_bpe", "text")],
    targets=[("wav.scp", "codec_ssl", "kaldi_ark")],
)

SPEECHLM_TASKS["codec_ssl_audiolm"] = SpeechLMTaskTemplate(
    conditions=[],
    targets=[("wav.scp", "codec_ssl", "kaldi_ark")],
)

SPEECHLM_TASKS["codec_ssl_se"] = SpeechLMTaskTemplate(
    conditions=[("mix.scp", "codec_ssl", "kaldi_ark")],
    targets=[("wav.scp", "codec_ssl", "kaldi_ark")],
    fixed_length_key="mix.scp",
)

SPEECHLM_TASKS["codec_ssl_tse"] = SpeechLMTaskTemplate(
    conditions=[("mix.scp", "codec_ssl", "kaldi_ark"), ("utt2spk", "spk", "text")],
    targets=[("wav.scp", "codec_ssl", "kaldi_ark")],
    fixed_length_key="mix.scp",
)

SPEECHLM_TASKS["aac_codecssl"] = SpeechLMTaskTemplate(
    conditions=[("wav.scp", "codec_ssl", "kaldi_ark")],
    targets=[("text", "text_bpe", "text")],
)

SPEECHLM_TASKS["ag_codecssl"] = SpeechLMTaskTemplate(
    conditions=[("text", "text_bpe", "text")],
    targets=[("wav.scp", "codec_ssl", "kaldi_ark")],
)

SPEECHLM_TASKS["text_dialogue"] = SpeechLMTaskTemplate(
    conditions=[],
    targets=[("dialogue", "dialogue", "dialogue_json")],
)

SPEECHLM_TASKS["audio_dialogue"] = SpeechLMTaskTemplate(
    conditions=[],
    targets=[("dialogue", "dialogue", "dialogue_json")],
)

SPEECHLM_TASKS["vision_dialogue"] = SpeechLMTaskTemplate(
    conditions=[],
    targets=[("dialogue", "dialogue", "dialogue_json")],
)

SPEECHLM_TASKS["image_to_text"] = SpeechLMTaskTemplate(
    conditions=[("image.scp", "image", "kaldi_ark")],
    targets=[("text", "text_bpe", "text")],
)

SPEECHLM_TASKS["text_to_image"] = SpeechLMTaskTemplate(
    conditions=[("text", "text_bpe", "text")],
    targets=[("image.scp", "image", "kaldi_ark")],
)

# END OF TASK DEFINITION #

# (3) Special token definition
# a. always reserve 256 slots for special tokens
#    0-31:    general special tokens
#    32-63:   modality identifier
#    64-127:  task identifier
#    128-255: reserved for future
# b. don't delete / modify it, otherwise the model trained
#    previously can become incompatible. New tokens can be
#    added - there are enough slots
# c. detailed explanation for frequently special tokens:
#    <pad>: padding tokens. These tokens is for padding only and will not participate
#           loss computing.
#    <sos/eos>: start-of-sentence/end-of-senetence. Each sequence always starts and
#           ends with this token.
#    <system_prompt>, <user_input>, <assistant_output>: role tokens in chat template.
#    <eou>: end-of-utternace, end of an utterance (or a short segment) of certain
#           modality. Usually used as a termination signal in decoding.
#    modality tokens, e.g., <text_bpe_start/end>: the token to indicate modality. This
#           token is always in the first place of a segment. e.g.,
#           <text_bpe_start/end>, text_token1, ..., text_tokenN
#    task tokens, e.g., <asr_task>: the indicator of a certain task. This token is
#           always in the second place of a whole sequence, following <sos/eos>.
#    Other special tokens are deprecated or not in frequent usage.
#    See use case in:
#    https://github.com/jctian98/espnet/tree/speechlm3/egs2/
#    TEMPLATE/speechlm1#example-sequence
special_tokens = [
    "<pad>",
    "<unk>",
    "<blank>",
    "<space>",
    "<continuous_placeholder>",
    "<sos/eos>",
    "<local_sos/eos>",
    "<unkown_task_identifer>",
    "<system_prompt>",
    "<user_input>",
    "<assistant_output>",
    "<eou>",
]


def pad_until(token_list, until):
    assert until > len(token_list)
    for idx in range(len(token_list), until):
        token_list.append(f"<unused_token_{idx}>")
    return token_list


special_tokens = pad_until(special_tokens, 32)

for m in MODALITIES.keys():
    special_tokens.append(f"<{m}_start/end>")
special_tokens = pad_until(special_tokens, 64)

for m in SPEECHLM_TASKS.keys():
    special_tokens.append(f"<{m}_task>")
special_tokens = pad_until(special_tokens, 128)

special_tokens = pad_until(special_tokens, 256)

# END OF SPECIAL TOKEN DEFINITION #
