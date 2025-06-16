#!/usr/bin/env python3

# Copyright 2025 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""
A demo code for ESPnet OpusLM inference. Before you run:

1. install Pytorch with CUDA support. Recommend to use Pytorch > 2.4.0
  pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
2. install ESPnet by pip
  pip install espnet
"""

from espnet2.bin.speechlm_inference_chat import SpeechLM, SpeechLMTask
from espnet2.speechlm.inference_utils import TaskOrientedWriter
from pathlib import Path

# Loading model
model = SpeechLM.from_pretrained("espnet/OpusLM_1.7B_Anneal", device='cuda')
processor = SpeechLMTask.build_preprocess_fn(
    model.train_args, train=False,
    online_tokenization=True,
    online_tokenizers=model.online_tokenizers
)

# check input format
model.helper("codec_ssl_tts")

# TTS input
input = {
    "raw_input": True,
    "task": "codec_ssl_tts",
    "text": "This is a text-to-speech system",
    "utt2spk": '/work/hdd/bbjs/shared/corpora/librispeech/LibriSpeech/test-clean/1089/134686/1089-134686-0001.flac'
}

output_dir="tmp_out"
writer = TaskOrientedWriter(
    train_args=model.train_args,
    task="codec_ssl_tts",
    output_dir=Path(output_dir),
    rank=0,
    inference_config=model.inference_config,
)

example_name="foo"
segments = model(input, processor)
writer.write(example_name, segments)


