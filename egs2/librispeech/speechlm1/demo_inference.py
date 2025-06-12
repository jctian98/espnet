#!/usr/bin/env python3

# Copyright 2025 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Before running this script, make sure you install ESPnet as a python package:
# pip install espnet

from espnet2.bin.speechlm_inference_chat import SpeechLM, SpeechLMTask
from espnet2.speechlm.inference_utils import TaskOrientedWriter
from pathlib import Path

# from espnet2.tasks.ssl import SSLTask
# xeus_model, xeus_train_args = SSLTask.build_model_from_file(
#     None,
#     '/work/nvme/bbjs/jtian1/tools/hf_home/hub/models--espnet--OpusLM_1.7B_Anneal/snapshots/7b448184574c954d99f0b8cf7329b11072af7e57/kmeans/model.pth',
#     'cpu',
# )


import kaldiio
mat = kaldiio.load_mat('dump/raw_codec_ssl_tts_librispeech/test_clean/data/wav_codec_ssl_ESPnet.1.ark:1444843')
mat = mat.reshape(-1, 9)
for x in mat:
    print(x, flush=True)

model = SpeechLM.from_pretrained("./cache", device='cuda')
processor = SpeechLMTask.build_preprocess_fn(
    model.train_args, train=False,
    online_tokenization=True,
    online_tokenizers=model.online_tokenizers
)

input = {
    "raw_input": True,
    "task": "codec_ssl_tts",
    "text": "Yuny is a stupid pig",
    "utt2spk": '/work/hdd/bbjs/shared/corpora/librispeech/LibriSpeech/test-clean/1089/134686/1089-134686-0001.flac'
}

writer = TaskOrientedWriter(
    train_args=model.train_args,
    task="codec_ssl_tts",
    output_dir=Path("tmp_out"),
    rank=0,
    inference_config=model.inference_config,
)

segments = model(input, processor)
writer.write('foo', segments)


