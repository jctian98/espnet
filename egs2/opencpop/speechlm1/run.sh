#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# spectrogram-related arguments
fs=24000 # 24000 or 441000
fmin=
fmax=
n_fft=
n_shift=
win_length=

if [ $fs -eq 24000 ]; then
    fmin=0
    fmax=22050
    n_fft=2048
    n_shift=300
    win_length=1200
elif [ $fs -eq 44100 ]; then
    fmin=0
    fmax=22050
    n_fft=2048
    n_shift=512
    win_length=2048
fi


train_set=tr_no_dev
valid_set=dev
test_sets="dev test"

# training and inference configuration
train_config=conf/train.yaml

codec_opts="--codec_choice ESPnet --codec_hf_model_tag espnet/owsmdata_soundstream_16k_200epoch"

# NOTE(Jinchuan): This script is only to prepare data. End at stage 5
./speechlm.sh \
    --stop_stage 5 \
    --task "svs" \
    --data_name opencpop \
    --fs "${fs}" \
    --ngpu 8 \
    --nj 32 \
    --train_config "${train_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --audio_format "wav" \
    --min_wav_duration 3.0 \
    --max_wav_duration 30.0 \
    ${codec_opts} \
    "$@"


