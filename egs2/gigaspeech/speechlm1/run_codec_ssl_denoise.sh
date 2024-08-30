#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=gigaspeech_train_xl_enh_sep
valid_set=gigaspeech_dev_enh_sep
test_sets="gigaspeech_test_enh_sep"

ssl_opts="--ssl_checkpoint_path exp/kmeans_xues/38epoch.pth --ssl_kmeans_path exp/kmeans_xues/km_5000.mdl --ssl_nlayer 16"
codec_opts="--codec_choice ESPnet --codec_hf_model_tag espnet/owsmdata_soundstream_16k_200epoch"
bpe_opts="--subword_choice huggingface --subword_model google/gemma-2b-it"

# NOTE(Jinchuan): stop at stage 5, for data preparation only
./speechlm.sh \
    --stop_stage 5 \
    --task "codec_ssl_denoise" \
    --data_name gigaspeech \
    --fs 16000 \
    --ngpu 1 \
    --nj 64 \
    --train_config conf/train_foo.yaml \
    --audio_format "flac.ark" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --min_wav_duration 1.0 \
    --max_wav_duration 30.0 \
    ${ssl_opts} ${codec_opts} ${bpe_opts} \
    "$@"