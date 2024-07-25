#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

fs=16000

opts=
if [ "${fs}" -eq 24000 ]; then
    # To suppress recreation, specify wav format
    opts="--audio_format wav "
else
    opts="--audio_format flac "
fi


train_set=dev-clean
valid_set=dev-clean
# test_sets="dev-clean test-clean"
test_sets="test-clean"

train_config=conf/train_encodec_16k.yaml
inference_config=conf/decode.yaml
score_config=conf/score_16k.yaml

./codec.sh \
    --local_data_opts "--trim_all_silence false" \
    --fs ${fs} \
    --inference_nj 4 --gpu_inference true \
    --train_config "${train_config}" \
    --dumpdir dump_16k_debug \
    --codec_stats_dir exp/debug_stats \
    --inference_config "${inference_config}" \
    --scoring_config ${score_config} \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" ${opts} "$@"