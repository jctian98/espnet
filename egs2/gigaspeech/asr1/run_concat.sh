#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_v5"
valid_set="dev_v5"
test_sets="test_v5"

asr_config=conf/tuning/train_asr_e_branchformer_owsm_base.yaml
asr_config=conf/tuning/train_asr_e_branchformer_owsm_base_k2stc.yaml
lm_config=conf/train_lm.yaml
inference_config=conf/decode_asr.yaml

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors=""

./asr.sh \
    --audio_format flac.ark \
    --lang en \
    --ngpu 2 \
    --nj 64 \
    --gpu_inference true \
    --inference_nj 5 \
    --use_lm false \
    --nbpe 5000 \
    --max_wav_duration 30 \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --asr_config "${asr_config}" \
    --lm_config "${lm_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --bpe_train_text "data/${train_set}/text" \
    --asr_stats_dir exp/concat_stats \
    --bpe_nlsyms data/${train_set}/nlsyms.txt \
    --local_score_opts "--inference_config ${inference_config} --use_lm false" "$@"
