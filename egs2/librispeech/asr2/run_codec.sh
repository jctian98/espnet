#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


src_lang=codec
tgt_lang=en

train_set="train_960"
train_dev="dev"
test_sets="test_clean test_other dev_clean dev_other"

asr_config=conf/tuning/train_asr_ebranchformer_codec.yaml
inference_config=conf/decode_ctc0.3.yaml

tgt_nbpe=5000   # if token_joint is True, then only tgt_nbpe is used
codec_opts="--codec_choice ESPnet --codec_hf_model_tag espnet/amuse_speech_soundstream_16k"

# ts: true sequence. codec should always in "ts" case
# rm: deduplicated sequence which removes duplicated tokens
src_case="ts"
tgt_case="ts"


./asr2.sh \
    --tokenization_choice "codec" \
    --nj 32 \
    --ngpu 8 \
    --audio_format flac.ark \
    --fs 16000 \
    --use_lm false \
    --src_lang ${src_lang} \
    --tgt_lang ${tgt_lang} \
    --src_token_type "null" \
    --tgt_token_type "bpe" \
    --tgt_nbpe $tgt_nbpe \
    --tgt_case ${tgt_case} \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${train_dev}" \
    --test_sets "${test_sets}" \
    --tgt_bpe_train_text "data/${train_set}/text" \
    ${codec_opts} "$@"
