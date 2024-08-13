#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


lang=$1 # en de fr cy tt kab ca zh-TW it fa eu es ru tr nl eo zh-CN rw pt zh-HK cs pl uk
fs=$2
codec_opts="$3"
stage=$4
stop_stage=$5

train_set=train_"$(echo "${lang}" | tr - _)"
valid_set=dev_"$(echo "${lang}" | tr - _)"
test_sets="${valid_set} test_$(echo ${lang} | tr - _)"

train_config=conf/tuning/train_asr_conformer5.yaml
lm_config=conf/train_lm.yaml
inference_config=conf/decode_asr.yaml

if [[ "zh" == *"${lang}"* ]]; then
  nbpe=2500
elif [[ "fr" == *"${lang}"* ]]; then
  nbpe=350
elif [[ "es" == *"${lang}"* ]]; then
  nbpe=235
else
  nbpe=150
fi

cleaner=tacotron
g2p=g2p_en_no_space # or g2p_en

# Note(Jinchuan): We only select audio range from 3s to 30s since:
#                 (1) The speech prompt is 3s
#                 (2) We limit the longest audio to 30s to avoid
#                     some corner cases in memeory
./speechlm.sh \
    --stage ${stage} \
    --stop_stage ${stop_stage} \
    --task "asr" \
    --data_name "commonvoice" \
    --fs "${fs}" \
    --ngpu 1 \
    --nj 8 \
    --cleaner "${cleaner}" \
    --g2p "${g2p}" \
    --inference_nj 1 \
    --gpu_inference true \
    --audio_format "flac.ark" \
    --train_config ${train_config} \
    --inference_config ${inference_config} \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --min_wav_duration 3.0 \
    --max_wav_duration 30.0 \
    --local_data_opts "--lang ${lang} --stage 1 " \
    ${codec_opts} 
