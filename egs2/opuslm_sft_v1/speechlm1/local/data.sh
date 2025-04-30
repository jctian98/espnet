#!/usr/bin/env bash

# Copyright 2025 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# This script prepare the SFT data for OpusLM-V1.

#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=4
stop_stage=4

TULU3=data/local/tulu3
OpenAudioBench=data/local/openaudiobench
OLMO2_SFT=/work/hdd/bbjs/jtian1/tools/olmo2_dpo
OLMO2_DPO=data/local/olmo2_dpo

. utils/parse_options.sh

. ./db.sh
. ./path.sh

if ! command -v huggingface-cli &> /dev/null; then
    echo "Error: huggingface-cli command not found. Please install it first." >&2
    exit 1
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Convert TULU3 dataset to ESPnet-SpeechLM data format"
    python3 local/data_prep_tulu3.py \
      --download_dir ${TULU3}/data \
      --output_dir dump/raw_text_dialogue_tulu3

    for dset in train valid; do
        dir=dump/raw_text_dialogue_tulu3/${dset}
        cp ${dir}/data/dialogue.1 ${dir}/dialogue
        python3 pyscripts/utils/make_speechlm_json.py \
          --task text_dialogue \
          --output_json ${dir}/data.json \
          --file_modality_type ${dir}/dialogue,dialogue,dialogue_json
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Convert OpenAudioBench dataset to ESPnet-SpeechLM data format"
    # huggingface-cli download --repo-type dataset --local-dir ${OpenAudioBench} baichuan-inc/OpenAudioBench
    python3 local/data_prep_openaudiobench.py \
      --download_dir ${OpenAudioBench}/eval_datas \
      --output_dir dump/raw_text_dialogue_openaudiobench
    
    for dset in alpaca_eval llama_questions trivia_qa web_questions; do
        dir=dump/raw_text_dialogue_openaudiobench/${dset}
        cp ${dir}/data/dialogue.1 ${dir}/dialogue
        python3 pyscripts/utils/make_speechlm_json.py \
          --task text_dialogue \
          --output_json ${dir}/data.json \
          --file_modality_type ${dir}/dialogue,dialogue,dialogue_json
    done
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Convert OLMo2 DPO dataset to ESPnet-SpeechLM data format"
    python local/data_prep_olmo2_7b_dpo.py \
      --download_dir ${OLMO2_DPO} \
      --output_dir dump/raw_text_dialogue_olmo2_dpo 
    for dset in train valid; do
        dir=dump/raw_text_dialogue_olmo2_dpo/${dset}
        cp ${dir}/data/dialogue.1 ${dir}/dialogue
        python3 pyscripts/utils/make_speechlm_json.py \
          --task text_dialogue \
          --output_json ${dir}/data.json \
          --file_modality_type ${dir}/dialogue,dialogue,dialogue_json
    done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Convert OLMO2_SFT dataset to ESPnet-SpeechLM data format"
    python3 local/data_prep_olmo2_7b_sft.py \
      --download_dir ${OLMO2_SFT}/data \
      --output_dir dump/raw_text_dialogue_olmo2_sft

    for dset in train valid; do
        dir=dump/raw_text_dialogue_olmo2_sft/${dset}
        cp ${dir}/data/dialogue.1 ${dir}/dialogue
        python3 pyscripts/utils/make_speechlm_json.py \
          --task text_dialogue \
          --output_json ${dir}/data.json \
          --file_modality_type ${dir}/dialogue,dialogue,dialogue_json
    done
fi