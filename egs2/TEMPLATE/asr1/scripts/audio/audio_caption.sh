#!/usr/bin/env bash

# Copyright 2025 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# Prepare ESPnet Text corpus for SpeechLM training.

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
stage=1
stop_stage=1

data_dir=
output_dir=
model=Qwen/Qwen2.5-Omni-7B
valid_suffix=".flac .wav .mp3"
nproc=2
nj=512

log "$0 $*"
. utils/parse_options.sh

. ./path.sh
. ./cmd.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Doing audio caption labeling with ${model}."
    log "Input folder: ${data_dir}"
    log "Output folder: ${output_dir}"

    mkdir -p ${output_dir}

    logdir=${output_dir}/logs
    mkdir -p ${logdir}

    # find all audio files and split by nj. Path is the name
    for suffix in ${valid_suffix}; do
        find ${data_dir} -name "*${suffix}"
    done | sort | awk '{print $1, $1}' > ${output_dir}/wav.scp

    split_scps=""
    for n in $(seq ${nj}); do
        split_scps="${split_scps} ${logdir}/wav.${n}.scp"
    done
    utils/split_scp.pl ${output_dir}/wav.scp ${split_scps}

    ${cuda_cmd} JOB=1:${nj} ${logdir}/audio_caption.JOB.log \
      python3 pyscripts/audio/audio_caption.py \
        --input_file ${logdir}/wav.JOB.scp \
        --output_file ${logdir}/result.JOB.jsonl \
        --nproc ${nproc} \
        --hf_tag ${model}
    
    for n in `seq 1 ${nj}`; do
        cat ${logdir}/result.${n}.jsonl
    done > ${output_dir}/result.jsonl
fi