#!/usr/bin/env bash

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
min() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -le "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}
SECONDS=0

# . ./path.sh
. ./cmd.sh

stage=1
stop_stage=1
nj=8
inference_nj=8
gpu_inference=true
nbest=1

gen_dir=
ref_dir=
key_file=

python=python3

log "$0 $*"
. utils/parse_options.sh

mkdir -p ${gen_dir}/scoring

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ${python} pyscripts/utils/eval_image_to_text.py \
      --ref ${ref_dir}/text \
      --hyp ${gen_dir}/text \
      > ${gen_dir}/scoring/final_result.txt
    cat ${gen_dir}/scoring/final_result.txt
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
