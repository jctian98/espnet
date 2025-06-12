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

stage=1
stop_stage=3
nj=16

manifests=
dumpdir=
dataname=
tokenization_task="audiolm" # audiolm for codec-only; codec_ssl_audiolm for codec+ssl
tasks="text-to-audio"
tokenization_opts=
maxlen=120

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

find_stem() {
    local file="$1"
    local filename=$(basename "$file")
    local stem="${filename%.*}"
    
    if [ "$stem" = "train" ] || [ "$stem" = "train_2" ]; then
        local parent_dir=$(dirname "$file")
        local parent_stem=$(basename "$parent_dir")
        local parent_stem=${parent_stem%.*}
        echo "$parent_stem"
    else
        echo "$stem"
    fi
}

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Prepare wav.scp and segments for tokenization"
    python3 local/parse_data_manifest.py \
      --manifests ${manifests} \
      --prefix ${dataname} \
      --output_dir data
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Tokenization"

    # First, dump all audio to disk
    bash run.sh \
    --stage 2 \
    --stop_stage 2 \
    --nj ${nj} \
    --dumpdir ${dumpdir} \
    --task ${tokenization_task} \
    --train_set ${dataname}_all \
    --data_name ${dataname} \
    ${tokenization_opts}

    # Second, tokenization
    bash run.sh \
    --stage 5 \
    --stop_stage 5 \
    --nj ${nj} \
    --dumpdir ${dumpdir} \
    --task ${tokenization_task} \
    --train_set ${dataname}_all \
    --data_name ${dataname} \
    ${tokenization_opts}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Assemble to dialogue"
    if [ ${tokenization_task} == "codec_ssl_audiolm" ]; then
        audio_modality="codec_ssl"
    elif [ ${tokenization_task} == "audiolm" ]; then
        audio_modality="codec"
    else
        log "Invalid tokenization task ${tokenization_task}"
    fi

    for task in ${tasks}; do
        python local/prepare_dialogue.py \
        --manifests ${manifests} \
        --dumpdir ${dumpdir}/raw_${tokenization_task}_${dataname} \
        --mapping data/${dataname}_all/mapping \
        --prefix ${dataname} \
        --task $task \
        --audio_modality ${audio_modality}

        for dset in `ls ${dumpdir}/raw_audio_dialogue_${dataname}`; do
            dir=${dumpdir}/raw_audio_dialogue_${dataname}/${dset}
            if [ -f ${dir}/dialogue ]; then
                python pyscripts/utils/make_speechlm_json.py \
                --task audio_dialogue \
                --file_modality_type ${dir}/dialogue,dialogue,dialogue_json \
                --output_json ${dir}/data.json
            fi
        done
    done
fi