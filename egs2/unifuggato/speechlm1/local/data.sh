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

stage=2
stop_stage=2
nj=3000

# Revise this codec_opts to use different codec; revise dumpdir to dump the token into different dump folders.
codec_opts="--codec_choice bigvgan --codec_checkpoint_path bigvgan/checkpoints/model.ckpt --codec_config_path bigvgan/checkpoints/config.json --codec_batch_size 20"
dumpdir=dump_rvq8_2k

# codec_opts="--codec_choice ESPnet  --codec_hf_model_tag ftshijt/espnet_codec_dac_large_v1.4_360epoch "
# dumpdir=dump_espnet

# codec_opts="--codec_choice xcodec --codec_checkpoint_path xcodec/checkpoint/model.pth         --codec_config_path xcodec/checkpoint/config.yaml         --codec_batch_size 20 --dumpdir dump_xcodec "
# dumpdir=dump_xcodec

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Process ETTA data for both understanding and generation"
    # (1) Prepare data for tokenization
    # python3 local/parse_data_manifest_etta.py \
    #   --prefix etta_train \
    #   --output_dir data
    
    # (2) tokenization.
    # NOTE(Jinchuan): it's recommend to run stage 2 and stage 5 separately
    # on Nvidia cluster, with "local" command and nj=2000 for stage 2 and
    # "slurm" command and nj=64 for stage 5. Stage 3 and 4 is empty.
    bash run.sh \
      --stage 5 \
      --stop_stage 5 \
      --nj ${nj} \
      --dumpdir ${dumpdir} \
      --task audiolm \
      --train_set etta_train \
      --data_name etta \
      ${codec_opts}

    # (3) Build the dialogue data based on tokenized audio
    python3 local/prepare_dialogue_etta.py \
        --output_dir ${dumpdir}/raw_audio_dialogue_etta \
        --wav_length data/etta_train/reco2dur \
        --audio_wav_scp ${dumpdir}/raw_audiolm_etta/etta_train/wav.scp \
        --task audio-to-text
    python3 local/prepare_dialogue_etta.py \
        --output_dir ${dumpdir}/raw_audio_dialogue_etta \
        --wav_length data/etta_train/reco2dur \
        --audio_wav_scp ${dumpdir}/raw_audiolm_etta/etta_train/wav.scp \
        --task text-to-audio
    # python3 local/prepare_dialogue_etta.py \
    #     --output_dir ${dumpdir}/raw_audio_dialogue_etta \
    #     --wav_length data/etta_train/reco2dur \
    #     --audio_wav_scp ${dumpdir}/audio_raw_audiolm_etta/etta_train/wav.scp \
    #     --task continuous_audio_caption

    for dset in `ls ${dumpdir}/raw_audio_dialogue_etta`; do
        python pyscripts/utils/make_speechlm_json.py \
          --task audio_dialogue \
          --file_modality_type ${dumpdir}/raw_audio_dialogue_etta/${dset}/dialogue,dialogue,dialogue_json \
          --output_json ${dumpdir}/raw_audio_dialogue_etta/${dset}/data.json 
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Process AF3 data for understanding "
    python local/parse_data_manifest_af3.py \
      --prefix AF3 \
      --output_dir data
    
    bash run.sh \
        --stage 2 \
        --stop_stage 2 \
        --nj ${nj} \
        --dumpdir ${dumpdir} \
        --task audiolm \
        --train_set AF3_all \
        --data_name AF3 \
        ${codec_opts}

    # for dset in `ls data | grep ^AF3` ; do
    #     wc -l data/${dset}/wav.scp
    #     echo "$(< data/${dset}/wav.scp | wc -l)" data/${dset}/wav.scp
    #     if [ $(wc -l < "data/${dset}/wav.scp") -gt 0 ]; then
    #         bash run.sh \
    #         --stage 2 \
    #         --stop_stage 2 \
    #         --nj ${nj} \
    #         --dumpdir ${dumpdir} \
    #         --task audiolm \
    #         --train_set ${dset} \
    #         --data_name AF3 \
    #         ${codec_opts}
    #     else
    #         echo "Skip ${dset} as it is empty"
    #     fi
    #     break
    # done
fi