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
stop_stage=1
nj=3000

# Revise this codec_opts to use different codec; revise dumpdir to dump the token into different dump folders.
codec_opts="--codec_choice bigvgan --codec_checkpoint_path bigvgan/checkpoints/model.ckpt --codec_config_path bigvgan/checkpoints/config.json --codec_batch_size 20"
tokenization_task="audiolm"
dumpdir=dump_rvq8_2k

codec_opts="--codec_choice bigvgan --codec_checkpoint_path bigvgan/checkpoints_fsq8_2k/model.ckpt --codec_config_path bigvgan/checkpoints_fsq8_2k/config.json --codec_batch_size 20"
tokenization_task="audiolm"
dumpdir=dump_fsq8_2k

# codec_opts="--codec_choice xcodec --codec_checkpoint_path xcodec/checkpoint/model.pth         --codec_config_path xcodec/checkpoint/config.yaml         --codec_batch_size 20  "
# tokenization_task="audiolm"
# dumpdir=dump_xcodec

# codec_opts="--codec_choice xcodec --codec_checkpoint_path xcodec/checkpoint/model.pth         --codec_config_path xcodec/checkpoint/config.yaml         --codec_batch_size 20 "
# tokenization_task="codec_ssl_audiolm"
# dumpdir=dump_xcodec_af3

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Process ETTA data for both understanding and generation"
    manifests=" \
    /lustre/fsw/portfolios/adlr/projects/adlr_audio_music/syn-cap-for-TTA-checkpoints/audioset_augmented_caption/AudioFlamingo2-3B-filtered-nvclap-organized/AudioSet_long_nvclap_geq_0.2_top1_AESfiltered.ndjson \
    /lustre/fsw/portfolios/adlr/projects/adlr_audio_music/syn-cap-for-TTA-checkpoints/audioset_augmented_caption/AudioFlamingo2-3B-filtered-nvclap-organized/AudioSet_short_nvclap_geq_0.2_top1_AESfiltered.ndjson \
    /lustre/fsw/portfolios/adlr/projects/adlr_audio_music/syn-cap-for-TTA-checkpoints/freesound-clap_augmented_caption/AudioFlamingo2-3B-filtered-nvclap-organized/FreeSound-clap_long_nvclap_geq_0.2_top1_AESfiltered.ndjson \
    /lustre/fsw/portfolios/adlr/projects/adlr_audio_music/syn-cap-for-TTA-checkpoints/freesound-clap_augmented_caption/AudioFlamingo2-3B-filtered-nvclap-organized/FreeSound-clap_short_nvclap_geq_0.2_top1_AESfiltered.ndjson \
    /lustre/fsw/portfolios/adlr/projects/adlr_audio_music/syn-cap-for-TTA-checkpoints/freesound-stereo-twins_augmented_caption/AudioFlamingo2-3B-filtered-nvclap-organized/FreeSound-stereo-twins_long_nvclap_geq_0.2_top1_AESfiltered.ndjson \
    /lustre/fsw/portfolios/adlr/projects/adlr_audio_music/syn-cap-for-TTA-checkpoints/freesound-stereo-twins_augmented_caption/AudioFlamingo2-3B-filtered-nvclap-organized/FreeSound-stereo-twins_short_nvclap_geq_0.2_top1_AESfiltered.ndjson \
    /lustre/fsw/portfolios/adlr/projects/adlr_audio_music/syn-cap-for-TTA-checkpoints/sounddescs_augmented_caption/AudioFlamingo2-3B-filtered-nvclap-organized/SoundDescs_long_nvclap_geq_0.2_top1_AESfiltered.ndjson \
    /lustre/fsw/portfolios/adlr/projects/adlr_audio_music/syn-cap-for-TTA-checkpoints/sounddescs_augmented_caption/AudioFlamingo2-3B-filtered-nvclap-organized/SoundDescs_short_nvclap_geq_0.2_top1_AESfiltered.ndjson \
    /lustre/fsw/portfolios/adlr/projects/adlr_audio_music/syn-cap-for-TTA-checkpoints/vggsound_augmented_caption/AudioFlamingo2-3B-filtered-nvclap-organized/VGGSound_long_nvclap_geq_0.2_top1_AESfiltered.ndjson \
    /lustre/fsw/portfolios/adlr/projects/adlr_audio_music/syn-cap-for-TTA-checkpoints/vggsound_augmented_caption/AudioFlamingo2-3B-filtered-nvclap-organized/VGGSound_short_nvclap_geq_0.2_top1_AESfiltered.ndjson \
    "

    bash local/convert_manifest_to_dialogue.sh \
      --stage 2 \
      --stop_stage 2 \
      --nj 16 \
      --manifests "${manifests}" \
      --dumpdir ${dumpdir} \
      --dataname etta_v2 \
      --tasks "text-to-audio" \
      --tokenization_task ${tokenization_task} \
      --tokenization_opts "${codec_opts}"

fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Process AF3 data for understanding "
    # python local/parse_data_manifest_af3.py \
    #   --prefix AF3 \
    #   --output_dir data
    
    # bash run.sh \
    #     --stage 2 \
    #     --stop_stage 2 \
    #     --nj ${nj} \
    #     --dumpdir ${dumpdir} \
    #     --task audiolm \
    #     --train_set AF3_all \
    #     --data_name AF3 \
    #     ${codec_opts}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Prepare text-only SFT data"

    # ACE Math
    dir=dump/raw_audio_dialogue_acemath
    python local/parse_data_acemath.py \
      --output_dir dump/raw_audio_dialogue_acemath
    for dset in `ls ${dir}`; do
        python pyscripts/utils/make_speechlm_json.py \
          --task audio_dialogue \
          --file_modality_type ${dir}/${dset}/dialogue,dialogue,dialogue_json \
          --output_json ${dir}/${dset}/data.json 
    done

    # Llama-Nemotron, very large so we process manually for efficiency
    huggingface-cli download \
      --repo-type dataset \
      --local-dir data/llama_nemotron_download \
      nvidia/Llama-Nemotron-Post-Training-Dataset
    
    dir=dump/raw_audio_dialogue_llama_nemotron
    python local/parse_data_llama_nemotron.py \
      --input_dir data/llama_nemotron_download/SFT \
      --output_dir ${dir}
    for dset in `ls ${dir}`; do
        python pyscripts/utils/make_speechlm_json.py \
          --task audio_dialogue \
          --file_modality_type ${dir}/${dset}/dialogue,dialogue,dialogue_json \
          --output_json ${dir}/${dset}/data.json 
    done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Prepare AudioSet evaluation"

    manifests="/lustre/fsw/portfolios/adlr/users/zkong/adlr_audio_music/syn-cap-for-TTA-checkpoints/AudioCaps-test.json"
    bash local/convert_manifest_to_dialogue.sh \
      --stage 1 \
      --stop_stage 3 \
      --nj 1 \
      --manifests "${manifests}" \
      --dumpdir ${dumpdir} \
      --dataname audioset \
      --tasks "text-to-audio" \
      --tokenization_task ${tokenization_task} \
      --tokenization_opts "${codec_opts}"
fi