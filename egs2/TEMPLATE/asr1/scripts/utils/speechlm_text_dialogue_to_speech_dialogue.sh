#!/usr/bin/env bash

# Copyright 2025 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# With the given text dialogue data, generate the spoken dialogue data for training and evaluation

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

SECONDS=0

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

stage=1
stop_stage=100
nj=16                # number of parallel jobs
fs=16000            # sampling frequency
python=python3      # Specify python to execute espnet commands.

# Overall settings
input_dir=                                                                          # input text dialogue data
output_dir=                                                                         # output spoken dialogue data
ready_audio_list="none"                                                             # reference audio file. Can be wav-list or kaldi-ark list
user_prompt_list=dump/raw_codec_ssl_tts_yodas/train_yodas/index_files/wav.scp       # prompt to generate user speech
assistant_prompt_list=data/assistant_prompt.scp                                     # prompt to generate assistant speech
task=audio_dialogue                                                                 # audio dialogue or audio-text dialogue

# Tokenizers
codec_choice=ESPnet
codec_hf_model_tag=ftshijt/espnet_codec_dac_large_v1.4_360epoch
ssl_choice=espnet_hubert
ssl_nlayer=18
ssl_checkpoint_path=exp/kmeans/38epoch.pth
ssl_kmeans_path=exp/kmeans/xeus_18_5000clusters/km_5000.mdl
ssl_batch_bins=5000000

# TTS simulation
exp_tag=Opuslm_7b_baseline
inference_model=5epoch.pth
expdir=exp
nbest=10
inference_config=conf/decode_general.yaml

. ./utils/parse_options.sh
. ./cmd.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Find all necessary audio segments for generation and tokenization"
    python pyscripts/utils/speechlm_dialogue_prepare.py \
      --input_dir ${input_dir} \
      --output_dir ${output_dir} \
      --task ${task} \
      --ready_audio_list ${ready_audio_list} \
      --assistant_prompt_list ${assistant_prompt_list} \
      --user_prompt_list ${user_prompt_list}
fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    if [ $(wc -l < ${output_dir}/tokenization/wav.scp) -eq 0 ]; then
        log "Skip tokenization, as there are zero lines"
    else
        log "Tokenizating ${output_dir}/tokenization"
        scripts/audio/format_wav_scp.sh \
            --nj "${nj}" \
            --cmd "${train_cmd}" \
            --audio-format "flac.ark" \
            --fs "${fs}" \
            --out_filename wav.scp \
            ${output_dir}/tokenization/wav.scp \
            ${output_dir}/tokenization/audio_raw
        
        mkdir -p ${output_dir}/tokenization/audio
        cp ${output_dir}/tokenization/audio_raw/utt2num_samples ${output_dir}/tokenization/audio
        scripts/feats/codec_ssl_tokenization.sh \
            --src_dir ${output_dir}/tokenization/audio_raw \
            --tgt_dir ${output_dir}/tokenization/audio \
            --file_name wav.scp \
            --fs ${fs} \
            --nj ${nj} \
            --codec_choice ${codec_choice} \
            --codec_hf_model_tag ${codec_hf_model_tag} \
            --codec_dump_audio false \
            --ssl_choice ${ssl_choice} \
            --ssl_checkpoint_path ${ssl_checkpoint_path} \
            --ssl_kmeans_path ${ssl_kmeans_path} \
            --ssl_nlayer ${ssl_nlayer} \
            --ssl_batch_bins ${ssl_batch_bins}
    fi
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    if [ $(wc -l < ${output_dir}/generation/wav.scp) -eq 0 ]; then
        log "Skip TTS generation, as there are zero lines"
    else
        python pyscripts/utils/make_speechlm_json.py \
          --task codec_ssl_tts \
          --output_json ${output_dir}/generation/data.json \
          --file_modality_type ${output_dir}/generation/text,text_bpe,text \
          --file_modality_type ${output_dir}/generation/wav.scp,codec_ssl,kaldi_ark \
          --file_modality_type ${output_dir}/generation/utt2spk,spk,kaldi_ark \
        
        bash run.sh \
          --stage 9 --stop_stage 10 \
          --skip_data_prep true \
          --inference_nj ${nj} \
          --tag ${exp_tag} \
          --inference_model ${inference_model} \
          --inference_config ${inference_config} \
          --expdir ${expdir} \
          --nbest ${nbest} \
          --test_jsons ${output_dir}/generation/data.json \
          --inference_dir ${output_dir}/generation/inference
        
        infer_dir=${output_dir}/generation/inference/codec_ssl_tts_generation
        utils/filter_scp.pl ${infer_dir}/scoring/selected_examples \
          ${infer_dir}/wav.scp_token.scp |\
          awk '{sub(/_sample[0-9]+$/, "", $1); $1=$1; print}' |\
          awk '{sub(/^codec_ssl_tts_/, ""); print}' \
          > ${output_dir}/generation/generated_token.scp
    fi
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Assemble all audio results"
    python pyscripts/utils/speechlm_dialogue_assemble.py \
      --input_dir ${input_dir} \
      --output_dir ${output_dir} \
      --task ${task} \
      --ready_audio ${output_dir}/tokenization/audio/wav.scp \
      --ready_audio ${output_dir}/generation/generated_token.scp \
      --ready_audio ${output_dir}/ark/wav.scp
    
    cp ${output_dir}/data/dialogue.1 ${output_dir}/dialogue
fi
