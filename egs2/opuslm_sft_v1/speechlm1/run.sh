#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_jsons="dump/raw_text_dialogue_tulu3/train/data.json"
valid_jsons="dump/raw_text_dialogue_tulu3/valid/data.json"

train_config=conf/train_delay_olmo2_7b.yaml
inference_config=conf/decode_general.yaml

token_list_dir=data/token_list/llm_vocab_olmo # use lllm vocab
bpe_opts="--subword_choice huggingface --subword_model allenai/OLMo-2-1124-7B"


./speechlm.sh \
    --skip_data_prep true \
    --data_combo_name tulu_sft \
    --fs 16000 \
    --ngpu 4 \
    --nj 16 \
    --inference_nj 16 \
    --nbest 10 \
    --gpu_inference true \
    --token_list_dir ${token_list_dir} \
    --train_config ${train_config} \
    --inference_config ${inference_config} \
    --audio_format "flac.ark" \
    --train_jsons "${train_jsons}" \
    --valid_jsons "${valid_jsons}" \
    --dumpdir dump \
    ${bpe_opts} \
    "$@"
