#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

stage=1
stop_stage=100

num_nodes=1
num_proc_per_node=8
node_rank=0
master_addr=localhost
master_port=8888


train_registered_specifier="/mnt/project/jinchuan/data/captioner/abab_pretrain_gemini_turn/audio_to_text.txt"
valid_registered_specifier="/mnt/project/jinchuan/data/captioner/abab_pretrain_gemini_turn/audio_to_text_valid.txt"

train_config=conf/train_stage1_qwen3_captioner.yaml

exp_dir=exp/stage1_qwen3_captioner
mkdir -p ${exp_dir}

inference_config=conf/inference.yaml
inference_step=10000
inference_nj=1

. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Node rank: ${node_rank} launch"

  mkdir -p ${exp_dir}/logs
  timestamp=$(date +"%Y-%m-%d_%H_%M")
  torchrun \
    --nnodes=${num_nodes} \
    --node_rank=${node_rank} \
    --nproc_per_node=${num_proc_per_node} \
    --master_addr=${master_addr} \
    --master_port=${master_port} \
      ../../../espnet2/speechlm/bin/train.py \
      --train-registered-specifier "${train_registered_specifier}" \
      --valid-registered-specifier "${valid_registered_specifier}" \
      --train-config ${train_config} \
      --output-dir ${exp_dir} \
      --save-loader-state \
      --wandb-mode online \
      --wandb-project anc \
      > ${exp_dir}/logs/train_node${node_rank}_${timestamp}.log 2>&1 
fi