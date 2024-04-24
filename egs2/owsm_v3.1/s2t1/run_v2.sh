#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train_v2
valid_set=dev_v2
test_sets=test

nbpe=5000
# s2t_config=conf/train_s2t_ebf_conv2d_size1024_e18_d18_piecewise_lr2e-4_warmup60k_flashattn.yaml
s2t_config=conf/train_s2t_ebf_conv2d_size384_e6_d6_piecewise_lr1e-3_warmup60k_flashattn_lessreg.yaml
s2t_config=conf/train_s2t_ebf_conv2d_size384_e6_d6_piecewise_lr1e-3_warmup60k_flashattn_lessreg_stc.yaml
inference_config=conf/decode_s2t.yaml

./s2t.sh \
    --stage 1 \
    --stop_stage 13 \
    --use_lm false \
    --num_nodes 1 \
    --ngpu 2 \
    --nj 64 \
    --gpu_inference true \
    --inference_nj 1 \
    --feats_type raw \
    --audio_format flac.ark \
    --token_type bpe \
    --nbpe ${nbpe} \
    --bpe_input_sentence_size 15000000 \
    --s2t_config "${s2t_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --bpe_train_text "dump/raw/${train_set}/text" \
    --bpe_nlsyms data/${train_set}/nlsyms.txt \
    --s2t_stats_dir exp/stats_v2 \
    --lm_train_text "dump/raw/${train_set}/text" "$@"
