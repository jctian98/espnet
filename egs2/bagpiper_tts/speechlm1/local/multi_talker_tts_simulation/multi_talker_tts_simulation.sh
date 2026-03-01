#!/usr/bin/env bash
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
#
# Multi-Talker TTS Data Simulation Pipeline
# Transforms (text + audio + rich caption) from multi-speaker recordings
# into dialogue-format training data:
#   System prompt -> User request -> CoT -> Rich Caption -> Audio

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

# Default parameters
stage=1
stop_stage=100
version=v1

# vLLM URLs: use ":" to separate multiple URLs serving the same model
vllm_url=
for id in 03 05 06; do
    vllm_url+="${vllm_url:+:}http://cnode1-0${id}:8000/v1"
done

vllm_url_stage1=${vllm_url}
vllm_url_stage2=${vllm_url}
vllm_url_stage3=${vllm_url}
vllm_url_stage4=${vllm_url}

model_stage1="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
model_stage2="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
model_stage3="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
model_stage4="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"

# Data root for multi-talker data (flat directory with wav.scp, text, rich_captions.jsonl)
dataset=train
data_root="/mnt/home/jinchuat-andr-d6b58f/jinchuat/espnet_speechlm_tts/egs2/gigaspeech/asr1/data/${dataset}_multi_talker"

num_workers=5000
timeout=1200
resume=true
num_samples=-1  # -1 means process all samples

# Stage 1 specific: revision loop parameters
max_revision_passes=3
wer_threshold=0.02
discard_threshold=0.15

# Stage 4 filtering thresholds
min_score=3
avg_score=3.5

log "$0 $*"
. utils/parse_options.sh

# Check required parameter
if [ -z "${version}" ]; then
    log "Error: --version is required"
    log "Usage: $0 --version <version_tag> [options]"
    exit 1
fi

# Set output directory with version
output_dir="data/${dataset}_multi_talker/${version}"
log "Output directory: ${output_dir}"

# Create directory structure
mkdir -p "${output_dir}/stage1_extract_revise"
mkdir -p "${output_dir}/stage2_user_requests"
mkdir -p "${output_dir}/stage3_cot"
mkdir -p "${output_dir}/stage4_judge_filter"
mkdir -p "${output_dir}/stage5_dialogues"

# Resume flag
resume_flag=""
if ${resume}; then
    resume_flag="--resume"
fi

# Stage 1: Extract transcription with revision loop
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Extracting transcription with revision loop (WER < ${wer_threshold}, discard > ${discard_threshold})"

    python3 local/multi_talker_tts_simulation/mt_tts_extract_revise_wer.py \
        --data_root "${data_root}" \
        --output_dir "${output_dir}/stage1_extract_revise" \
        --vllm_url "${vllm_url_stage1}" \
        --model "${model_stage1}" \
        --num_workers ${num_workers} \
        --timeout ${timeout} \
        --num_samples ${num_samples} \
        --version "${version}" \
        --wer_threshold ${wer_threshold} \
        --discard_threshold ${discard_threshold} \
        --max_revision_passes ${max_revision_passes} \
        ${resume_flag}

    stage1_output="${output_dir}/stage1_extract_revise/stage1_filtered.jsonl"
    if [ -f "${stage1_output}" ]; then
        count=$(wc -l < "${stage1_output}")
        log "Stage 1 completed: ${count} samples passed WER filter"
    fi
fi

# Stage 2: Generate multi-talker user requests
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Generating multi-talker user requests"

    stage1_output="${output_dir}/stage1_extract_revise/stage1_filtered.jsonl"
    if [ ! -f "${stage1_output}" ]; then
        log "Error: Stage 1 output not found: ${stage1_output}"
        exit 1
    fi

    python3 local/multi_talker_tts_simulation/mt_tts_generate_user_requests.py \
        --input_file "${stage1_output}" \
        --output_dir "${output_dir}/stage2_user_requests" \
        --vllm_url "${vllm_url_stage2}" \
        --model "${model_stage2}" \
        --num_workers ${num_workers} \
        --timeout ${timeout} \
        --num_samples ${num_samples} \
        --version "${version}" \
        ${resume_flag}

    stage2_output="${output_dir}/stage2_user_requests/user_requests.jsonl"
    if [ -f "${stage2_output}" ]; then
        count=$(wc -l < "${stage2_output}")
        log "Stage 2 completed: ${count} user requests generated"
    fi
fi

# Stage 3: Generate chain-of-thought reasoning traces
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Generating chain-of-thought reasoning traces"

    stage2_output="${output_dir}/stage2_user_requests/user_requests.jsonl"
    if [ ! -f "${stage2_output}" ]; then
        log "Error: Stage 2 output not found: ${stage2_output}"
        exit 1
    fi

    python3 local/multi_talker_tts_simulation/mt_tts_generate_cot.py \
        --input_file "${stage2_output}" \
        --output_dir "${output_dir}/stage3_cot" \
        --vllm_url "${vllm_url_stage3}" \
        --model "${model_stage3}" \
        --num_workers ${num_workers} \
        --timeout ${timeout} \
        --num_samples ${num_samples} \
        --version "${version}" \
        ${resume_flag}

    stage3_output="${output_dir}/stage3_cot/cot.jsonl"
    if [ -f "${stage3_output}" ]; then
        count=$(wc -l < "${stage3_output}")
        log "Stage 3 completed: ${count} CoT traces generated"
    fi
fi

# Stage 4: LLM-as-judge quality validation and filtering
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: LLM-as-judge quality validation (min_score=${min_score}, avg_score=${avg_score})"

    stage3_output="${output_dir}/stage3_cot/cot.jsonl"
    if [ ! -f "${stage3_output}" ]; then
        log "Error: Stage 3 output not found: ${stage3_output}"
        exit 1
    fi

    python3 local/multi_talker_tts_simulation/mt_tts_judge_quality.py \
        --input_file "${stage3_output}" \
        --output_dir "${output_dir}/stage4_judge_filter" \
        --vllm_url "${vllm_url_stage4}" \
        --model "${model_stage4}" \
        --num_workers ${num_workers} \
        --timeout ${timeout} \
        --num_samples ${num_samples} \
        --min_score "${min_score}" \
        --avg_score "${avg_score}" \
        ${resume_flag}

    stage4_output="${output_dir}/stage4_judge_filter/filtered.jsonl"
    if [ -f "${stage4_output}" ]; then
        count=$(wc -l < "${stage4_output}")
        log "Stage 4 completed: ${count} samples passed quality filter"
    fi
fi

# Stage 5: Assemble dialogues
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "Stage 5: Assembling dialogues"

    stage4_output="${output_dir}/stage4_judge_filter/filtered.jsonl"
    if [ ! -f "${stage4_output}" ]; then
        log "Error: Stage 4 output not found: ${stage4_output}"
        exit 1
    fi

    python3 local/multi_talker_tts_simulation/mt_tts_assemble_dialogue.py \
        --input_file "${stage4_output}" \
        --output_dir "${output_dir}/stage5_dialogues" \
        --version "${version}"

    stage5_output="${output_dir}/stage5_dialogues/dialogues.jsonl"
    if [ -f "${stage5_output}" ]; then
        count=$(wc -l < "${stage5_output}")
        log "Stage 5 completed: ${count} dialogues assembled"

        # Prepare dataset JSON for training
        dataset_json="${output_dir}/stage5_dialogues/dataset.json"
        log "Stage 5: Preparing dataset JSON"
        PYTHONPATH=../../..:${PYTHONPATH:-} python3 ../../../espnet2/speechlm/bin/prepare_dataset_json.py \
            --triplets "dialogue,${stage5_output},dialogue" \
            --output_json "${dataset_json}"
        log "Stage 5: Dataset JSON saved to ${dataset_json}"
    fi
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
