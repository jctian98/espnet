#!/usr/bin/env bash
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
#
# Intention-to-Speech (ITS) Data Simulation Pipeline
# Transforms transcriptions with rich captions into dialogue-format training data
# where the user provides communicative INTENTION (not exact words) and the model
# infers the transcription.
#
# Input: JSONL from general TTS stage1 (with example_id, caption, whisper_lyrics, etc.)
# Output: System prompt -> Intention request -> CoT (with inferred text) -> Caption -> Audio

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
vllm_url="http://cnode1-004:8000/v1:http://cnode1-003:8000/v1:http://cnode1-005:8000/v1:http://cnode1-001:8000/v1:http://cnode1-002:8000/v1"

vllm_url_stage1=${vllm_url}
vllm_url_stage2=${vllm_url}
vllm_url_stage3=${vllm_url}
vllm_url_stage4=${vllm_url}

model_stage1="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
model_stage2="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
model_stage3="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
model_stage4="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"

# Input file: JSONL from general TTS stage1 filtered output
input_file="data/general_tts/v1/stage1_extract_filter/stage1_filtered.jsonl"

num_workers=5000
timeout=1200
resume=true
num_samples=-1  # -1 means process all samples

# Stage 1 intent score threshold (1-5)
stage1_min_score=4

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
output_dir="data/intention_to_speech/${version}"
log "Output directory: ${output_dir}"
log "Input file: ${input_file}"

# Create directory structure
mkdir -p "${output_dir}/stage1_filter"
mkdir -p "${output_dir}/stage2_user_requests"
mkdir -p "${output_dir}/stage3_cot"
mkdir -p "${output_dir}/stage4_judge_filter"
mkdir -p "${output_dir}/stage5_dialogues"

# Resume flag
resume_flag=""
if ${resume}; then
    resume_flag="--resume"
fi

# Stage 1: Filter transcriptions suitable for intention-to-speech
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Filtering transcriptions for ITS suitability"

    if [ ! -f "${input_file}" ]; then
        log "Error: Input file not found: ${input_file}"
        exit 1
    fi

    python3 local/intention_to_speech_simulation/its_filter_transcriptions.py \
        --input_file "${input_file}" \
        --output_dir "${output_dir}/stage1_filter" \
        --vllm_url "${vllm_url_stage1}" \
        --model "${model_stage1}" \
        --num_workers ${num_workers} \
        --timeout ${timeout} \
        --num_samples ${num_samples} \
        --version "${version}" \
        --min_score ${stage1_min_score} \
        ${resume_flag}

    stage1_output="${output_dir}/stage1_filter/stage1_filtered.jsonl"
    if [ -f "${stage1_output}" ]; then
        count=$(wc -l < "${stage1_output}")
        log "Stage 1 completed: ${count} samples passed suitability filter"
    fi
fi

# Stage 2: Generate intention-only user requests
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Generating intention-only user requests"

    stage1_output="${output_dir}/stage1_filter/stage1_filtered.jsonl"
    if [ ! -f "${stage1_output}" ]; then
        log "Error: Stage 1 output not found: ${stage1_output}"
        exit 1
    fi

    python3 local/intention_to_speech_simulation/its_generate_user_requests.py \
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
        log "Stage 2 completed: ${count} intention requests generated"
    fi
fi

# Stage 3: Generate chain-of-thought with transcription inference
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Generating CoT with transcription inference"

    stage2_output="${output_dir}/stage2_user_requests/user_requests.jsonl"
    if [ ! -f "${stage2_output}" ]; then
        log "Error: Stage 2 output not found: ${stage2_output}"
        exit 1
    fi

    python3 local/intention_to_speech_simulation/its_generate_cot.py \
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

    python3 local/intention_to_speech_simulation/its_judge_quality.py \
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

    python3 local/intention_to_speech_simulation/its_assemble_dialogue.py \
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
