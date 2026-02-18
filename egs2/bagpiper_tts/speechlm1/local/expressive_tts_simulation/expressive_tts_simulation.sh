#!/usr/bin/env bash
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
#
# Expressive TTS Data Simulation Pipeline
# Transforms (text + audio + rich caption) into dialogue-format training data:
#   System prompt -> User request -> CoT -> Rich Caption -> Audio
#
# Supports two input modes:
#   1. LibriTTS-R format: --data_root + --subset (Kaldi files + captions)
#   2. JSONL format: --metadata_jsonl + --transcription_jsonl (metadata + transcription)

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
for id in 02 03 05 06 10; do
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

# Input mode 1: LibriTTS-R format (data_root + subset)
data_root=
subset="genshin-voice"

# Input mode 2: JSONL format (metadata + transcription)
metadata_jsonl="/mnt/home/haoranw4-andr-49167f/data/sft_data/part3_known_high_quality/metadata/genshin-voice/metadata.jsonl"
transcription_jsonl="/mnt/home/haoranw4-andr-49167f/data/sft_data/part3_known_high_quality/metadata/genshin-voice/transcription.jsonl"

num_workers=5000
timeout=1200
resume=true
num_samples=-1  # -1 means process all samples

# Stage 4 filtering thresholds
min_score=3
avg_score=3.5

log "$0 $*"
. utils/parse_options.sh

output_base="data/${subset}"

# Check required parameter
if [ -z "${version}" ]; then
    log "Error: --version is required"
    log "Usage: $0 --version <version_tag> [options]"
    exit 1
fi

# Set output directory with version
output_dir="${output_base}/${version}"
log "Output directory: ${output_dir}"

# Create directory structure
mkdir -p "${output_dir}/stage1_extract_filter"
mkdir -p "${output_dir}/stage2_user_requests"
mkdir -p "${output_dir}/stage3_cot"
mkdir -p "${output_dir}/stage4_judge_filter"
mkdir -p "${output_dir}/stage5_dialogues"

# Resume flag
resume_flag=""
if ${resume}; then
    resume_flag="--resume"
fi

# Stage 1: Extract transcription from rich captions and filter by WER=0
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Extracting transcription from rich captions and filtering by WER=0"

    # Build Stage 1 input arguments based on input mode
    stage1_input_args=""
    if [ -n "${metadata_jsonl}" ] && [ -n "${transcription_jsonl}" ]; then
        log "Stage 1: Using JSONL input mode"
        stage1_input_args="--metadata_jsonl ${metadata_jsonl} --transcription_jsonl ${transcription_jsonl}"
    elif [ -n "${data_root}" ]; then
        log "Stage 1: Using LibriTTS-R input mode"
        stage1_input_args="--data_root ${data_root} --subsets ${subset}"
    else
        log "Error: Must provide either (--metadata_jsonl + --transcription_jsonl) or --data_root"
        exit 1
    fi

    python3 local/expressive_tts_simulation/tts_extract_and_filter_wer.py \
        ${stage1_input_args} \
        --output_dir "${output_dir}/stage1_extract_filter" \
        --vllm_url "${vllm_url_stage1}" \
        --model "${model_stage1}" \
        --num_workers ${num_workers} \
        --timeout ${timeout} \
        --num_samples ${num_samples} \
        --version "${version}" \
        ${resume_flag}

    stage1_output="${output_dir}/stage1_extract_filter/stage1_filtered.jsonl"
    if [ -f "${stage1_output}" ]; then
        count=$(wc -l < "${stage1_output}")
        log "Stage 1 completed: ${count} samples passed WER=0 filter"
    fi
fi

# Stage 2: Generate user requests
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Generating user requests"

    stage1_output="${output_dir}/stage1_extract_filter/stage1_filtered.jsonl"
    if [ ! -f "${stage1_output}" ]; then
        log "Error: Stage 1 output not found: ${stage1_output}"
        exit 1
    fi

    python3 local/expressive_tts_simulation/tts_generate_user_requests.py \
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

    python3 local/expressive_tts_simulation/tts_generate_cot.py \
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

    python3 local/expressive_tts_simulation/tts_judge_quality.py \
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

    python3 local/expressive_tts_simulation/tts_assemble_dialogue.py \
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
