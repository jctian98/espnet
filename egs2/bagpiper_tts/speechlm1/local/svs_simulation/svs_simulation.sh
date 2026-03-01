#!/usr/bin/env bash
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
#
# Singing Voice Synthesis (SVS) Data Simulation Pipeline
# Transforms (singing audio + rich caption) into dialogue-format training data:
#   System prompt -> User SVS request -> CoT -> Rich Caption -> Audio
#
# Stage 0: Qwen3-ASR transcription to obtain lyrics reference
# Stage 1: LLM extracts lyrics from caption; filter by WER ≤ 20%
# Stage 2: Generate SVS user requests
# Stage 3: Generate chain-of-thought reasoning traces
# Stage 4: LLM-as-judge quality filtering
# Stage 5: Assemble dialogues and prepare dataset JSON

set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

# ---------------------------------------------------------------------------
# Default parameters
# ---------------------------------------------------------------------------
stage=0
stop_stage=100
version=v1

# Qwen3-ASR vLLM endpoints (Stage 0)
# cnode1-003: ports 8000-8007 (8 instances)
# cnode1-005: ports 8000-8007 (8 instances)
whisper_url=""
for port in 8000 8001 8002 8003 8004 8005 8006 8007; do
    whisper_url+="${whisper_url:+:}http://cnode1-003:${port}/v1"
done
for port in 8000 8001 8002 8003 8004 8005 8006 8007; do
    whisper_url+="${whisper_url:+:}http://cnode1-005:${port}/v1"
done
whisper_model="Qwen/Qwen3-ASR-1.7B"

# Text LLM vLLM endpoints (Stages 1-4)
# cnode1-005: port 8000
vllm_url="http://cnode1-004:8000/v1:http://cnode1-003:8000/v1:http://cnode1-005:8000/v1:http://cnode1-001:8000/v1"
vllm_url_stage1=${vllm_url}
vllm_url_stage2=${vllm_url}
vllm_url_stage3=${vllm_url}
vllm_url_stage4=${vllm_url}

model_stage1="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
model_stage2="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
model_stage3="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
model_stage4="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"

# Input data: metadata.jsonl (audiocaps-style)
# Fields: example_id, dataset, audio_path, duration, sample_rate, caption
input_file=/mnt/home/jinchuat-andr-d6b58f/jinchuat/data/raw_sft/music_2m/metadata.jsonl

num_workers=5000
whisper_workers=2048   # concurrent workers across all ASR server instances
whisper_timeout=120    # per-request ASR timeout (shorter than LLM generation timeout)
timeout=1200
resume=true
num_samples=-1        # -1 means all samples

# Stage 4 quality thresholds
min_score=3
avg_score=3.5

# WER threshold for Stage 1 (lenient for singing data)
wer_threshold=0.10

log "$0 $*"
. utils/parse_options.sh

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if [ -z "${version}" ]; then
    log "Error: --version is required (e.g. v1)"
    exit 1
fi

if [ -z "${input_file}" ]; then
    log "Error: --input_file is required"
    log "  Expected format: metadata.jsonl with fields:"
    log "    example_id, dataset, audio_path, duration, sample_rate, caption"
    exit 1
fi

# Derive output directory from input_file path
dataset_name=svs
output_base="data/${dataset_name}"
output_dir="${output_base}/${version}"
log "Input file   : ${input_file}"
log "Output dir   : ${output_dir}"

# ---------------------------------------------------------------------------
# Create directory structure
# ---------------------------------------------------------------------------
mkdir -p "${output_dir}/stage0_whisper"
mkdir -p "${output_dir}/stage1_extract_filter"
mkdir -p "${output_dir}/stage2_user_requests"
mkdir -p "${output_dir}/stage3_cot"
mkdir -p "${output_dir}/stage4_judge_filter"
mkdir -p "${output_dir}/stage5_dialogues"

resume_flag=""
if ${resume}; then
    resume_flag="--resume"
fi

# ---------------------------------------------------------------------------
# Stage 0: Whisper transcription to get lyrics reference
# ---------------------------------------------------------------------------
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "Stage 0: Qwen3-ASR transcription of singing audio"

    python3 local/svs_simulation/svs_whisper_transcribe.py \
        --input_file "${input_file}" \
        --output_dir "${output_dir}/stage0_whisper" \
        --whisper_url "${whisper_url}" \
        --model "${whisper_model}" \
        --num_workers ${whisper_workers} \
        --timeout ${whisper_timeout} \
        --num_samples ${num_samples} \
        ${resume_flag}

    stage0_output="${output_dir}/stage0_whisper/transcriptions.jsonl"
    if [ -f "${stage0_output}" ]; then
        count=$(wc -l < "${stage0_output}")
        log "Stage 0 completed: ${count} audio files transcribed"
    fi
fi

# ---------------------------------------------------------------------------
# Stage 1: Extract lyrics from caption; filter by WER ≤ wer_threshold
# ---------------------------------------------------------------------------
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Extracting lyrics from captions (WER ≤ ${wer_threshold})"

    stage0_output="${output_dir}/stage0_whisper/transcriptions.jsonl"
    if [ ! -f "${stage0_output}" ]; then
        log "Error: Stage 0 output not found: ${stage0_output}"
        exit 1
    fi

    python3 local/svs_simulation/svs_extract_and_filter_wer.py \
        --input_file "${stage0_output}" \
        --output_dir "${output_dir}/stage1_extract_filter" \
        --vllm_url "${vllm_url_stage1}" \
        --model "${model_stage1}" \
        --num_workers ${num_workers} \
        --timeout ${timeout} \
        --num_samples ${num_samples} \
        --version "${version}" \
        --wer_threshold "${wer_threshold}" \
        ${resume_flag}

    stage1_output="${output_dir}/stage1_extract_filter/stage1_filtered.jsonl"
    if [ -f "${stage1_output}" ]; then
        count=$(wc -l < "${stage1_output}")
        log "Stage 1 completed: ${count} samples passed WER ≤ ${wer_threshold}"
    fi
fi

# ---------------------------------------------------------------------------
# Stage 2: Generate SVS user requests
# ---------------------------------------------------------------------------
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Generating SVS user requests"

    stage1_output="${output_dir}/stage1_extract_filter/stage1_filtered.jsonl"
    if [ ! -f "${stage1_output}" ]; then
        log "Error: Stage 1 output not found: ${stage1_output}"
        exit 1
    fi

    python3 local/svs_simulation/svs_generate_user_requests.py \
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
        log "Stage 2 completed: ${count} SVS user requests generated"
    fi
fi

# ---------------------------------------------------------------------------
# Stage 3: Generate chain-of-thought reasoning traces
# ---------------------------------------------------------------------------
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "Stage 3: Generating chain-of-thought reasoning traces"

    stage2_output="${output_dir}/stage2_user_requests/user_requests.jsonl"
    if [ ! -f "${stage2_output}" ]; then
        log "Error: Stage 2 output not found: ${stage2_output}"
        exit 1
    fi

    python3 local/svs_simulation/svs_generate_cot.py \
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

# ---------------------------------------------------------------------------
# Stage 4: LLM-as-judge quality validation and filtering
# ---------------------------------------------------------------------------
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "Stage 4: LLM-as-judge quality validation (min=${min_score}, avg=${avg_score})"

    stage3_output="${output_dir}/stage3_cot/cot.jsonl"
    if [ ! -f "${stage3_output}" ]; then
        log "Error: Stage 3 output not found: ${stage3_output}"
        exit 1
    fi

    python3 local/svs_simulation/svs_judge_quality.py \
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

# ---------------------------------------------------------------------------
# Stage 5: Assemble dialogues and prepare dataset JSON
# ---------------------------------------------------------------------------
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    log "Stage 5: Assembling SVS dialogues"

    stage4_output="${output_dir}/stage4_judge_filter/filtered.jsonl"
    if [ ! -f "${stage4_output}" ]; then
        log "Error: Stage 4 output not found: ${stage4_output}"
        exit 1
    fi

    python3 local/svs_simulation/svs_assemble_dialogue.py \
        --input_file "${stage4_output}" \
        --output_dir "${output_dir}/stage5_dialogues" \
        --version "${version}"

    stage5_output="${output_dir}/stage5_dialogues/dialogues.jsonl"
    if [ -f "${stage5_output}" ]; then
        count=$(wc -l < "${stage5_output}")
        log "Stage 5 completed: ${count} SVS dialogues assembled"

        dataset_json="${output_dir}/stage5_dialogues/dataset.json"
        log "Stage 5: Preparing dataset JSON for training"
        PYTHONPATH=../../..:${PYTHONPATH:-} python3 ../../../espnet2/speechlm/bin/prepare_dataset_json.py \
            --triplets "dialogue,${stage5_output},dialogue" \
            --output_json "${dataset_json}"
        log "Stage 5: Dataset JSON saved to ${dataset_json}"
    fi
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
