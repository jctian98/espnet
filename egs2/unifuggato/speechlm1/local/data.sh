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

# codec_opts="--codec_choice xcodec --codec_checkpoint_path xcodec/checkpoint/model.pth         --codec_config_path xcodec/checkpoint/config.yaml         --codec_batch_size 50 "
# tokenization_task="codec_ssl_audiolm"
# dumpdir=dump_xcodec_af3

codec_opts="--codec_choice xcodec --codec_checkpoint_path xcodec/checkpoint/model.pth         --codec_config_path xcodec/checkpoint/config.yaml         --codec_batch_size 20 "
tokenization_task="audiolm"
dumpdir=dump_xcodec_v2

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
      --stage 3 \
      --stop_stage 3 \
      --nj 100 \
      --manifests "${manifests}" \
      --dumpdir ${dumpdir} \
      --dataname etta \
      --tasks "text-to-audio continuous_audio_generation" \
      --tokenization_task ${tokenization_task} \
      --tokenization_opts "${codec_opts}"

fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Process AF3 data for understanding "

    # Failed subset:
    # /lustre/fs12/portfolios/nvr/users/hucky/vila_speech_data/all_vila_speech_jsons/asr_TEDLIUM3_filtered_train.json
    # /lustre/fs12/portfolios/nvr/users/hucky/vila_speech_data/all_vila_speech_jsons/excd_asr_wsj_train_si284_whisper.json 
    # /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/SpokenSquadQA/train.json

    manifests="
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/YoutubeVeCaps/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/FisherSummaryQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/IEMOCAPEmoStateQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/FisherConnectingQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/DailyTalkDiverseQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/DailyTalkDetailQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/VoxPopuliReasonBehindQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MultiDialogDetailQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MeldConnectingQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/DailyTalkConnectingQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/SwitchboardOrderQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MeldReasonBehindQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MeldOrderQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MeldEmoReasonQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/LibriSpeechDetailQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/IEMOCAPDetailQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/FisherOrderQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/EuroParlSummaryQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/EuroParlOrderQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/SwitchboardDetailQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/LibriSpeechSummaryQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/LibriSpeechConnectingQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/IEMOCAPEmoReasonQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/EuroParlDiverseQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/VoxPopuliSummaryQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/VoxPopuliConnectingQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/SwitchboardReasonBehindQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/SwitchboardDiverseQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/IEMOCAPEmoSarQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/EuroParlDetailQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MultiDialogReasonBehindQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MeldRespondQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/LibriSpeechOrderQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/IEMOCAPSummaryQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/IEMOCAPRespondQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/IEMOCAPReasonBehindQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/DailyTalkReasonBehindQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/VoxPopuliOrderQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/SwitchboardSummaryQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MultiDialogRespondQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/LibriSpeechReasonBehindQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/IEMOCAPEmoFlipQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/DailyTalkSummaryQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/DailyTalkOrderQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/VoxPopuliDiverseQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/VoxPopuliDetailQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MultiDialogOrderQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MeldEmoStateQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MeldDiverseQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/LibriSpeechDiverseQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/IEMOCAPOrderQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/IEMOCAPDiverseQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/EuroParlReasonBehindQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MultiDialogDiverseQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MeldSummaryQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MeldEmoSarQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/FisherReasonBehindQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/FisherDetailQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MeldEmoFlipQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/FisherDiverseQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/EuroParlConnectingQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MultiDialogSummaryQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MeldDetailQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/IEMOCAPConnectingQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/DailyTalkRespondQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/SwitchboardRespondQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/SwitchboardConnectingQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MultiDialogConnectingQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MiraNeedleQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MiraPlotQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MiraTemporalQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/Mira-LongAudioCaptioning/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/Recap-LongAudioCaptioning/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/Recap-LongAudioDiverseQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/ReasoningData/train.json \
      /lustre/fsw/portfolios/adlr/projects/adlr_audio_music/datasets/AMI_ASR/ami/annotations/train.json \
      /lustre/fsw/portfolios/adlr/projects/adlr_audio_music/datasets/VoiceAssistant-400K/voiceassistant400k_final.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/VoxCeleb2ConnectingQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/VoxCeleb2DetailQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/VoxCeleb2DiverseQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/VoxCeleb2OrderQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/VoxCeleb2ReasonBehindQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/VoxCeleb2SummaryQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/mtg-jamendo-MusicCaptioning/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/Music4All/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MQDA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/CountingQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/DCASE-2025/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/SpeechSoundCaps/train.json \
      /lustre/fsw/portfolios/adlr/users/arushig/VILA-Internal/sift_100k.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/SpeechSoundCapsQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/GigaSpeech-Long-QA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/TemporalQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/GigaSpeech-QA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/GigaSpeech-Full-QA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/Music4AllQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MSD-MusicCaptioning/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/CV-ASR/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MELD-EmotionClassification/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/BBCSoundEffects-AudioDescription/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/SWBD-ASR/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/WavCaps-SoundBible-AudioCaptioning/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/AudioSet-Speech-Audio-QA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/SONYC-UST-EventClassification/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/VoxPopuli-ASR/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/FSD50k-EventClassification/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/SalmonnQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/YesNoQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/emov-db-EmotionClassification/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/LLARK_MagnaTagATune-mir/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/tess-EmotionClassification/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/Europarl-ASR/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/jl-corpus-EmotionClassification/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/Ego-10-AudioCaptioning/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/SPGI-ASR/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/LLARK_MTG-Jamendo-reasoning/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/CREMA-D-EmotionClassification/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MusicBenchQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/WavCaps-BBC_Sound_Effects-AudioCaptioning/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/NSynth-Instrument/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/NSynth-MIR/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/LLARK_MTG-Jamendo-mir/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/AudioEntailmentQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/GigaSpeech-ASR/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/WavCaps-AudioSet_SL-AudioCaptioning/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/NonSpeech7k-EventClassification/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/chime-home-EventClassification/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MusicCaps-AudioCaptioning/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/LP-MusicCaps-MSD-AudioCaptioning/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/Ego-30-AudioCaptioning/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/NSynth-Source/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/Clotho-v2-AudioCaptioning/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/LP-MusicCaps-MC-AudioCaptioning/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/Clotho-AQA-EventClassification/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/WavCaps-FreeSound-AudioCaptioning/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/LLARK_MagnaTagATune-reasoning/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/AudioSet-Temporal-Speech-Audio-QA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/TUT-EventClassification/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/ESC50-EventClassification/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/WavText5K-Tagging/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MELD-SentimentClassification/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/Music-AVQA-AQA_All/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/Music-AVQA-AVQA_All/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MACS-AudioCaptioning/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/Medley-solos-DB-InstrClassification/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/AudioSet-EventClassification/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/OMGEmotion-EmotionClassification/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/mtg-jamendo-MusicTagging/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/FMA-GenreClassification/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/Epidemic_sound-AudioCaptioning/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/CochlScene-SceneClassification/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/LLARK_FMA-reasoning/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/ravdess-EmotionClassification/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/CompA-R-AQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MU-LLAMA-AQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/musdbhq-InstrClassification/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/UrbanSound8K-EventClassification/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/audiocaps-AudioCaptioning/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/VocalSound-VocalClassification/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/CLAP_freesound-AudioCaptioning/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MMAUQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/SongDescriber-AudioCaptioning/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/HeySQuADQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/Mira-AudioCaptioning/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/Clotho-AQA-AQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/LibriSpeech-ASR/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/IEMOCAP-EmotionClassification/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/AudioSetFullwoAudioMusicCaps-EventClassification/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/MSP-PODCAST-Publish-1.9-EmotionClassification/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/OpenAQA-AQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/SoundDescs-AudioDescription/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/LibriSQA/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/LLARK_FMA-mir/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/LP-MusicCaps-MTT-AudioCaptioning/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/GTZAN-GenreClassification/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/musdbhq-captioning/train.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MSD-MusicQA/train_2.json \
      /lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/SIFT/train_2.json \
    "

    bash local/convert_manifest_to_dialogue.sh \
      --stage 3 \
      --stop_stage 3 \
      --nj 512 \
      --manifests "${manifests}" \
      --dumpdir ${dumpdir} \
      --dataname AF3 \
      --tasks "continuous_audio_caption" \
      --tokenization_task ${tokenization_task} \
      --tokenization_opts "${codec_opts}"
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