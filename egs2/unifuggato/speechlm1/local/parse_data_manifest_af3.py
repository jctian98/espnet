import argparse
import json
import os
import librosa
import shutil
import multiprocessing as mp

from pathlib import Path
from functools import partial

MAX_AUDIO_LEN = 5 * 60
raw_files = [
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/FisherSummaryQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/YoutubeVeCaps/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/IEMOCAPEmoStateQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/FisherConnectingQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/DailyTalkDiverseQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/DailyTalkDetailQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/VoxPopuliReasonBehindQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MultiDialogDetailQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MeldConnectingQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/DailyTalkConnectingQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/SwitchboardOrderQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MeldReasonBehindQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MeldOrderQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MeldEmoReasonQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/LibriSpeechDetailQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/IEMOCAPDetailQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/FisherOrderQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/EuroParlSummaryQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/EuroParlOrderQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/SwitchboardDetailQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/LibriSpeechSummaryQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/LibriSpeechConnectingQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/IEMOCAPEmoReasonQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/EuroParlDiverseQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/VoxPopuliSummaryQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/VoxPopuliConnectingQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/SwitchboardReasonBehindQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/SwitchboardDiverseQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/IEMOCAPEmoSarQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/EuroParlDetailQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MultiDialogReasonBehindQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MeldRespondQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/LibriSpeechOrderQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/IEMOCAPSummaryQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/IEMOCAPRespondQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/IEMOCAPReasonBehindQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/DailyTalkReasonBehindQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/VoxPopuliOrderQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/SwitchboardSummaryQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MultiDialogRespondQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/LibriSpeechReasonBehindQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/IEMOCAPEmoFlipQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/DailyTalkSummaryQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/DailyTalkOrderQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/VoxPopuliDiverseQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/VoxPopuliDetailQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MultiDialogOrderQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MeldEmoStateQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MeldDiverseQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/LibriSpeechDiverseQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/IEMOCAPOrderQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/IEMOCAPDiverseQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/EuroParlReasonBehindQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MultiDialogDiverseQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MeldSummaryQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MeldEmoSarQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/FisherReasonBehindQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/FisherDetailQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MeldEmoFlipQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/FisherDiverseQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/EuroParlConnectingQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MultiDialogSummaryQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MeldDetailQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/IEMOCAPConnectingQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/DailyTalkRespondQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/SwitchboardRespondQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/SwitchboardConnectingQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MultiDialogConnectingQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MiraNeedleQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MiraPlotQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MiraTemporalQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/Mira-LongAudioCaptioning/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/Recap-LongAudioCaptioning/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/Recap-LongAudioDiverseQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/ReasoningData/train.json",
    "/lustre/fsw/portfolios/adlr/projects/adlr_audio_music/datasets/AMI_ASR/ami/annotations/train.json",
    "/lustre/fsw/portfolios/adlr/projects/adlr_audio_music/datasets/VoiceAssistant-400K/voiceassistant400k_final.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/VoxCeleb2ConnectingQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/VoxCeleb2DetailQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/VoxCeleb2DiverseQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/VoxCeleb2OrderQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/VoxCeleb2ReasonBehindQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/VoxCeleb2SummaryQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/mtg-jamendo-MusicCaptioning/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/Music4All/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MQDA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/CountingQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/DCASE-2025/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/SpeechSoundCaps/train.json",
    "/lustre/fs12/portfolios/nvr/users/hucky/vila_speech_data/all_vila_speech_jsons/excd_asr_wsj_train_si284_whisper.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MSD-MusicQA/train_2.json",
    "/lustre/fsw/portfolios/adlr/users/arushig/VILA-Internal/sift_100k.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/SIFT/train_2.json",
    "/lustre/fs12/portfolios/nvr/users/hucky/vila_speech_data/all_vila_speech_jsons/asr_TEDLIUM3_filtered_train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/SpeechSoundCapsQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/GigaSpeech-Long-QA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/TemporalQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/GigaSpeech-QA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/GigaSpeech-Full-QA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/Music4AllQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MSD-MusicCaptioning/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/CV-ASR/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MELD-EmotionClassification/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/BBCSoundEffects-AudioDescription/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/SWBD-ASR/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/WavCaps-SoundBible-AudioCaptioning/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/AudioSet-Speech-Audio-QA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/SONYC-UST-EventClassification/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/VoxPopuli-ASR/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/FSD50k-EventClassification/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/SalmonnQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/YesNoQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/emov-db-EmotionClassification/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/LLARK_MagnaTagATune-mir/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/tess-EmotionClassification/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/Europarl-ASR/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/jl-corpus-EmotionClassification/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/Ego-10-AudioCaptioning/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/SPGI-ASR/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/LLARK_MTG-Jamendo-reasoning/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/CREMA-D-EmotionClassification/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MusicBenchQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/WavCaps-BBC_Sound_Effects-AudioCaptioning/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/NSynth-Instrument/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/SpokenSquadQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/NSynth-MIR/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/LLARK_MTG-Jamendo-mir/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/AudioEntailmentQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/GigaSpeech-ASR/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/WavCaps-AudioSet_SL-AudioCaptioning/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/NonSpeech7k-EventClassification/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/chime-home-EventClassification/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MusicCaps-AudioCaptioning/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/LP-MusicCaps-MSD-AudioCaptioning/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/Ego-30-AudioCaptioning/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/NSynth-Source/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/Clotho-v2-AudioCaptioning/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/LP-MusicCaps-MC-AudioCaptioning/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/Clotho-AQA-EventClassification/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/WavCaps-FreeSound-AudioCaptioning/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/LLARK_MagnaTagATune-reasoning/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/AudioSet-Temporal-Speech-Audio-QA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/TUT-EventClassification/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/ESC50-EventClassification/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/WavText5K-Tagging/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MELD-SentimentClassification/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/Music-AVQA-AQA_All/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/Music-AVQA-AVQA_All/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MACS-AudioCaptioning/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/Medley-solos-DB-InstrClassification/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/AudioSet-EventClassification/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/OMGEmotion-EmotionClassification/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/mtg-jamendo-MusicTagging/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/FMA-GenreClassification/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/Epidemic_sound-AudioCaptioning/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/CochlScene-SceneClassification/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/LLARK_FMA-reasoning/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/ravdess-EmotionClassification/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/CompA-R-AQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MU-LLAMA-AQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/musdbhq-InstrClassification/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/UrbanSound8K-EventClassification/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/audiocaps-AudioCaptioning/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/VocalSound-VocalClassification/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/CLAP_freesound-AudioCaptioning/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/MMAUQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/SongDescriber-AudioCaptioning/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/HeySQuADQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/Mira-AudioCaptioning/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/Clotho-AQA-AQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/LibriSpeech-ASR/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/IEMOCAP-EmotionClassification/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/AudioSetFullwoAudioMusicCaps-EventClassification/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/MSP-PODCAST-Publish-1.9-EmotionClassification/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/OpenAQA-AQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data/SoundDescs-AudioDescription/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/LibriSQA/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/LLARK_FMA-mir/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/LP-MusicCaps-MTT-AudioCaptioning/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/GTZAN-GenreClassification/train.json",
    "/lustre/fsw/portfolios/adlr/users/sreyang/iccv/audio_qa_data_w_duration/musdbhq-captioning/train.json"
]

def get_parser():
    parser = argparse.ArgumentParser(description='Processing the manifest from ETTA')
    
    # Add arguments
    parser.add_argument(
        '--prefix',
        type=str,
        help='Prefix of each dataset',
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        help='Output directory',
    )

    return parser

def get_audio_length(path, cache_dir):
    """Get audio length in seconds using librosa"""

    # (1) If no such file, return None
    if not Path(path).exists():
        print(f"path : {path} doesn't exist.")
        return None
    
    # (2) If path contains space, make a copy of it to cache
    if " " in path:
        example_name = Path(path).stem.replace(" ", "-")
        example_path = (cache_dir / example_name).resolve()
        if not example_path.exists():
            shutil.copy(path, example_path)
            print(f'Copy the file path with space: {path} -> {example_path}')
        path = example_path

    # (3) get the duration
    try:
        audio_len = librosa.get_duration(filename=path)
        if audio_len >= MAX_AUDIO_LEN:
            return None
        return path
    except Exception as e:
        print(f"Error processing {os.path.basename(path)}: {e}")
        return None

def get_audio_lengths(original_paths, cache_dir, num_processes=None):
    """Process multiple audio files in parallel to get their lengths"""
    fn = partial(get_audio_length, cache_dir=cache_dir)
    with mp.Pool(processes=num_processes or mp.cpu_count()) as pool:
        real_paths = pool.map(fn, original_paths)
    
    # original_path: (real_path, length)
    retval = {
        str(opath): str(rpath)
        for opath, rpath in zip(original_paths, real_paths)
        if rpath is not None
    }
    return retval

def process_one_json(json_file, output_dir):
    if (output_dir / '.done').exists():
        print('already finished this subset. Skip')
        return

    try:
        json_list = json.load(open(json_file))
    except:
        json_list = [json.loads(line) for line in open(json_file)]

    output_dir.mkdir(parents=True, exist_ok=True)

    # find all audio paths and make sure they are all valid
    all_audio = set()
    cache_dir = output_dir / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    for info in json_list:
        all_audio.add(info['sound'])
    retval = get_audio_lengths(all_audio, cache_dir)

    result_writer = open(output_dir / 'result.json', 'w')
    json.dump(retval, result_writer)

    (output_dir / '.done').touch()

def main():
    parser = get_parser()
    args = parser.parse_args()

    subdirs = list()
    for json_file in raw_files:
        subset_name = args.prefix + "_" + Path(json_file).parent.stem
        print(f'processing: {json_file}')
        process_one_json(json_file, args.output_dir / subset_name)
        subdirs.append(args.output_dir / subset_name)
    
    results = dict()
    for subdir in subdirs:
        result = json.load(open(subdir / 'result.json'))
        results.update(result)

    output_dir = args.output_dir / f'{args.prefix}_all'
    output_dir.mkdir(parents=True, exist_ok=True)
    results_writer = open(output_dir /'result.json', 'w')
    json.dump(results, results_writer)

    wav_scp_writer = open(output_dir / 'wav.scp', 'w')
    for path in results.values():
        wav_scp_writer.write(f"{path} {path}\n")
    

if __name__ == "__main__":
    main()