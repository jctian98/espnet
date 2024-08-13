#!/bin/bash

# 語言列表
# langs=("ceb_ph")
# 提交每個語言的作業
task=asr
tag=run_speechlm_prepare_data_${task}

langs=(cy en de fr cy tt kab ca zh-TW it fa eu es ru tr nl eo zh-CN rw pt zh-HK cs pl uk)
fs=16000
codec_choice=ESPnet
codec_opts="--codec_choice ${codec_choice} --codec_hf_model_tag espnet/amuse_speech_soundstream_16k "
# codec_opts="--codec_choice ${codec_choice} "
log_dir=slurm_logs/${tag}_${codec_choice}_${fs}
mkdir -p ${log_dir}
# rm -r ${log_dir}/*.txt
# rm -r ${log_dir}/*.err
# for lang in "${undone_langs[@]}"
# do
#   sbatch --job-name=${tag}_${lang} \
#          --output=${log_dir}/${tag}_${lang}_%j.txt \
#          --error=${log_dir}/${tag}_${lang}_%j.err \
#          --time=2-00:00:00 \
#          --partition=RM-shared \
#          --cpus-per-task=9 \
#          --mem=17G \
#          --wrap="bash run_codec_unseen.sh ${lang}"
# done
stage=1
stop_stage=2
for lang in "${langs[@]}"
do
  sbatch --job-name=${tag}_${lang} \
         --output=${log_dir}/${tag}_${lang}.txt \
         --error=${log_dir}/${tag}_${lang}.err \
         --time=2-00:00:00 \
         --partition=RM-shared \
         --cpus-per-task=8 \
         --mem=15G \
         --wrap="bash run_asr.sh ${lang} ${fs} '${codec_opts}' ${stage} ${stop_stage}"
done