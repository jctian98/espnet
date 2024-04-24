s2t_exp=exp/s2t_owsm_base
s2t_exp=exp/s2t_v2_stc_penalty05
inference_s2t_model="valid.total_count.ave.pth"
inference_s2t_model="valid.total_count.best.pth"
beam_size=1
ctc_weight=1.0

. ./utils/parse_options.sh 

./run.sh --stage 12 --stop_stage 13 \
    --test_sets "test" \
    --inference_s2t_model ${inference_s2t_model} \
    --s2t_exp ${s2t_exp} \
    --cleaner whisper_en --hyp_cleaner whisper_en \
    --inference_nj 1 \
    --feats_type raw \
    --audio_format flac.ark \
    --inference_args "--beam_size ${beam_size} --ctc_weight ${ctc_weight} --lang_sym eng"

# decode long
# eng_corpus="tst2020_long tst2021_long tst2022_long"
# eng_corpus="tst2020_long"
# for corp in $eng_corpus; do
#     ./run.sh --stage 12 --stop_stage 13 \
#         --test_sets ${corp} \
#         --inference_s2t_model ${inference_s2t_model} \
#         --s2t_exp ${s2t_exp} \
#         --cleaner whisper_en --hyp_cleaner whisper_en \
#         --inference_nj 1 \
#         --feats_type raw \
#         --audio_format flac.ark \
#         --decode_long true \
#         --skip_train true \
#         --inference_args "--beam_size ${beam_size} --ctc_weight ${ctc_weight} --lang_sym eng" &
# done; wait