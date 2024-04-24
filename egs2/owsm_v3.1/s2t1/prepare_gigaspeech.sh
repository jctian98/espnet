tag=
train_set=train
dev_set=dev
test_set=test

. ./utils/parse_options.sh 

if [ -z $tag ]; then
    echo "empty tag. Exit" && exit 1;
fi

# --include_thre: untranscribed clips longer than this threshold will be included in audio segment
# --noise_tag_thre: untranscribed clips longer than this threshold will have text **noise** or **long_noise**
# --long_noise_tag_thre: untranscribed clips longer than this threshold will have text **long_noise**
# --merge_thre: for untranscribed clips longer than this threshold, merge the transcribed clips before and after it
# --max_clip_len: the maximum length of a clip (not segment) when merging transcribed clips
# --apply_overlap: allow overlap when applying the sliding windows.

case ${tag} in
    v1)
        # Basic config: Simply concat all clips, with time-stamps
        opts="--include_thre 100 --noise_tag_thre 1.0 --long_noise_tag_thre 30.0 --merge_thre 0.0 --max_clip_len 30.0"
        ;;
    v2)
        # Add the special <untranscribed> symbol when an untranscribed clip > 1s
        opts="--include_thre 0.01 --noise_tag_thre 1.0 --long_noise_tag_thre 30.0 --merge_thre 0.0 --max_clip_len 30.0"
        ;;
    v3)
        # Vanilla concat, without time-stamps
        opts="--include_thre 100 --noise_tag_thre 1.0 --long_noise_tag_thre 30.0 --merge_thre 0.0 --max_clip_len 30.0 --use_time_stamp false"
        ;;
    v4)
        # Based on v2, apply shift augment
        opts="--include_thre 0.01 --noise_tag_thre 1.0 --long_noise_tag_thre 30.0 --merge_thre 0.0 --max_clip_len 30.0 --apply_overlap true"
        ;;
    v5)
        # based on v3, but with special noise symbol.
        # We apply further cleanning for v3 and v5
        opts="--include_thre 0.01 --noise_tag_thre 1.0 --long_noise_tag_thre 30.0 --merge_thre 0.0 --max_clip_len 30.0 --use_time_stamp true"
        ;;
    esac

echo "working on tag ${tag} and options: ${opts}"

for dset in ${train_set} ${dev_set}; do
    python3 local/shift_aug.py ${opts} --data_dir data/${dset} --output_dir data/${dset}_${tag}
    bash utils/fix_data_dir.sh --utt_extra_files "text.ctc text.prev" data/${dset}_${tag}
done

# bash s2t.sh \
#     --stage 2 --stop_stage 4 \
#     --train_set ${train_set}_${tag} \
#     --valid_set ${dev_set}_${tag} \
#     --test_sets ${test_set} \
#     --audio_format "flac.ark" \
#     --nj 32

# bash utils/fix_data_dir.sh --utt_extra_files "text.ctc text.prev" dump/raw/${train_set}_${tag} 
# bash utils/fix_data_dir.sh --utt_extra_files "text.ctc text.prev" dump/raw/${dev_set}_${tag}  


