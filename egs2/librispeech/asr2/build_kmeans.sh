. ./path.sh
. ./cmd.sh

scripts/feats/perform_kmeans.sh \
    --stage 2 --stop-stage 2 \
    --train_set train_AF3 \
    --datadir dump/raw \
    --featdir dump/extract \
    --audio_format flac.ark \
    --feature_type qwen2audio \
    --km_dir exp/kmeans_init5 \
    --portion 0.03 \
    --nclusters 5000 \
    --storage_save_mode true \
    --use_gpu true \
    --nj 8 \
    --cpu_cmd "${train_cmd}" \
    --cuda_cmd "${cuda_cmd}" \
    --batch_bins 1