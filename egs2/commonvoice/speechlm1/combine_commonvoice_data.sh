
log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

langs=(cy en de fr cy tt kab ca zh-TW it fa eu es ru tr nl eo zh-CN rw pt zh-HK cs pl uk)


dset_train_set="train_all"
dset_valid_set="dev_all"

src_train_sets=""
src_valid_sets=""
mkdir -p data/${dset_train_set}
mkdir -p data/${dset_train_dev}
log "Combine the training and valid sets"
rm data/${dset_train_set}/* data/${dset_train_dev}/*

for lang in "${langs[@]}"
do
    src_train_set=data/train_"$(echo "${lang}" | tr - _)"
    src_valid_set=data/dev_"$(echo "${lang}" | tr - _)"
    src_train_sets="${src_train_sets} ${src_train_set}"
    src_valid_sets="${src_valid_sets} ${src_valid_set}"
done

utils/combine_data.sh data/${dset_train_set} ${src_train_sets}