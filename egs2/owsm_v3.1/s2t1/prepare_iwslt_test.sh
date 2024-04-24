dataset=

. ./utils/parse_options.sh 

cat ${dataset}/text.lc.rm | awk '{print $1}' > ${dataset}/utt_list
cat ${dataset}/text.lc.rm | awk '{$1=""; print $0}' > ${dataset}/text_tmp

python3 NeMo-text-processing/nemo_text_processing/text_normalization/normalize.py \
  --input_file="${dataset}/text_tmp" \
  --output_file="${dataset}/text_normed" \
  --language=en

paste ${dataset}/utt_list ${dataset}/text_normed | tr '[:lower:]' '[:upper:]' | sed 's/TED_/ted_/g' > ${dataset}/text
