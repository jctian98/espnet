train_dir=data/train_m
valid_dir=data/dev
test_dir=data/test

# (1) splice data. For dev and test, force continuous and long. That means all these samples don't have gap > 1s and are in lengths of 10-30s.
python3 local/prepare_long_dataset.py --input_dir ${train_dir} --output_dir ${train_dir}_long --force_long_and_continuous false
./utils/fix_data_dir.sh ${train_dir}_long
python3 local/prepare_long_dataset.py --input_dir ${valid_dir} --output_dir ${valid_dir}_long --force_long_and_continuous true
./utils/fix_data_dir.sh ${valid_dir}_long
python3 local/prepare_long_dataset.py --input_dir ${test_dir} --output_dir ${test_dir}_long --force_long_and_continuous true
./utils/fix_data_dir.sh ${test_dir}_long


# (2) Additionally, we will remove the <sep> in ${train_dir}_long to build another set ${train_dir}_long_nosep.
# this will be used with standard CTC
for dset in ${train_dir} ${valid_dir} ${test_dir}; do
    cp -r ${dset}_long ${dset}_long_nosep;
    cat ${dset}_long/text | sed 's/<sep>//g' > ${dset}_long_nosep/text
    ./utils/fix_data_dir.sh ${dset}_long_nosep
done
