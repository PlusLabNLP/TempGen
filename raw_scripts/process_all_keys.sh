#!/bin/bash
# mkdir -p proc_output

set -eu

data_dir=../data/muc34
output_dir=${data_dir}/proc_output
train_output_path=${output_dir}/train.json
dev_output_path=${output_dir}/dev.json
test_output_path=${output_dir}/test.json

if [ ! -d ${output_dir} ]; then
    mkdir ${output_dir}
fi


python process_all_keys.py  --input_corpus ${output_dir}/doc_train --input_pattern ../data/muc34/TASK/CORPORA/dev/key-dev- --output_path ${train_output_path}
python process_all_keys.py  --input_corpus ${output_dir}/doc_dev  --input_pattern ../data/muc34/TASK/CORPORA/tst[12]/key-tst --output_path ${dev_output_path}
python process_all_keys.py  --input_corpus ${output_dir}/doc_test  --input_pattern ../data/muc34/TASK/CORPORA/tst[34]/key-tst --output_path ${test_output_path}
