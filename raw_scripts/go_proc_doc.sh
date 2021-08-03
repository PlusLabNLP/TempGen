#!/bin/bash
# mkdir -p proc_output
set -eu
data_dir=../data/muc34
output_dir=${data_dir}/proc_output
train_output_path=${output_dir}/doc_train
dev_output_path=${output_dir}/doc_dev
test_output_path=${output_dir}/doc_test

if [ ! -d ${output_dir} ]; then
    mkdir ${output_dir}
fi

# train
cat ../data/muc34/TASK/CORPORA/dev/dev-*     | python proc_texts.py > ${train_output_path}

# dev
cat ../data/muc34/TASK/CORPORA/tst1/tst1-muc3 | python proc_texts.py > ${dev_output_path}
cat ../data/muc34/TASK/CORPORA/tst2/tst2-muc4 | python proc_texts.py >> ${dev_output_path}

# test
cat ../data/muc34/TASK/CORPORA/tst3/tst3-muc4 | python proc_texts.py > ${test_output_path}
cat ../data/muc34/TASK/CORPORA/tst4/tst4-muc4 | python proc_texts.py >> ${test_output_path}

