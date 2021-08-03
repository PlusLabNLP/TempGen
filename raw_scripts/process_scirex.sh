data_dir=../data/scirex
raw_dir=${data_dir}/release_data
output_dir=${data_dir}/proc_output
train_output_path=${output_dir}/train.json
dev_output_path=${output_dir}/dev.json
test_output_path=${output_dir}/test.json

if [ ! -d ${output_dir} ]; then
    mkdir ${output_dir}
fi


python process_scirex.py --input_path $raw_dir/train.jsonl --output_path $train_output_path --cardinality 2
python process_scirex.py --input_path $raw_dir/dev.jsonl --output_path $dev_output_path --cardinality 2
python process_scirex.py --input_path $raw_dir/test.jsonl --output_path $test_output_path --cardinality 2

python process_scirex.py --input_path $raw_dir/train.jsonl --output_path ${output_dir}/train_4ary.json --cardinality 4
python process_scirex.py --input_path $raw_dir/dev.jsonl --output_path ${output_dir}/dev_4ary.json --cardinality 4
python process_scirex.py --input_path $raw_dir/test.jsonl --output_path ${output_dir}/test_4ary.json --cardinality 4