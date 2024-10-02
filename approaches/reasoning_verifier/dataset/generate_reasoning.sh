model_name=llama2-7B-chat
load_dir=./outputs_250ques/
model_path=meta-llama/Llama-2-7b-chat
qa_data_path=../../data/medqa/medqa_test_dataset.csv
output_dir=./outputs/real_questions/
decoding=greedy
cot_path=./kj_instruct_2.txt
max_new_tokens=1024
mkdir -p ${output_dir}

cmd="python generate_reasoning.py --model_name ${model_name} --load_dir ${load_dir} \
--model_path ${model_path} --qa_data ${qa_data_path} --output_dir ${output_dir} --decoding ${decoding} \
--cot_path ${cot_path} --max_new_tokens ${max_new_tokens}"

$cmd