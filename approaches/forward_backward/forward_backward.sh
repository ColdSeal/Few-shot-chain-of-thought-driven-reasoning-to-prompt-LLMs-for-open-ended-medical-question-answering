# Generating sample forward outputs
model_name=llama2-7B-chat
qa_data_path=../../data/medqa/medqa_test_dataset.csv
# qa_data_path=/dccstor/ojasgr/data/real_life_questions.csv
prompt=kj_instruct_2
num_options=10
uniq_options=10
max_options=30
f_cot_path=./prompts/${prompt}.txt
b_cot_path=./prompts/codex0.txt
model_path=meta-llama/Llama-2-7b-chat
max_new_tokens=1024
output_dir=./outputs/real_questions/fb_${uniq_options}_opts/${prompt}/
temp=1.0

start=0
end_=500
mkdir -p ${output_dir}
cmd="python forward_backward.py --model_name ${model_name} --load_dir ${load_dir} --model_path ${model_path} \
--qa_data ${qa_data_path} --output_dir ${output_dir} --decoding ${decoding} --greedy_path ${greedy_path} \
--f_cot_path ${f_cot_path} --b_cot_path ${b_cot_path} --max_new_tokens ${max_new_tokens} --temp ${temp} --prompt ${prompt} --num_options ${num_options} \
--uniq_options ${uniq_options} --max_options ${max_options} --start ${start} --end ${end_} --sorting"

# $cmd
# jbsub -q x86_24h -mem 96G -cores 1x8+${num_gpus} -require a100_80gb ${cmd}
jbsub -q x86_24h -cores 1x8+0  -mem 96G ${cmd}

# echo $cmd >> ${output_dir}log.txt