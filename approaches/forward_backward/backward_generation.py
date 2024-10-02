import pandas as pd
import numpy as np
from transformers import pipeline, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--start", help="start index", required=True, type=int)
parser.add_argument("--end", help="end index", required=True, type=int)
args = parser.parse_args()

start,end = args.start,args.end

# pipe = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf")

device = torch.device("cuda")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-chat-hf")
tokenizer.pad_token = tokenizer.eos_token

quantization_config = BitsAndBytesConfig(load_in_8bit=False, load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-chat-hf",
    quantization_config=quantization_config,
    # device_map=device_map,
    # trust_remote_code=args.trust_remote_code,
    # num_labels=1,
    # cache_dir="/dccstor/ojasgr/cache"
)

# model = model.to(device)
# start,end = 0,100

codex_prompt_path = "./codex_prompt.txt"
data_path = "./Updated_Test_Codex_dataset.xlsx"
data = pd.read_excel(data_path)

with open(codex_prompt_path,'r') as f:
    few_shots = f.read()


# end = min(end,len(data))
output_path = f"./llama2-70b-4Q/generation_outputs-llama2-70b-4Q_{start}-{end}.csv"

cols = ['instring','question','options','outputs','prediction','gold_answer']
output_df = pd.DataFrame(columns = cols)

batch_size = 1

for i in range(start,end):
    # elems = []
    print(f"Question {i}")
    # elems = []
    # quess = []
    # optionss = []
    # q_inputs = []
    # instrings = []
    # bs = (len(data) - i)%batch_size

    # pdb.set_trace()
    elem = data.iloc[i]
    # elems.append(elem)
    ques = elem['Questions']
    options = elem['Options']
    q_input = elem['Input']
    gold = elem['gold_option']
    instring = f"{few_shots}\n\n{q_input}"
    
    inputs = tokenizer(instring, return_tensors="pt")
    inputs = inputs.to(device)
    
    outputs = model.generate(inputs.input_ids, max_new_tokens = 1024)
    greedy_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    greedy_output = greedy_output.split(instring)[-1]
    gen_text = greedy_output.strip()

    # model_out = model.generate([instring])[0]

    output_delimiter = " answer is ("
    
    pred = ""
    pred_list = gen_text.split(output_delimiter)
    if len(pred_list) > 1:
        pred_str = pred_list[1]
        if pred_str:
            pred = pred_str[0]
    
    print("instring\n",instring,'\n')
    print("ques\n",ques,'\n')
    print("options\n",options,'\n')
    print("gen_text\n",gen_text,'\n')
    print("pred\n",pred,'\n')
    
    
#     print("ques\n")
#     print(q_input)
#     print("gen_text\n")
#     print(gen_text,'\n')
#     print("pred\n")
#     print(pred,'\n\n')

    output_df.loc[i] = [instring,ques,options,gen_text,pred,gold]

    # if i%20 == 19:
    output_df.to_csv(output_path,index = False)
    print(f"SAVED AT {i} steps")

print("COMPLETE")
