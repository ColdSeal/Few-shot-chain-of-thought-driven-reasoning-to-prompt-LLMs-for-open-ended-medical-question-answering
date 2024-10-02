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

device = torch.device("cuda")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-chat-hf")
tokenizer.pad_token = tokenizer.eos_token

quantization_config = BitsAndBytesConfig(load_in_8bit=False, load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-chat-hf",
    quantization_config=quantization_config,
)

model = model.to(device)

codex_prompt_path = "./codex_prompt.txt"
data_path = "./Updated_Test_Codex_dataset.xlsx"
data = pd.read_excel(data_path)

with open(codex_prompt_path,'r') as f:
    few_shots = f.read()

output_path = f"./llama2-70b-4Q/generation_outputs-llama2-70b-4Q_{start}-{end}.csv"

cols = ['instring','question','options','outputs','prediction','gold_answer']
output_df = pd.DataFrame(columns = cols)

batch_size = 1

for i in range(start,end):
    print(f"Question {i}")
    elem = data.iloc[i]
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
    output_df.loc[i] = [instring,ques,options,gen_text,pred,gold]
    output_df.to_csv(output_path,index = False)
    print(f"SAVED AT {i} steps")
print("COMPLETE")
