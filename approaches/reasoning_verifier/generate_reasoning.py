# Generating sample forward outputs

import argparse
import random
import pdb
import pandas as pd
import torch
import math
import numpy as np
import ast
from transformers import pipeline, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

login(token = "<Enter your token>")

# random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", help="Model to evaluate", required=True, type=str)
parser.add_argument("--model_path", help="Path for Model to evaluate", required=True, type=str)
parser.add_argument("--qa_data", help="QA data", required=True, type=str)
parser.add_argument("--output_dir", help="output directory", required=False, type=str, default="./")
parser.add_argument("--load_dir", help="load previous results from this directory", required=False, type=str, default="./")
parser.add_argument("--decoding", help="decoding strategy : {sample,greedy}", required=False, type=str, default="sample")
parser.add_argument("--instruction", help="instruction to use", required=True, type=str)
parser.add_argument("--instruction_path", help="path to instruction to use", required=True, type=str)
parser.add_argument("--max_new_tokens", help="maximum new tokens to generate", required=False, type=int, default = 200)
parser.add_argument("--sorting", help="sorting options on the basis of log probabilities or not", required=False, action = 'store_false')
parser.add_argument("--num_options", help="number of options : number of samples to generate for each question", required=False, type=int, default = 4)
# parser.add_argument("--start", help="start from question number", required=False, type=int, default = 0)
# parser.add_argument("--end", help="end with question number", required=False, type=int, default = 50)

args = parser.parse_args()

print("LOADING MODEL...")
device = torch.device("cuda")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-70b-chat-hf")
tokenizer.pad_token = tokenizer.eos_token

quantization_config = BitsAndBytesConfig(load_in_8bit=False, load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-chat-hf",
    quantization_config=quantization_config,
)

model = model.to(device)
print("MODEL LOADED")

qa_df = pd.read_csv(args.qa_data)

qa = []
for i in range(len(qa_df)):
    qa.append({'q' : qa_df.loc[i,'Modified question line'], 'options' : qa_df.loc[i,'options'], 'gold' : qa_df.loc[i,'answer']})

columns = ["questions", "options", "gold answers"] + [f"instring{i}" for i in [1,2,3,4]] + [f"output{i}" for i in [1,2,3,4]]
iodf = pd.DataFrame(columns = columns)

with open(args.instruction_path) as f:
    inst = f.read()

for i,elem in zip([index for index in range(args.start,args.end)],qa[args.start:args.end]):

    print(f"\n------ QUESTION {i} ------")

    if str(elem["q"]).strip() == "":
        print("Empty Question")
        continue


    options = elem['options']
    options = ast.literal_eval(options)

    outputs = []
    instrings = []

    print(f"question :\n{elem['q']}\n")
    print(f"gold answer :\n{elem['gold']}\n")

    for opt_label in options:
        op = options[opt_label]

        instring = f"{inst}\n\nQuestion: {elem['q']}\nAnswer: {op}\nReasoning:"

        instrings.append(instring)

        print(f"instring :\n{instring}\n")

        model_out = model.generate([instring])[0]
        output = model_out.generated_text
        outputs.append(output)

        print(f"reasoning:\n{output}\n")

    iodf.loc[i] = [elem['q'],elem['options'],elem['gold']] + instrings + outputs

    if i%50 == 49:
        iodf.to_csv(f"{args.output_dir}all_options_reasoning_{args.instruction}_{args.start}-{args.end}.csv",index = False)

iodf.to_csv(f"{args.output_dir}all_options_reasoning_{args.instruction}_{args.start}-{args.end}.csv",index = False)
print("SAVED IO")
