# Generating sample forward outputs

import argparse
import random
import pdb
import json
import pandas as pd
import torch
import re
import pdb
from tqdm import tqdm

from genai.credentials import Credentials
from genai.model import Model
from genai.schemas import GenerateParams
from genai.extensions.langchain import LangChainInterface

from huggingface_hub import login

login(token = "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")

device = torch.device("cuda")

print("LOADING MODEL...")

api_key = "pak-zLIXfwz3UmF2EIOrXr-T4YNw8s3A0MUCZ-5pmFezfFY"
api_url = "https://bam-api.res.ibm.com/v1"
creds = Credentials(api_key,api_url)

# Instantiate parameters for text generation
params = GenerateParams(decoding_method="greedy", max_new_tokens=1024, repetition_penalty=1.1)
# Instantiate a model proxy object to send your requests
model = Model("meta-llama/llama-2-7b-chat", params=params, credentials=creds)
# model = LangChainInterface(model="meta-llama/llama-2-7b", params=params, credentials=creds)

codex_prompt_path = "./codex_prompt.txt"
data_path = "./Updated_Test_Codex_dataset.xlsx"
output_path = "./generation_outputs.csv"

with open(codex_prompt_path,'r') as f:
    few_shots = f.read()

data = pd.read_excel(data_path)

cols = ['instring','question','options','outputs','prediction','gold_option']
output_df = pd.DataFrame(columns = cols)

# output_df = pd.read_csv("generation_outputs.csv")

batch_size = 4

for i in tqdm(range(len(data)),unit = "Question"):

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

    model_out = model.generate([instring])[0]
    
    gen_text = model_out.generated_text

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

    if not i%49:
        output_df.to_csv(output_path,index = False)
        print(f"SAVED AT {i} steps\n")

output_df.to_csv(output_path,index = False)
print("COMPLETE")
