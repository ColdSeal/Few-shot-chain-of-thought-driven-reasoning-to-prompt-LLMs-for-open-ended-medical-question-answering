import streamlit as st
import argparse
import random
import pdb
import pandas as pd
import torch
import math
import numpy as np
import re
from transformers import pipeline, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification

from huggingface_hub import login
hf_token = "<Enter your token>"
login(token = hf_token)

device = torch.device("cuda")

if 'submit' not in st.session_state:
    st.session_state.submit = False
if 'submit_backward' not in st.session_state:
    st.session_state.submit_backward = False

models = ['Llama-2-7B-chat','Llama-2-70B-chat']

model_name = st.selectbox("Choose the language model:", models)

@st.cache_resource
def load_models():

    if model_name == "Llama-2-7B-chat":
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        model = model.to(device)
        reward_model = AutoModelForSequenceClassification.from_pretrained('meta-llama/Llama-2-7b-chat-hf', num_labels = 1)
        reward_model.load_adapter("/dccstor/ojasgr/public_github/MedQA-No-Opt-Experiments/models/Reward_models/llama2-7B-chat-reward_model/reward_model/")
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
    elif model_name == 'Llama-2-70B-chat':
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-70b-chat-hf", device_map = 'auto')
        reward_model = AutoModelForSequenceClassification.from_pretrained('meta-llama/Llama-2-70b-chat-hf', num_labels = 1)
        reward_model.load_adapter("/dccstor/ojasgr/public_github/MedQA-No-Opt-Experiments/models/Reward_models/llama2-70B-chat-reward_model/reward_model/")
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-70b-chat-hf')

    reward_model.config.pad_token_id = reward_model.config.eos_token_id
    reward_model = reward_model.to(device)

    tokenizer.pad_token = tokenizer.eos_token
    
    return [model,reward_model,tokenizer]

reasoning_delimiter = "Answer: "
output_delimiter = "Q:"
instruction = "Use just the given patient history to answer the question. Do not assume any further information about the patient. Strictly Limit your response to 200 words."

model, reward_model, tokenizer = load_models()

def option_exists(new_op,old_ops):
    new_op = new_op.strip(". ").lower()
    for old_op in old_ops:
        old_op = old_op.strip(". ").lower()
        if (new_op in old_op or old_op in new_op):
            return(1)
    return(0)

@st.cache_resource
def create_options(instring, num_unique_ops = 4, options_generate_limit = 15):
    unique_options = []
    all_options = []
    op_to_reason = {}

    inputs = tokenizer(instring, return_tensors="pt")
    inputs = inputs.to(device)

    print("Creating Options")

    while(len(all_options) < options_generate_limit and len(unique_options) < num_unique_ops):

        if not len(all_options):
            outputs = model.generate(inputs.input_ids, max_new_tokens = 1024, repetition_penalty = 1.1)
        else:
            outputs = model.generate(inputs.input_ids, max_new_tokens = 1024, do_sample = True, temperature = 1, repetition_penalty = 1.1)

        text_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        text_output = text_output.split(instring)[-1]
        text_output = text_output.strip()

        if reasoning_delimiter in text_output:
            op_reasoning, op_text  = text_output.split(reasoning_delimiter)
            op_reasoning, op_text = op_reasoning.strip(), op_text.strip().split("\n")[0].strip()

            st.write("Option Created :",op_text)

            if not option_exists(op_text,unique_options):

                st.write("New Option Created : Accepted")
                unique_options.append(op_text)
                op_to_reason[op_text] = op_reasoning
            else:
                st.write("Option already exists.. Discarding..")
            
        else:
            op_text = "<parsing error>"
            st.write(op_text)

        all_options.append(op_text)

    st.write("Final Options :", unique_options)

    return [unique_options,op_to_reason]

st.title("Choose the algorithm you want to run :")
algorithms = ["MedCodex - Greedy", "Codex - Greedy", "MedCodex + Codex (F+B)", "MedCodex + Verifier (F + RM)"]
algo = st.selectbox("Choose an option:", algorithms)

st.write("You selected:", algo)

user_input = st.text_area("Enter input prompt", value="", height=50, max_chars=None)

submit = st.button("Submit input")

if submit:
    st.session_state.submit = True

output = ""

if algo == "MedCodex - Greedy" or algo == "Codex - Greedy":
    if algo == "MedCodex - Greedy":
        with open("./prompts/kj_instruct_2.txt", 'r') as f:
            few_shot_cot = f.read()

        instruction = "Use just the given patient history to answer the question. Do not assume any further information about the patient. Strictly Limit your response to 200 words."
        instring = f'''{few_shot_cot}\n\n{instruction}\nQ: {user_input}\nA: Let's think step-by-step.'''

    elif algo == "Codex - Greedy":
        with open("./prompts/codex_2.txt", 'r') as f:
            few_shot_cot = f.read()

        instring = f'''{few_shot_cot}\n\nQ: {user_input}\nA: Let's think step-by-step.'''

    if submit:

        inputs = tokenizer(instring, return_tensors="pt")
        inputs = inputs.to(device)

        greedy_out = model.generate(inputs.input_ids, max_new_tokens = 1024, repetition_penalty = 1.1)
        greedy_out = tokenizer.batch_decode(greedy_out, skip_special_tokens=True)[0]
        greedy_out = greedy_out.split(instring)[-1]
        greedy_output = greedy_out.strip()

        if reasoning_delimiter in greedy_output:
            greedy_output = greedy_output.replace(reasoning_delimiter,f"\n\n{reasoning_delimiter}")
        else:
            greedy_ans = "<parsing error>"

        output = greedy_output

elif algo == "MedCodex + Codex (F+B)" or algo == "MedCodex + Verifier (F + RM)":

    with open("./prompts/kj_instruct_2.txt", 'r') as f:
        few_shot_cot = f.read()

    with open("./prompts/codex_2.txt", 'r') as f:
        backward_prompt = f.read()

    instruction = "Use just the given patient history to answer the question. Do not assume any further information about the patient. Strictly Limit your response to 200 words."
    
    if submit or st.session_state.submit:
        f_instring = f'''{few_shot_cot}\n\n{instruction}\nQ: {user_input}\nA: Let's think step-by-step.'''
        uniq_options, op_to_reason = create_options(f_instring)
        
        if (len(uniq_options) < 4):
            st.write("Not enough Options")

        else:
            if algo == "MedCodex + Codex (F+B)":

                options_text = ""
                for op_num,op in enumerate(uniq_options):
                    options_text += f"({chr(ord('A') + op_num)}) {op} "

                options_text = options_text.strip() + '\n'

                b_instring = f"{backward_prompt}\n\nQ: {user_input}\n{options_text}A: Let's think step-by-step."

                b_inputs = tokenizer(b_instring, return_tensors="pt")
                b_inputs = b_inputs.to(device)

                b_output = model.generate(b_inputs.input_ids, max_new_tokens = 1024, repetition_penalty = 1.1)
                b_output = tokenizer.batch_decode(b_output, skip_special_tokens=True)[0]
                b_output = b_output.split(b_instring)[-1]
                b_output = b_output.strip()

                if b_output.strip() == "":
                    st.write("output empty")
                    b_output = "<empty>"

                b_answer = re.findall(r"\([A-D]\)",b_output)
                if len(b_answer):
                    b_answer = b_answer[0]
                    option = ord(b_answer[1]) - ord('A') + 1
                    b_answer_idx = ord(b_answer[1]) - ord('A')
                    output = b_answer + " " + op_to_reason[uniq_options[b_answer_idx]] + "\nAnswer: " + uniq_options[b_answer_idx]
                else:
                    b_answer ="<parsing error>" 
                    option = 0
                    output = "<parsing error>"

            elif algo == "MedCodex + Verifier (F + RM)":

                instrings = []
                logits = []

                for k in range(4):
                    instring = f"Question : {user_input}\nReasoning : Let's think step by step. {op_to_reason[uniq_options[k]]}\nAnswer : {uniq_options[k]}"

                inputs = tokenizer(instring, padding = True, return_tensors="pt")

                inputs = inputs.to(device)
                with torch.no_grad():
                    outputs = reward_model(**inputs)

                logits = outputs.logits
                logits = logits.reshape(-1)
                logits = logits.tolist()

                answer_idx = np.argmax(logits)
                answer = f"({chr(answer_idx + ord('A'))})"

                output = "Option " + answer + "\n" + op_to_reason[uniq_options[answer_idx]] + "\nAnswer: " + uniq_options[answer_idx]

st.write("""
    <div style='border: 1px solid black; padding: 10px;'>
        <h3>Your desired output is as follows:</h3>
        <blockquote>{}</blockquote>
    </div>
    """.format(output), unsafe_allow_html=True)