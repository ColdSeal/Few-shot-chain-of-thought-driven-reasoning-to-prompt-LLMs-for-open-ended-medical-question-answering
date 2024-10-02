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

from genai.credentials import Credentials
from genai.model import Model
from genai.schemas import GenerateParams
from genai.extensions.langchain import LangChainInterface

from huggingface_hub import login
login(token = "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")

api_key = "pak-zLIXfwz3UmF2EIOrXr-T4YNw8s3A0MUCZ-5pmFezfFY"
api_url = "https://bam-api.res.ibm.com/v1"
creds = Credentials(api_key,api_url)

device = torch.device("cuda")

if 'submit' not in st.session_state:
    st.session_state.submit = False
if 'submit_backward' not in st.session_state:
    st.session_state.submit_backward = False

def reset_session():
    # Clear all session state variables
    # st.session_state.clear()
    # Inject JavaScript to simulate a page reload
    # st.write('<script>location.reload()</script>', unsafe_allow_html=True)
    st.experimental_rerun()

    
@st.cache_resource
def load_models():
    params = GenerateParams(decoding_method="sample", max_new_tokens=1024, temperature = 1, top_p = 1, top_k = 50, typical_p = 1, stop_sequences=["\nQ: ","Use just the given patient history to answer the question."], return_options={"generated_tokens" : True, "token_logprobs" : True})
    greedy_params = GenerateParams(decoding_method="greedy", max_new_tokens=1024, repetition_penalty=1.1, stop_sequences=["\nQ: ","Use just the given patient history to answer the question."], return_options={"generated_tokens" : True, "token_logprobs" : True})
    # Instantiate a model proxy object to send your requests
    model = Model("meta-llama/Llama-2-7b-chat", params=params, credentials=creds)
    greedy_model = Model("meta-llama/Llama-2-7b-chat", params=greedy_params, credentials=creds)

    reward_model = AutoModelForSequenceClassification.from_pretrained('meta-llama/Llama-2-7b-chat-hf', num_labels = 1)
    reward_model.load_adapter(f"/dccstor/ojasgr/scripts/approaches/reasoning_verifier/trl/examples/scripts/output_old/7b-chat/checkpoint-1900/")
    reward_model.config.pad_token_id = reward_model.config.eos_token_id
    reward_model = reward_model.to(device)

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
    tokenizer.pad_token = tokenizer.eos_token
    
    return [greedy_model,model,reward_model,tokenizer]

reasoning_delimiter = "Answer: "
output_delimiter = "Q:"
instruction = "Use just the given patient history to answer the question. Do not assume any further information about the patient. Strictly Limit your response to 200 words."

greedy_model, model, reward_model, tokenizer = load_models()

def option_exists(new_op,old_ops):
    new_op = new_op.strip(". ").lower()
    for old_op in old_ops:
        old_op = old_op.strip(". ").lower()
        # print(f"new:{new_op}")
        # print(f"old:{old_op}")
        if (new_op in old_op or old_op in new_op):
            return(1)
    return(0)

@st.cache_resource
def create_options(instring, num_unique_ops = 4, options_generate_limit = 15):
    unique_options = []
    all_options = []
    op_to_reason = {}

    # inputs = tokenizer(instring, return_tensors="pt")
    # inputs = inputs.to(device)

    print("Creating Options")

    while(len(all_options) < options_generate_limit and len(unique_options) < num_unique_ops):

        if not len(all_options):
            outputs = greedy_model.generate([instring])[0]
            # outputs = model.generate(inputs.input_ids, max_new_tokens = 1024, repetition_penalty = 1.1)
        else:
            outputs = model.generate([instring])[0]
            # outputs = model.generate(inputs.input_ids, max_new_tokens = 1024, do_sample = True, temperature = 1, repetition_penalty = 1.1)

        text_output = outputs.generated_text.strip()
        # text_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        # text_output = text_output.split(instring)[-1]
        # text_output = text_output.strip()

        if reasoning_delimiter in text_output:
            op_reasoning, op_text  = text_output.split(reasoning_delimiter)
            op_reasoning, op_text = op_reasoning.strip(), op_text.strip().split("\n")[0].strip()

            # print("Option Created :",op_text)
            st.write("Option Created :",op_text)

            if not option_exists(op_text,unique_options):
                # print("New Option :", op_text)
                st.write("New Option Created : Accepted")
                unique_options.append(op_text)
                op_to_reason[op_text] = op_reasoning
            else:
                # print("Option already exists.. Discarding..")
                st.write("Option already exists.. Discarding..")
            
        else:
            op_text = "<parsing error>"
            # print(op_text)
            st.write(op_text)

        all_options.append(op_text)
        # print("Existing Options :", unique_options, "\n")
        # st.write("Existing Options :", unique_options, "\n")

    # print("Final Options :", unique_options)
    st.write("Final Options :", unique_options)

    return [unique_options,op_to_reason]

# st.write("""
# # DEMO for running multiple prompt interfaces
# """)

st.title("Choose the algorithm you want to run :")
# option_ind = int(input("choose an algorithm to run"))
algorithms = ["MedCodex - Greedy", "Codex - Greedy", "MedCodex + Codex (F+B)", "MedCodex + Verifier (F + RM)"]
# algo = algorithms[option_ind]
# Drop-down menu with four options
algo = st.selectbox("Choose an option:", ["MedCodex - Greedy", "Codex - Greedy", "MedCodex + Codex (F+B)", "MedCodex + Verifier (F + RM)"])

# Display the selected optio
# st.write("You selected:", option)
# print("You selected:", algo)
st.write("You selected:", algo)

# user_input = st.text_input("Enter input prompt")
user_input = st.text_area("Enter input prompt", value="", height=50, max_chars=None)
# user_input = input("User input here : ")
# print("The input is :\n",user_input)
# st.write("The input is :\n",user_input)

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

        # submit = True

    if submit:

        # st.write("The input is :\n",user_input)

        # inputs = tokenizer(instring, return_tensors="pt")
        # inputs = inputs.to(device)

        greedy_out = greedy_model.generate([instring])[0]
        greedy_output = greedy_out.generated_text.strip()
        # greedy_out = model.generate(inputs.input_ids, max_new_tokens = 1024, repetition_penalty = 1.1)
        # greedy_out = greedy_out.split(instring)[-1]
        # greedy_output = greedy_out.strip()

        if reasoning_delimiter in greedy_output:
            greedy_output = greedy_output.replace(reasoning_delimiter,f"\n\n{reasoning_delimiter}")
            # greedy_ans = greedy_output.split(reasoning_delimiter)[1].strip()
        else:
            greedy_ans = "<parsing error>"

        output = greedy_output
        # output = "F output"

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
            # output = "Not enough options generated!"
            st.write("Not enough Options")
            # print("Not enough Options")

        else:
            if algo == "MedCodex + Codex (F+B)":

                options_text = ""
                for op_num,op in enumerate(uniq_options):
                    options_text += f"({chr(ord('A') + op_num)}) {op} "

                options_text = options_text.strip() + '\n'

                # backward_ques = input("Input backward Question : ")
                backward_ques = st.text_area("Enter backward Question", value="", height=50, max_chars=None)

                submit_backward = st.button("Submit backward input")

                if submit_backward or st.session_state.submit_backward:
                    
                    st.session_state.submit_backward = True

                    b_instring = f"{backward_prompt}\n\nQ: {backward_ques}\n{options_text}A: Let's think step-by-step."

                    # b_inputs = tokenizer(b_instring, return_tensors="pt")
                    # b_inputs = b_inputs.to(device)

                    b_out = greedy_model.generate([b_instring])[0]
                    b_output = b_out.generated_text.strip()

                    # b_output = model.generate(b_inputs.input_ids, max_new_tokens = 1024, repetition_penalty = 1.1)
                    # b_output = b_output.split(b_instring)[-1]
                    # b_output = b_output.strip()

                    if b_output.strip() == "":
                        st.write("output empty")
                        # print("output empty")
                        b_output = "<empty>"

                    # if str(elem["a"]).strip() == "":
                    #     elem["a"] = "<empty>"

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

                # logit = outputs.logits.item()
                # logits.append(logit)

                logits = outputs.logits
                logits = logits.reshape(-1)
                logits = logits.tolist()

                # print('logit', chr(ord('A') + k-1), f"model : {logit:.3f}")

                answer_idx = np.argmax(logits)
                answer = f"({chr(answer_idx + ord('A'))})"

                output = "Option " + answer + "\n" + op_to_reason[uniq_options[answer_idx]] + "\nAnswer: " + uniq_options[answer_idx]

# st.write("Your desired output is as follows :\n", output)
st.write("""
    <div style='border: 1px solid black; padding: 10px;'>
        <h3>Your desired output is as follows:</h3>
        <blockquote>{}</blockquote>
    </div>
    """.format(output), unsafe_allow_html=True)
# print("Your desired output is as follows :\n", output)


# Button to reset the session
if st.button("Reset Session"):
    user_input = ""
    reset_session()

# clear = st.button("Clear")
# if clear:
#     st.session_state.submit_backward = False
#     st.session_state.submit = False