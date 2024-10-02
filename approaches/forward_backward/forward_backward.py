# Generating sample forward outputs

import argparse
import random
import pdb
import pandas as pd
import torch
import math
import numpy as np
import re
from transformers import pipeline, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
# random.seed(42)
from huggingface_hub import login
login(token = "<Enter your token>")
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", help="Model to evaluate", required=True, type=str)
parser.add_argument("--model_path", help="Path for Model to evaluate", required=True, type=str)
parser.add_argument("--qa_data", help="QA data", required=True, type=str)
parser.add_argument("--f_cot_path", help="forward cot path", required=True, type=str)
parser.add_argument("--b_cot_path", help="backward cot path", required=True, type=str)
parser.add_argument("--output_dir", help="output directory", required=False, type=str, default="./")
parser.add_argument("--load_dir", help="load previous results from this directory", required=False, type=str, default="./")
parser.add_argument("--greedy_path", help="greedy output file", required=True, type=str)
parser.add_argument("--decoding", help="decoding strategy : {sample,greedy}", required=False, type=str, default="sample")
parser.add_argument("--max_new_tokens", help="maximum new tokens to generate", required=False, type=int, default = 200)
parser.add_argument("--prompt", help="type of prompt", required=True, type=str)
parser.add_argument("--sorting", help="sorting options on the basis of log probabilities or not", required=False, action = 'store_false')
parser.add_argument("--temp", help="temperature for sampling", required=False, type=float, default = 1.0)
parser.add_argument("--num_options", help="number of options : number of samples to generate for each question", required=False, type=int, default = 4)
parser.add_argument("--uniq_options", help="number of options : number of uniq sample options to generate for each question", required=False, type=int, default = 4)
parser.add_argument("--max_options", help="number of options : number of maximum sample options to generate for each question", required=False, type=int, default = 4)
parser.add_argument("--start", help="start from question number", required=False, type=int, default = 0)
parser.add_argument("--end", help="end with question number", required=False, type=int, default = 50)
args = parser.parse_args()

device = torch.device("cuda")

print("LOADING MODEL...")

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
    qa.append({'q' : qa_df.loc[i,'Descriptive question'], 'a' : qa_df.loc[i,'Correct answer text']})

columns = ["inputs","questions", "gold answers", "full output", "predictions"] + ["greedy_option"] + [f"op{i+1}" for i in range(args.max_options)] + ["greedy_output"] + [f"out{i+1}" for i in range(args.max_options)] + ["log_probs"] + ["backward_input","backward_options","backward_answer","backward_output","backward_pred_text"]
iodf = pd.DataFrame(columns = columns)

with open(args.f_cot_path, 'r') as f:
    few_shot_cot = f.read()

with open(args.b_cot_path, 'r') as f:
    backward_prompt = f.read()

def create_options_all(ans_probs, instring):
    
    uniq_opts = []
    options = []
    enough_opts = False
    num_opt = 0

    inputs = tokenizer(instring, return_tensors="pt")
    inputs = inputs.to(device)

    while not enough_opts:
        # Generate tokens with logits
        model_out = model.generate(
            inputs.input_ids, 
            max_new_tokens=1024, 
            repetition_penalty=1.1, 
            do_sample=True, 
            temperature=1, 
            output_scores=True,  # To get logits
            return_dict_in_generate=True  # To get all details in the output dict
        )
        
        # Get generated sequences (tokens) and scores (logits)
        generated_tokens = model_out.sequences
        logits_per_step = model_out.scores  # List of logits for each step

        # Decode the generated tokens into text
        decoded_output = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        
        # Get the generated text after the prompt (remove input part)
        f_output = decoded_output.split(instring)[-1].strip()
        
        # Compute log probabilities for each token
        log_probs = []
        for step_logits, token_id in zip(logits_per_step, generated_tokens[0][inputs.input_ids.shape[-1]:]):
            token_log_probs = torch.log_softmax(step_logits, dim=-1)
            log_probs.append(token_log_probs[0, token_id].item())
        
        # Compute the average log probability for the generated text
        avg_log_prob = sum(log_probs) / len(log_probs) if log_probs else -math.inf

        # Extract the option based on the reasoning delimiter
        if reasoning_delimiter in f_output:
            option = f_output.split(reasoning_delimiter)[1].strip().split("\n")[0]
        else:
            option = "<parsing error>"
            avg_log_prob = -math.inf

        # Append the result
        ans_probs.append([avg_log_prob, option, f_output])

        num_opt += 1
        print(f"Prediction {num_opt}:\n{f_output}\n")
        print(f"Option: {option}")

        if len(uniq_opts) < args.uniq_options:
            if option != "<parsing error>":
                if len(uniq_opts):
                    option_stripped = option.strip(". ").lower()
                    present = False
                    for opt in uniq_opts:
                        opt_stripped = opt.strip(". ").lower()
                        if (opt_stripped in option_stripped or option_stripped in opt_stripped):
                            present = True
                            break
                    if not present:
                        uniq_opts.append(option)
                else:
                    uniq_opts.append(option)
            print(f"unique options : {uniq_opts}\n")

        elif num_opt >= args.num_options:
            print(f"Completed {num_opt} possible options")
            print(f"unique options: {uniq_opts}\n")
            break

        if num_opt == args.max_options:
            print("Reached Max possible options")
            print(f"unique options: {uniq_opts}\n")
            break

    return ans_probs

def option_exists(new_op,old_ops):
    new_op = new_op.strip(". ").lower()
    for old_op in old_ops:
        old_op = old_op.strip(". ").lower()
        if (new_op in old_op or old_op in new_op):
            return(1)
    return(0)

def make_options_strict(forward_options,backward_opts):
    options_text = ""
    options_list = []
    op_num = 0
    
    print("[Randomising Options - Strict]")
    ind = 1
    for opt in forward_options:
        if type(opt) == float:
            break
        ind += 1

    forward_options = forward_options[:ind]

    for op in forward_options:
        op = str(op).strip()
        op = op.strip(". ")
        if op != "<parsing error>":
            if option_exists(op,options_list):
                continue
            options_list.append(op)
            options_text += f"({chr(ord('A') + op_num)}) {op} "
            op_num += 1

        if op_num == backward_opts:
            break

    return [options_list,options_text]

def make_options(forward_options,backward_opts):
    options_text = ""
    options_list = []
    op_num = 0

    print("[Randomising Options]")
    ind = 1
    for opt in forward_options:
        if type(opt) == float:
            break
        ind += 1
    forward_options = forward_options[:ind]

    for op in forward_options:
        op = str(op).strip()
        op = op.strip(". ")
        if op != "<parsing error>":
            options_list.append(op)
            options_text += f"({chr(ord('A') + op_num)}) {op} "
            op_num += 1

        if op_num == backward_opts:
            break

    return [options_list,options_text]

reasoning_delimiter = "Answer: "
output_delimiter = "Q:"
instruction = "Use just the given patient history to answer the question. Do not assume any further information about the patient. Strictly Limit your response to 200 words."
option_delimiter = "<op>"

for i,elem in zip([index for index in range(args.start,args.end)],qa[args.start:args.end]):
    print(f"\n------ QUESTION {i} ------")

    # FORWARD

    if str(elem['q']) == "nan":
        print(f"{i}-empty")
        continue
    
    f_instring = f'''{few_shot_cot}\n\n{instruction}\nQ: {elem['q']}\nA: Let's think step-by-step.'''
    print(f"question :\n{elem['q']}\n")
    print(f"gold :\n{elem['a']}\n")
 
    log_probs = []
    ans_probs = []

    ans_probs = create_options_all(ans_probs, f_instring)

    f_inputs = tokenizer(f_instring, return_tensors="pt")
    f_inputs = f_inputs.to(device)

    greedy_out = model.generate(f_inputs.input_ids, max_new_tokens = 1024, repetition_penalty = 1.1, output_scores=True, return_dict_in_generate=True)
    generated_tokens = greedy_out.sequences
    logits_per_step = greedy_out.scores
    greedy_out = tokenizer.decode(generated_tokens[0], skip_special_tokens = True)
    greedy_out = greedy_out.split(f_instring)[-1]

    greedy_log_probs = []
    for step_logits, token_id in zip(logits_per_step, generated_tokens[0][f_inputs.input_ids.shape[-1]:]):
        token_log_probs = torch.log_softmax(step_logits, dim=-1)
        greedy_log_probs.append(token_log_probs[0, token_id].item())
    greedy_log_prob = sum(greedy_log_probs) / len(greedy_log_probs) if greedy_log_probs else -math.inf
    if reasoning_delimiter in greedy_out:
        greedy_ans = greedy_out.split(reasoning_delimiter)[1].strip()
        # greedy_answer = greedy_answer.strip("\n")
    else:
        greedy_ans = "<parsing error>"
        greedy_log_prob = -math.inf

    print(f"greedy prediction:\n{greedy_out}\n")
    print(f"greedy answer:\n{greedy_ans}\n")
    print(f"log prob:\n{greedy_log_prob}\n")

    # For sorting on the basis of log probabilities
    if args.sorting:
        ans_probs = np.array(ans_probs)
        ans_probs = ans_probs[ans_probs[:,0].argsort()]
        ans_probs = ans_probs.tolist()

    ans_probs = [[greedy_log_prob, greedy_ans, greedy_out]] + ans_probs
    ans_probs = np.array(ans_probs)
        
    f_answers = list(ans_probs[:,1])
    f_outputs = list(ans_probs[:,2])

    f_answers += ["" for i in range(args.max_options + 1 - len(f_answers))]
    f_outputs += ["" for i in range(args.max_options + 1 - len(f_outputs))]

    pred_options = '\n'.join(f_answers).strip()
    full_output = '\n'.join(f_outputs).strip()
    
    log_probs = list(ans_probs[:,0])
    log_probs = [f"{float(elem):.2f}" for elem in log_probs]
    log_probs = ','.join(log_probs)

    print(f"log probabilities :\n{log_probs}\n")
    print(f"All answers :\n{'|'.join(f_answers).strip()}")

    # BACKWARD

    options_list, options_text = make_options_strict(f_answers,args.uniq_options)

    if len(options_list) < args.uniq_options:
        options_list, options_text = make_options(f_answers,args.uniq_options)

    if len(options_list) < args.uniq_options:
        print("NOT ENOUGH OPTIONS")
        b_instring,b_options,b_answer,b_output,pred_text = ["<parsing error>"]*5
        iodf.loc[i] = [f_instring,elem['q'],elem['a'],full_output,pred_options] + f_answers + f_outputs + [log_probs] + [b_instring,b_options,b_answer,b_output,pred_text]
        continue

    options_text = options_text.strip() + '\n'

    b_instring = f"{backward_prompt}\n\nQ: {elem['q']}\n{options_text}A: Let's think step-by-step."
    b_options = ""

    for k,op in enumerate(options_list):
        b_options += f"({chr(ord('A') + k)}) {op}\n"

    b_options = b_options.strip()
    
    print(f"question :\n{elem['q']}\n")
    print(f"options :\n{b_options}\n")
    print(f"input :\n{b_instring}\n")

    try:
        b_instring_inputs = tokenizer(b_instring, return_tensors = "pt")
        b_instring_inputs = b_instring_inputs.to(device)
        b_output = model.generate(b_instring.input_ids, max_new_tokens = 1024, repetition_penalty = 1.1)
        b_output = tokenizer.batch_decode(b_output, skip_special_tokens = True)[0]
        # b_output = b_output.split(b_instring)[-1].strip()
    except Exception as error:
        print("An exception occurred :",error)
        b_output = ""

    b_output = b_output.split(output_delimiter)[0].strip()

    if b_output.strip() == "":
        print("output empty")
        b_output = "<empty>"
    
    if str(elem["a"]).strip() == "":
        elem["a"] = "<empty>"
    
    b_answer = re.findall(r"\([A-D]\)",b_output)
    if len(b_answer):
        b_answer = b_answer[0]
        option = ord(b_answer[1]) - ord('A') + 1
    else:
        b_answer ="<parsing error>" 
        option = 0

    print(f"backward output :\n{b_output}\n")

    if option:
        pred_text = options_list[option - 1]
    else:
        pred_text = "<parsing error>"
        
    print(f"backward answer text :\n{pred_text}\n")

    iodf.loc[i] = [f_instring,elem['q'],elem['a'],full_output,pred_options] + f_answers + f_outputs + [log_probs] + [b_instring,b_options,b_answer,b_output,pred_text]

    if i%50 == 49:
        iodf.to_csv(f"llama2-7b-chat-outputs/medqa_test_{args.uniq_options}-{args.num_options}_sampled_{args.start}-{args.end}.csv", index = False)
 
        print("RESULT CHECKPOINT SAVED")

   
iodf.to_csv(f"llama2-7b-chat-outputs/medqa_test_{args.uniq_options}-{args.num_options}_sampled_{args.start}-{args.end}.csv", index = False)
print("SAVED IO")
