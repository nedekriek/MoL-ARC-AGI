import time
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import datetime
from tqdm import tqdm
import json
import torch
import numpy as np
import os
# Import the standard modules we need.

from induction_runner import inductor
from prompt_creator import prompt_creator
from dataloader import problem, load_data
# Import the inductor and prompt_creator from the previous snippets.


# Load the configuration file
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

##################################
# The model directory
##################################

BASE_MODEL = config["BASE_MODEL"]
LORA_DIR = config["LORA_DIR"]

##################################
# Configs
##################################


num_of_samples_per_problem = config["num_of_samples_per_problem"]
# Here the num_of_samples_per_problem is the total number of program samples we want for a simple problem.
TENSOR_PARALLEL = config["TENSOR_PARALLEL"]
mode = config["MODE"]
prompt_engineering = config["prompt_engineering"]

##################################
# Load the tokenizer
##################################

if LORA_DIR:
    tokenizer = AutoTokenizer.from_pretrained(LORA_DIR)
else:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

#################################################
# Set the device and load the model as class LLM in vllm
#################################################

device = "cuda" if torch.cuda.is_available() else "cpu"

if LORA_DIR:
    llm = LLM(model=BASE_MODEL, enable_lora=True, max_lora_rank=256, max_model_len=12000,
            enable_prefix_caching=True, tensor_parallel_size=TENSOR_PARALLEL)
    lora_request=LoRARequest("barc_adapter", 1, LORA_DIR)
else:
    llm = LLM(model=BASE_MODEL, enable_lora=False, max_model_len=12000,
            enable_prefix_caching=True, tensor_parallel_size=TENSOR_PARALLEL, device = device)
    

#################################################
# Date directory
#################################################

folderpath = 'ARC-AGI/data/' + mode
    
#################################################
# Induction
#################################################



def main():
    # Create an inductor object
    inductor_obj = inductor(llm, tokenizer, device)
    
    # Create a prompt creator object
    prompt_creator_obj = prompt_creator(device = device, mode = mode)

    # Load the data
    problems = load_data(mode)

    # Create prompt
    for problem in problems:
        prompt = prompt_creator_obj.create_prompt_from_test_task(problem)
        if prompt_engineering:
            prompt = prompt_creator_obj.add_prompt_engineering(prompt)
        problem.give_prompt(prompt)
    
    # Induce the programs
    for problem in tqdm(problems):
        program = inductor_obj.model_sample(problem, num_of_samples_per_problem)
        problem.give_program(program)
    
    # Save the programs
    with open('ARC-AGI/data/' + mode + '_with_programs.json', 'w') as file:
        for problem in problems:
            json.dump(problem.id, file)
            file.write('\n')
            for program in problem.program:
                json.dump(program, file, default=lambda x: x.__dict__)
                file.write('\n')


    
        
        

