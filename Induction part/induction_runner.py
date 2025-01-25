from transformers import AutoTokenizer
from vllm import LLM
import datetime
from tqdm import tqdm
import json
import torch
import numpy as np
# Import the standard modules we need.

#################################################
# Define the inductor
#################################################

class inductor:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def model_generate_once(self, prompt):
        # This function gives a prompt into the model and get the output of the model. 
        if isinstance(prompt,str):
            inputs  = self.tokenizer(prompt, tokenize=False, add_generation_prompt=True)
        elif isinstance(prompt,list):
            inputs = self.tokenizer.apply_chat_template([
            {"role":"system", "content":prompt[0]['content']},
            {"role":"user", "content":prompt[1]['content']}
            ], tokenize=False, add_generation_prompt=True)
        with torch.no_grad():
            outputs = self.model.generate(**inputs)
        return outputs
    
    def model_sample(self, problem, sample_number:int):
        outputlist = []
        prompt = problem.prompt
        for i in range(sample_number):
            output = self.model_generate_once(prompt)
            outputlist.append(output)
        return outputlist

#################################################
