import numpy as np
import torch
import json
from dataloader import load_data
from problem_class import problem

################################################
# This is the color map
################################################

COLOR_MAPPING = {
0: "Black",
1: "Blue",
2: "Red",
3: "Green",
4: "Yellow",
5: "Grey",  # instead of "Grey"
6: "Pink",
7: "Orange",
8: "Teal",
9: "Maroon"
}

################################################
# The function to convert a grid to text
################################################


def grid_to_text(grid, transpose = False):
    # This function converts a grid to a textual representation of it.
    if transpose:
        transformed_grid = grid.T
    else:
        transformed_grid = grid
    text = "\n".join(" ".join(COLOR_MAPPING[c] for c in row) for row in transformed_grid) + "\n"
    return text

################################################
# The prompt creator class. We assume we are in evaluation mode.
################################################

class prompt_creator:
    def __init__(self, max_length=400, device='cpu', mode = 'evaluation'):
        self.max_length = max_length
        self.device = device
        self.mode = mode
        if mode not in ['evaluation', 'training', 'test']:
            raise ValueError("mode must be either 'evaluation' or 'training' or 'test', what are you doing now?")

    
    def create_prompt_from_test_task(self, problem, transpose = False):
        # This is the function that turns a task into a prompt.
        prompt ="Given input-output grid pairs as reference examples, carefully observe the patterns to predict the output grid for new test input. Each pair follows the same transformation rule. Grids are 2D arrays represented as strings, with cells (colors) separated by spaces and rows by newlines."
        prompt += "\nHere are the input and output grids for the reference examples:\n"
        i = 0
        for pair in problem.train:
            i += 1
            prompt += f"Example {i+1}\n"
            prompt += f"Input:\n{grid_to_text(pair['input'] , transpose)}\nOutput:\n{grid_to_text(pair['output'], transpose)}\n\n" 
        prompt += "Here is the input grid for the test example:\n"
        prompt += "Input:\n" +  grid_to_text(problem.test[0]["input"], transpose) + "\n"
        common_lib_prefix = ""
        prompt = common_lib_prefix + prompt
        prompt += "\nWrite a Python function `transform` that can convert any given input grid to its corresponding output grid based on the pattern observed in the reference examples."
        return prompt
    
    def add_prompt_engineering(self, prompt):
        # This function adds additional prompt engineering to the prompt and turns it into a dictionary that fits the apply_chat_template function of tokenizers.
        message = [
            {'role': 'system', 'content': "You are an world-class puzzle solver who are extremely good at spotting patterns and solving puzzles. You are also an expert Python programmer who can write code to solve puzzles."}
            , {'role': 'user', 'content': prompt}
        ]
        return message



