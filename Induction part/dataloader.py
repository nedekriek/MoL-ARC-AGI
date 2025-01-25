## This file is used to load the data from the files and return the data in the form of a list of problems
import os
import json
import numpy as np

################################################
# We create a class to represent a data token
################################################


class problem:
    def __init__(self, id = None, train = None, test = None, type = None, prompt = None, program = None):
        self.id = id
        self.train = train
        self.test = test
        self.type = type
        self.prompt = prompt
        self.program = program
    
    def give_prompt(self, prompt):
        self.prompt = prompt

    def give_program(self, program):
        self.program = program

################################################
# The function to load the data
################################################


def load_data(mode):
    problems = []
    folder_path = 'ARC-AGI/data/' + mode + '/'
    for filename in os.listdir(folder_path):
        with open(os.path.join(folder_path, filename), 'r') as file:
            data = json.load(file)
            id = filename.split('.')[0]
            problem_instance = problem(id = id, train = data['train'], tests = data['test'], type = mode)
            problems.append(problem_instance)
    return problems