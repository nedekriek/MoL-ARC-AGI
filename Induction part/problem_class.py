import numpy as np
import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException

# The apply_program function takes in an input and a program, and returns the output of the program on the input.
def apply_program(input, program, kill_time=30):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(kill_time)
    try:
        exec(program)
        output = transform(input)
    except TimeoutException:
        output = None
    finally:
        signal.alarm(0)  # Disable the alarm
    return output

def check_program(train, program):
    indicator = True
    for example in train:
        if apply_program(example['input'], program) != example['output']:
            indicator = False  
    return indicator


################################################
# We create a class to represent a data token
################################################


class problem:
    def __init__(self, id = None, train = None, test = None, type = None, prompt = None, program = None, induction_solution = None):
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
    
    def filter_program(self):
        working_program = []
        for program in self.program:
            if check_program(self.train, program):
                working_program.append(program)
        self.give_program(working_program)

    def give_induction_solution(self, induction_solution):
        self.induction_solution = induction_solution