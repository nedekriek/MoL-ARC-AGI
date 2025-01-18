import json
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor


# Load the JSON file
def safe_load_json(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError as e:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {file_path}: {e}")
        return None
    
# Load Files concurrently
def load_all_json_files_concurrently(directory):
    json_files_names = list(entry.name for entry in os.scandir(directory) if entry.is_file() and entry.name.endswith('.json'))
    json_files_paths = [os.path.join(directory, file) for file in json_files_names]
    
    with ThreadPoolExecutor() as executor:
        data = dict(zip(json_files_names, executor.map(safe_load_json, json_files_paths)))
    
    return data

# Convert 'train' and 'test' data within the task dictionary from lists to numpy arrays
def convert_task_to_numpy_arrays(task):
    if 'train' in task:
        for i in range(len(task['train'])):
            task['train'][i]['input'] = np.array(task['train'][i]['input'], dtype=np.uint8)
            task['train'][i]['output'] = np.array(task['train'][i]['output'], dtype=np.uint8)
    if 'test' in task:
        for i in range(len(task['test'])):
            task['test'][i]['input'] = np.array(task['test'][i]['input'], dtype=np.uint8)
            task['test'][i]['output'] = np.array(task['test'][i]['output'], dtype=np.uint8)
    
    return task