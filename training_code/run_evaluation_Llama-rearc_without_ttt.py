# Copyright 2024 Daniel Franzen and Jan Disselhoff
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
# from datetime import datetime

from unsloth import FastLanguageModel
from diskcache import Cache

from arc_loader import ArcDataset
from model_tools import load_unsloth_4bit
from inference_tools import inference_run
from selection import EvalTool

# input paths
# print(f"Current working directory: {os.getcwd()}") 
# >> /ARChitects
base_model = os.path.join("da-fr/Llama-3.2-3B-ARChitects-ReArc-bnb-4bit") # auto-downloaded from huggingface.co
print(f"base_model path: {base_model}")
# >> ../Llama-3.2-3B-ARChitects-ReArc-bnb-4bit
# arc_data_path = "data"
# >> ../data

# output paths
output_path = os.path.join('output_evaluation_Llama-rearc_without_ttt')
os.makedirs(output_path, exist_ok=True)

inference_cache = os.path.join(output_path, 'inference_cache')
submission_file = os.path.join(output_path, 'submission.json')
# ./output_evaluation_Llama-rearc_without_ttt/submission.json

# load training dataset (loading from local SCRATCH)
path_arc_data_train = os.path.join("scratch-local", "dlindberg", "data", "arc-agi_training_challenges.json")
path_arc_data_test = os.path.join("scratch-local", "dlindberg", "data", "arc-agi_training_solutions.json")
arc_eval_set = ArcDataset.load_from_json(path_arc_data_train)
arc_eval_set = arc_eval_set.load_solutions(path_arc_data_test)
print(f"Successfully loaded data from {path_arc_data_train}")

# load model
model, tokenizer = load_unsloth_4bit(base_model, local_files_only=False) # Try to load from local files
print(f"Successfully loaded 4-bit model using unsloth")

# set formatting options
fmt_opts = dict(
    preprompt='ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz',
    query_beg='I',
    reply_beg='\n+/-=O',
    reply_end='\n' + tokenizer.eos_token,
    lines_sep='\n',
    max_tokens=128000,
)

# run inference
FastLanguageModel.for_inference(model)
infer_aug_opts = dict(tp='all', rt='all', perm=True, shfl_ex=True, seed=10000)
infer_dataset = arc_eval_set.repeat(2).augment(**infer_aug_opts)
model_cache = Cache(inference_cache).memoize(typed=True, ignore=set(['model_tok', 'guess']))
eval_tool = EvalTool(n_guesses=2)
inference_results = inference_run(
    model_tok=(model, tokenizer),
    fmt_opts=fmt_opts,
    dataset=infer_dataset,
    min_prob=0.1,
    aug_score_opts=infer_aug_opts,
    callback=eval_tool.process_result,
    cache=model_cache,
)

inference_results = "This is just some dummy data for debugging"

print(f"Writing results to submission file: {submission_file}...")
# write submission
with open(submission_file, 'w') as f:
    json.dump(arc_eval_set.get_submission(inference_results), f)
with open(submission_file, 'r') as f:
    print(f"Score for '{submission_file}':", arc_eval_set.validate_submission(json.load(f)))
