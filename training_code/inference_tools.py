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

import sys
import torch
import hashlib
import numpy as np
from tqdm import tqdm


def is_unsloth_model(model):
    return model.model_tags is not None and 'unsloth' in model.model_tags


def logits_to_score(sequence, logits):
    assert sequence.ndim == 1
    assert logits.ndim == 2
    assert len(sequence) == len(logits)
    return -logits.log_softmax(-1)[torch.arange(len(logits)), sequence].sum().item()


def calc_score(input, reply, model_tok, cache=None, **_):
    if cache is not None:  # try loading result from cache
        return cache(calc_score)(input=input, reply=reply, model_tok=model_tok)

    # prepare model and tokenizer
    model, tokenizer = model_tok if isinstance(model_tok, (list, tuple)) else model_tok()

    with torch.no_grad():  # calculate score
        input_len = len(tokenizer(input)['input_ids'])
        tokenized = tokenizer([input+reply], return_tensors='pt')
        tokenized.pop('token_type_ids', None)
        sequence = tokenized['input_ids'][0][input_len:].cpu()
        logits = model(**tokenized.to(model.device))['logits'][0, input_len-1: -1].float().cpu()
        return logits_to_score(sequence, logits)


def explore(model, logits, path, eos, max_new_tokens, max_score, pos, cache, score=0.0):
    first_token_logits, logits = logits[0], (logits[1:] if len(logits) > 1 else None)
    softmax = list(enumerate(-first_token_logits.detach().float().log_softmax(-1).cpu()))

    if len(path):  # follow precomputed path first
        softmax[0], softmax[path[0]], path = softmax[path[0]], softmax[0], path[1:]

    return_suffixes = []
    for i, s in softmax:  # loop over all possible tokens
        next_score = score + s.item()
        if next_score < max_score:  # check if still below the score limit, otherwise stop exploration
            if i == eos:  # candidate found, append to suffixes (tokens are aggregated on backward pass)
                suffixes = [([], next_score)]
            elif max_new_tokens > 1:  # check if still below token limit, otherwise stop exploration
                if logits is None:  # if not following the initial guess, calculate logits to pass to explore function
                    if pos < cache[0][0][0].shape[2]:  # cut back key-value-cache when backtracking
                        cache[0] = tuple(tuple(c[:, :, :pos] for c in l) for l in cache[0])
                    logits, cache[0] = model(
                        input_ids=torch.full((1, 1), i, device=model.device),
                        position_ids=torch.full((1, 1), pos, device=model.device),
                        past_key_values=cache[0],
                    )[:2]
                    logits = logits[0]  # unbatch
                # explore suffixes
                suffixes = explore(model, logits, path, eos, max_new_tokens-1, max_score, pos+1, cache, next_score)
            else: suffixes = []

            # update suffixes
            for suffix in suffixes:
                suffix[0].append(i)
            return_suffixes.extend(suffixes)

        logits = None
    return return_suffixes


def dfs(model, input_ids, eos_token_id, max_new_tokens, min_prob, pos=None, attention_mask=None):
    assert not torch.is_grad_enabled()
    assert attention_mask is None or attention_mask.all(), 'not implemented'
    sys.setrecursionlimit(1000 + max_new_tokens)  # avoid stack overflows

    # prepare inputs
    input_ids = torch.as_tensor(input_ids, device=model.device, dtype=int)
    if input_ids.ndim == 2:
        input_ids = input_ids.squeeze(0)
    assert input_ids.ndim == 1, 'batching not supported'

    if pos is None:
        # no guess passed, set generation starting position to length of input
        pos = len(input_ids)
    elif pos < len(input_ids):
        # if guess passed, remove final eos_token from input
        if input_ids[-1] == eos_token_id:
            input_ids = input_ids[:-1]

    # process prompt and best guess
    logits, cache = model(input_ids=input_ids[torch.newaxis])[:2]
    logits = logits[0, pos-1:]

    # run dfs
    result = explore(model, logits, input_ids[pos:], eos_token_id, max_new_tokens, -np.log(min_prob), pos, [cache])

    # return results sorted by scores
    return sorted([(np.array(suffix[::-1]), score_val) for suffix, score_val in result], key=lambda x: x[1])


def infer_single(prompt, model_tok, guess=None, min_prob=None, cache=None, **kwargs):
    assert len(prompt)

    if cache is not None:  # try loading result from cache
        return cache(infer_single)(prompt=prompt, model_tok=model_tok, guess=guess, min_prob=min_prob, **kwargs)

    # prepare model and tokenizer
    model, tokenizer = model_tok if isinstance(model_tok, (list, tuple)) else model_tok()

    with torch.no_grad():
        # tokenize input
        tokenized = tokenizer(prompt, return_tensors='pt').to(model.device)
        input_len = tokenized['input_ids'].shape[-1]
        tokenized.pop('token_type_ids', None)

        if min_prob is not None:  # run dfs if 'min_prob' is passed
            if guess is not None:
                tokenized = tokenizer(guess, return_tensors='pt').to(model.device)
                tokenized.pop('token_type_ids', None)
            ret = dfs(model, **tokenized, pos=input_len, min_prob=min_prob, eos_token_id=tokenizer.eos_token_id, **kwargs)

        else:  # run model 'generate' function
            assert kwargs.get('num_beams', 1) == 1 or not is_unsloth_model(model)
            gen = model.generate(**tokenized, return_dict_in_generate=True, output_logits=True, use_cache=True,
                                 eos_token_id=tokenizer.eos_token_id, **kwargs)
            sequence = gen['sequences'][0, input_len:].cpu()
            logits = torch.stack(gen['logits'], axis=-2)[0].float().cpu()
            ret = [(sequence, logits_to_score(sequence, logits))]

        return [(tokenizer.decode(o), s) for o, s in ret]


def infer_task(keys, dataset, fmt_opts, aug_score_opts=None, pass_guess=True, print_func=print, **kwargs):
    unique_results = {}
    best_guess = (None, float('inf'))
    for key in keys:
        # format task
        key, fmt = dataset.get_task(key, **fmt_opts)
        input_len = dataset.count_tokens(fmt['input'])
        reply_len = dataset.count_tokens(fmt['reply']) if 'reply' in fmt else '?'

        # get current best guess
        guess = None
        if pass_guess and best_guess[0] is not None:
            guess = dataset.get_task(key, reply=best_guess[0], **fmt_opts)[1]['text']
            assert guess.startswith(fmt['input'])

        # run inference
        data = infer_single(prompt=fmt['input'], guess=guess, **kwargs)

        # loop over inference outputs
        for i, (sequence, score) in enumerate(data):
            # decode output
            output, correct, corr_info = dataset.decode(sequence, fmt_opts['lines_sep'], key)

            # print some info
            token_info = f" in:{input_len:>4} out:{dataset.count_tokens(sequence):>3}/{reply_len:>3}"
            score_info = f"{min(np.exp(-score), 0.99):>3.0%}"
            shape_info = f'{output.shape[0]:>2}x{output.shape[1]:<2}' if output is not None else '--x--'
            print_func(f"{token_info} > {shape_info} {corr_info} p={score_info} [{key}.out{i}]")

            if output is not None:
                # add output to results
                hashable = tuple(map(tuple, output))
                if hashable not in unique_results:
                    unique_results[hashable] = dict(output=output, correct=correct, scores_inf={})
                res = unique_results[hashable]

                # calculate score
                res['scores_inf'][key] = score
                if aug_score_opts and 'scores_aug' not in res:
                    aug_score_opts_copy = aug_score_opts.copy()
                    key_hash = int(hashlib.md5(key.split('.')[0].encode('utf-8')).hexdigest()[:6], 16)
                    out_hash = int(hashlib.md5(str(hashable).encode('utf-8')).hexdigest()[:6], 16)
                    np.random.seed(aug_score_opts_copy.pop('seed') + key_hash + out_hash)
                    aug_keys = dataset.augment_keys([key.split('.', 1)[0]], **aug_score_opts_copy)
                    aug_key_fmt = [dataset.get_task(k, reply=output, **fmt_opts) for k in aug_keys]
                    res['scores_aug'] = {key: calc_score(**fmt, **kwargs) for key, fmt in aug_key_fmt}

                # update best guess
                new_score = min(res['scores_inf'].values())
                if new_score < best_guess[1]:
                    best_guess = res['output'], new_score

    return list(unique_results.values())


def inference_run(dataset, fmt_opts, max_new_tokens=None, callback=None, **kwargs):
    # set token limits
    if max_new_tokens is None:
        max_new_tokens = dataset.max_new_tokens(**fmt_opts)
    if 'max_tokens' in fmt_opts:
        fmt_opts = {**fmt_opts, 'max_tokens': fmt_opts['max_tokens'] - max_new_tokens, 'len_name': 'input'}

    # iterate over dataset
    results = {}
    with tqdm(dataset.grouped_keys().items(), desc='inference') as pbar:
        for base_key, tasks in pbar:
            results[base_key] = []
            for task_num, task in enumerate(tasks):
                res = infer_task(keys=task, dataset=dataset, fmt_opts=fmt_opts, max_new_tokens=max_new_tokens,
                                 print_func=pbar.write, **kwargs)
                results[base_key].append(res)
                if callback is not None:
                    callback(res, name=f'{base_key}_{task_num}', value=1/len(tasks), print_func=pbar.write)
    return results
