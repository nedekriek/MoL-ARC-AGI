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

import numpy as np


def max_gen_prob(res):
    return min(res['scores_inf'].values())


def max_aug_prob(res):
    return min(res['scores_aug'].values())


def min_aug_prob(res):
    return max(res['scores_aug'].values())


def sum_aug_prob(res):
    scores = list(res['scores_aug'].values())
    return sum([-np.exp(-s) for s in scores])

def sum_all_prob(res):
    scores = list(res['scores_aug'].values())
    scores.extend(res['scores_inf'].values())
    return sum([-np.exp(-s) for s in scores])

def mul_aug_prob(res, base_log_prob=3):
    scores = list(res['scores_aug'].values())
    return sum([s - base_log_prob for s in scores])


def mul_all_prob(res, base_log_prob=3):
    scores = list(res['scores_aug'].values())
    scores.extend(res['scores_inf'].values())
    return sum([s - base_log_prob for s in scores])

def top_n_augs_mult(res,n = 4):
    # first, sum scores across types of augmentations    
    augtype_sum = dict()
    for key in res['scores_aug']:
        augtype_key = key[key.find('.'):] #this is the aug suffix, e.g.  key1.tp.rt -> .tp.rt
    
        if augtype_key not in augtype_sum:
            augtype_sum[augtype_key] = res['scores_aug'][key]
        else: 
            augtype_sum[augtype_key] += res['scores_aug'][key]
    
    # find the top n augmentations by sum of their scores  
    top_n_augs = sorted(augtype_sum, key=augtype_sum.get, reverse=True)[:n]

    # get base key / use list comprehension to reformat top_n_augs
    # DO THEY ALL HAVE THE SAME BASE KEY???? I believe so.
    base_key = list(res['scores_aug'].keys())[0].split('.',1)[0]
    top_keys = [base_key + aug_suffix for aug_suffix in top_n_augs]

    # filter score dict to top augs
    top_keys_w_scores = {key: res['scores_aug'][key] for key in top_keys}

    # multiply top scores with inf score
    scores = list(top_keys_w_scores.values())
    scores.extend(res['scores_inf'].values())
    return sum([s - base_log_prob for s in scores])

all_score_algos = [
    max_gen_prob,  # highest probability from inference results
    max_aug_prob,  # highest probability from augmented scoring
    min_aug_prob,  # lowest probability from augmented scoring
    sum_aug_prob,  # sum of probabilites from augmented scoring
    sum_all_prob,
    #,  # sum of probabilities from inference results and augmented scoring
    # mul_aug_prob,  # sum of log probabilities from augmented scoring
    # mul_all_prob,  # sum of log probabilities from inference results and augmented scoring combined
    top_n_augs_mult(res,4) # only takes the best 4 augmentation types for each task and multiplies them
]


class EvalTool(object):   # providing on-the-fly evaluation of scoring algorithms
    def __init__(self, n_guesses, score_algos=all_score_algos, sorting_algo=-1):
        self.score_algos = score_algos
        self.n_guesses = n_guesses  # number of guesses allowed
        self.sorting_algo = sorting_algo  # sorting algorithm for results, relevant for final submission (default: last)
        self.n_acc = [0] * len(score_algos)  # counting correct n-guesses for different scoring algorithms
        self.a_acc = 0  # counting cases where the solution is found at all
        self.count = 0  # counting number of tasks seen

    def process_result(self, res, name, value, print_func=print):
        """
        res: contains multiple - len(res) many - candidate outputs
        name: the task id
        """
        # Calculate scores for each result using all scoring algorithms
        for r in res:
            r['scores_alg'] = [algo(r) for algo in self.score_algos]
        
        # Find the position of the correct solution if any
        pos = ([i for i, r in enumerate(res) if r['correct']] + [None])[0]

        # The rest is essentially just bookkeeping and displaying results for this task
        self.count += value
        self.a_acc += value if pos is not None else 0
        corr_info = f"{len(res)} candidates, correct solution {'not found' if pos is None else 'FOUND'}"
        if print_func is not None:
            print_func(f" * task '{name}': {corr_info}")
        for i, algo in enumerate(self.score_algos):
            if pos is not None:
                scores = [r['scores_alg'][i] for r in res]
                rank = np.argsort(np.argsort(scores))[pos]
                if rank < self.n_guesses:
                    self.n_acc[i] += value
            rank_info = f", corr_sol. @{rank + 1:>2} / {len(res)}" if pos is not None else ''
            n_acc_info = f"{self.n_acc[i] / self.count:7.2%} ({self.n_acc[i]:>6.2f}/{self.count:>6.2f})"
            if print_func is not None:
                print_func(f"   {f'{self.score_algos[i].__name__}:':14} {n_acc_info}{rank_info}")
        a_acc_info = f"{self.a_acc / self.count:7.2%} ({self.a_acc:>6.2f}/{self.count:>6.2f})"
        if print_func is not None:
            print_func(f"   {'correct_found:':14} {a_acc_info}\n")
        if self.sorting_algo is not None:
            res.sort(key=lambda x: x['scores_alg'][self.sorting_algo])
