# %% [markdown] cell 0
# # ARC-AGI-2: NVARC Qwen3-4B TTT + DFS (our hardened variant)
# 
# Derived from `mikelou1/arc-agi2-lb33-89-minimal-perfpatch` (NVARC per-task LoRA test-time
# training, turbo DFS decoding over 16 augmented views, augmented-NLL rescoring, `score_kgmon`
# selection, 4 x L4 workers). Model maths, hyper-parameters, prompts, token ids and the
# selection algorithm are unchanged. Our changes (see README for the full diff):
# 
# 1. **Determinism**: stable md5-based seeds instead of `hash(bk)`, `PYTHONHASHSEED=0`, per-task seeding, sorted result loading.
# 2. **Robust attempts**: `attempt_2` is never a duplicate of `attempt_1`, nor `[[0]]` when any other candidate exists.
# 3. **CPU program search** (background, niced): small DSL; only programs that reproduce *all* train pairs may fill empty/duplicate slots; it never replaces a model `attempt_1`.
# 4. **Time/crash safety**: placeholder submission written first; training stops at the deadline; hard kill of workers; a crashed rank no longer kills the other ranks (no `mp.spawn`); per-task exception handling; atomic pickle writes; tolerant loading; guarded final assembly and format validation.
# 5. **Scheduling**: cheap tasks first; tasks that crashed or produced no candidate get one retry with new augmentation seeds when time remains.

# %% cell 1
# Keep the original global 10-minute submission/write buffer.
import time
global_end_time = time.time() + 12 * 3600 - 600

# [ours] run configuration
import os
ARC_ORDER = "cheap_first"      # "cheap_first" (ours) or "sorted" (base order)
ARC_EVAL_SUBSET = "smoke"      # commit (non-rerun) mode only: "smoke" = base 4 tasks, "all" = 120 eval tasks
DSL_TIME_LIMIT = 3600          # seconds for the background CPU program search
RERUN_MODE = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
COMP_DIR = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2"
TEST_PATH = f"{COMP_DIR}/arc-agi_test_challenges.json" if RERUN_MODE else f"{COMP_DIR}/arc-agi_evaluation_challenges.json"
SOLUTIONS_PATH = None if RERUN_MODE else f"{COMP_DIR}/arc-agi_evaluation_solutions.json"
print("rerun mode:", RERUN_MODE, "| test file:", TEST_PATH)

# %% cell 2
# Preserve the baseline environment workaround.
# !pip uninstall -y tensorflow

# %% cell 3
# %%writefile arc_loader.py
import json
import numpy as np
from transformers import AutoTokenizer


def convert_grid_to_string(grid) -> str:
    text = ""
    for row in grid:
        for cell in row:
            text += str(int(cell))
        text += "\n"
    return text.strip()

def is_valid_solution(guess):
    return isinstance(guess, np.ndarray) and guess.ndim == 2 and all(0 < x <= 30 for x in guess.shape)

def shuffled(data_list):
    return np.random.permutation(data_list).tolist()

def permute_mod(a, descriptor, invert=False):
    permutation = [int(i) for i in descriptor if str(i).isdigit()]
    assert sorted(permutation)==list(range(10))
    a = np.asarray(a)
    if a.ndim==3:
        if not invert: permutation = np.argsort(permutation)
        a = a[..., permutation]
    else:
        assert a.ndim==2
        if invert: permutation = np.argsort(permutation)
        a = np.asarray(permutation)[a]
    return a

def permute_rnd_all_(query):
    permutation = np.random.permutation(10).tolist()
    return 'permute' + ''.join(map(str, permutation))


class QwenFormatter:

    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def fmt_query(self, query) -> str:
        grid_input = convert_grid_to_string(query[0]["input"])
        return "<|im_start|>user\n" + grid_input + "<|im_end|><|im_start|>assistant\n"

    def fmt_reply(self, reply) -> str:
        return convert_grid_to_string(reply[0]) + "<|im_end|>"

    def fmt_train(self, train, last_is_challenge=False) -> str:
        if last_is_challenge:
            test = train[-1]
            train = train[:-1]
        else:
            test = None
        text = ""
        for x in train:
            grid_input = convert_grid_to_string(x["input"])
            grid_output = convert_grid_to_string(x["output"])
            text += f"<|im_start|>user\n{grid_input}<|im_end|><|im_start|>assistant\n{grid_output}<|im_end|>"
        if test is not None:
            text += self.fmt_query([test]) + self.fmt_reply([test["output"]])
        return text

    def max_new_tokens(self):
        max_sized_reply = np.zeros([30, 30], dtype=int)
        tokens = self.tokenizer.encode(self.fmt_reply([max_sized_reply]))
        return len(tokens) + 1

    def convert_tokens_to_array(self, tokens, limit_rows=30):
        if len(tokens) < 2:
            return None
        text = self.tokenizer.decode(tokens[:-1])
        try:
            lines = text.strip().split("\n")
            by_rows = [row for row in [[int(x) for x in line if x.isdigit()] for line in lines] if len(row)]
            if len(by_rows) > limit_rows:
                by_rows = by_rows[:limit_rows]
            array = np.array(by_rows, dtype=int)
            if is_valid_solution(array):
                return array
        except:
            pass
        return None


class ArcDataset:

    @staticmethod
    def forward_mod(a, key, use_perm=True):
        if a is None: return a
        for op in key.split('.')[1:]:
            if   op=='rot90':              a = np.rot90(a)
            elif op=='transpose':          a = np.swapaxes(a, 0, 1)
            elif op.startswith('permute'): a = permute_mod(a, op, invert=False) if use_perm else a
            elif op.startswith('copy'):    a = np.copy(a)
            elif op.startswith('out'):     a = a
            elif op.startswith('ex'):      a = a
            elif op.startswith('run'):     a = a
            else: raise NotImplementedError(f"Inversion of operation '{op}' unknown.")
        return a

    @staticmethod
    def invert_mod(a, key, inv_perm=True):
        if a is None: return a
        for op in key.split('.')[1:][::-1]:
            if   op=='rot90':              a = np.rot90(a, k=3)
            elif op=='transpose':          a = np.swapaxes(a, 0, 1)
            elif op.startswith('permute'): a = permute_mod(a, op, invert=True) if inv_perm else a
            elif op.startswith('copy'):    a = np.copy(a)
            elif op.startswith('out'):     a = a
            elif op.startswith('ex'):      a = a
            elif op.startswith('run'):     a = a
            else: raise NotImplementedError(f"Inversion of operation '{op}' unknown.")
        return a

    def __init__(self, queries, replies={}, keys=None, is_orig=False):
        if keys is not None: keys = [k for k in keys if k is not None]
        self.queries = queries if keys is None else {k: queries[k] for k in keys}
        self.replies = replies if keys is None else {k: replies[k] for k in keys if k in replies}
        self.is_orig = is_orig
        self.keys = sorted(queries.keys()) if keys is None else keys
        self.transposed_dataset = None

    def change_keys(self, keys, keep_flags=False):
        flags = dict(is_orig=self.is_orig) if keep_flags else {}
        return self.__class__(queries=self.queries, replies=self.replies, keys=keys, **flags)

    @classmethod
    def from_file(cls, queries_file, keys=None):
        with open(queries_file) as f:
            queries = f.read()
        return cls(
            queries=json.loads(queries),
            is_orig=True,
            keys=keys,
        )

    def load_replies(self, replies_file):
        print(f"*** Load solutions from '{replies_file}'...")
        with open(replies_file) as f: replies = f.read()
        replies_parsed = json.loads(replies)
        self.replies = {k: replies_parsed[k] for k in self.keys}
        return self

    def split_multi_replies(self):
        key_indices = [(k, i) for k in self.keys for i in range(len(self.queries[k]['test']))]
        return self.__class__(
            keys=[f'{k}_{i}' for k, i in key_indices],
            queries={f'{k}_{i}': {'train': self.queries[k]['train'], 'test': [self.queries[k]['test'][i]]} for k, i in key_indices},
            replies={f'{k}_{i}': [self.replies[k][i]] for k, i in key_indices if k in self.replies},
        )

    def shuffled(self):
        return self.__class__(queries=self.queries, replies=self.replies, keys=shuffled(self.keys))

    def append(*datasets):
        return datasets[0].__class__(
            queries={k: v for d in datasets for k, v in d.queries.items()},
            replies={k: v for d in datasets for k, v in d.replies.items()},
            keys   =[k    for d in datasets for k    in d.keys           ],
        )

    def mod_single(self, mod_func, descriptor, i, keep_key, inputs_only):
        queries = {}
        replies = {}
        keys    = []
        for k0 in self.keys:
            desc = (('copy{i}' if mod_func is np.copy else mod_func.__name__) if descriptor is None else descriptor if isinstance(descriptor, str) else descriptor(self.queries[k0])).format(i=i)
            func = lambda a, d: np.asarray(mod_func(a) if descriptor is None else mod_func(a, d)).tolist()
            k1 = k0 if keep_key else f"{k0}.{'I' if inputs_only else ''}{desc}"
            keys.append(k1)
            queries[k1] = {m: [{t: (func(a, desc) if t=='input' or not inputs_only else a) for t, a in x.items()} for x in e] for m, e in self.queries[k0].items()}
            if k0 in self.replies:
                replies[k1] = [func(a, desc) for a in self.replies[k0]]
        ret = self.__class__(queries=queries, replies=replies, keys=keys)
        return ret

    def mod(self, mod_func, descriptor=None, n=1, stack=None, keep=False, keep_key=False, shuffle=False, join=True, inputs_only=False):
        assert not (keep and keep_key)
        cur = self
        ret = [cur.shuffled() if shuffle else cur] if keep else []
        if stack is None: stack = mod_func.__name__.startswith('rot')
        for i in range(n):
            cur = (cur if stack else self).mod_single(mod_func, descriptor, i=i, keep_key=keep_key, inputs_only=inputs_only)
            ret.append(cur.shuffled() if shuffle else cur)
        return self.__class__.append(*ret) if join else ret

    def get(self, key, formatter: QwenFormatter):
        train = formatter.fmt_train(self.queries[key]['train'])
        query = formatter.fmt_query(self.queries[key]['test'])
        reply = formatter.fmt_reply(self.replies[key]) if key in self.replies else ''
        text = train+query+reply if reply else formatter.fmt_train(self.queries[key]['train'], last_is_challenge=True)
        return dict(key=key, train=train, query=query, reply=reply, input=train+query, text=text)

    def as_list(self, formatter: QwenFormatter):
        return [self.get(key, formatter) for key in self.keys]

    def get_length(self, key, formatter: QwenFormatter, name, max_of_transposed=False):
        if formatter is None:
            if   name=='input': return sum(np.prod(np.shape(v)) for v3 in self.queries[key].values() for v2 in v3 for v in v2.values())
            elif name=='reply': return sum(np.prod(np.shape(v)) for v in self.replies[key])
            else: assert False
        else:
            datasets = [self]
            if max_of_transposed:
                if self.transposed_dataset is None: self.transposed_dataset = self.mod(np.transpose, keep=False, keep_key=True)
                datasets.append(self.transposed_dataset)
            return max(len(formatter.tokenizer.encode(ds.get(key, formatter=formatter)[name])) for ds in datasets)

    def cut_to_len(self, formatter, name, max_len, from_end=False):
        temp_ds = self.change_keys(self.keys)
        new_keys = []
        new_queries = {}
        new_replies = {}
        for key in self.keys:
            reply = temp_ds.replies.get(key)
            while max_len<temp_ds.get_length(key, formatter=formatter, name=name):
                query = temp_ds.queries[key]
                if not key.split('.')[-1].startswith('ex'):
                    key = f"{key}.ex{''.join(map(str, range(len(query['train']))))}"
                key_split = key.split('.')
                assert key_split[-1].startswith('ex')
                key = '.'.join(key_split[:-1] + [f'ex{key_split[-1][2:-1] if from_end else key_split[-1][3:]}'])
                temp_ds.queries[key] = {k: ((v[:-1] if from_end else v[1:]) if k=='train' else v) for k, v in query.items()}
                if reply is not None:
                    temp_ds.replies[key] = reply
            new_keys.append(key)
            new_queries[key] = temp_ds.queries[key]
            if reply is not None: new_replies[key] = reply
        return self.__class__(keys=new_keys, queries=new_queries, replies=new_replies)
    
    def shuffle_ex(self, perm=None, keep_max=None):
        new_keys = []
        new_queries = {}
        new_replies = {}
        for key in self.keys:
            n = len(self.queries[key]['train'])
            p = np.random.permutation(n) if perm is None else perm
            if keep_max is not None: p = p[:keep_max]
            new_key = f'{key}.ex' + ('-' if (p.max()>9) else '').join(map(str, p.tolist()))
            new_keys.append(new_key)
            new_queries[new_key] = {k: (np.array(v, dtype=object)[p].tolist() if k=='train' else v) for k, v in self.queries[key].items()}
            if key in self.replies: new_replies[new_key] = self.replies[key]
        return self.__class__(queries=new_queries, replies=new_replies, keys=new_keys)

    def augment(self, n=1, shfl_keys=False, seed=42):
        np.random.seed(seed)
        d = self
        d = d.mod(np.transpose, keep=True)
        d = d.mod(np.rot90, n=3, keep=True)
        d = d.mod(permute_mod, permute_rnd_all_, n=n, shuffle=shfl_keys, keep=False)
        d = d.shuffle_ex()
        return d

    def get_submission(self, results=None):
        assert self.is_orig==True, 'Must be run on original dataset.'
        submission = {k: [{f'attempt_{i+1}': [[0]] for i in range(2)} for _ in range(len(self.queries[k]['test']))] for k in self.keys}
        if results is not None: self.fill_submission(results, submission)
        return submission

    @staticmethod
    def fill_submission(results, submission):
        print(f'*** Generating submission for {len(results)} outputs...')
        for k, v in results.items():
            base_id, base_nr = k.split('_')
            target_dict = submission[base_id][int(base_nr)]
            for i, g in enumerate(v[:len(target_dict)]):
                target_dict[f'attempt_{i+1}'] = g.tolist()

    def validate_submission(self, submission):
        assert self.is_orig==True, 'Must be run on original dataset.'
        score = 0
        for k, v in self.replies.items():
            for i, r in enumerate(v):
                for attempt in ['attempt_1', 'attempt_2']:
                    if np.array_equal(r, submission[k][i][attempt]):
                        score += 1 / len(v)
                        break
        return score

# %% cell 4
# %%writefile arc_decoder.py
import os
import bz2
import pickle
import numpy as np

def hashable(guess):
    return tuple(map(tuple, guess))

def score_sum(guesses, getter):
    guess_list = list(guesses.values())
    scores = {}
    for g in guess_list:
        h = hashable(g["solution"])
        x = scores[h] = scores.get(h, [[], g["solution"]])
        x[0].append(g)
    scores = [(getter(sc), o) for sc, o in scores.values()]
    scores = sorted(scores, key=(lambda x: x[0]), reverse=True)
    ordered_outputs = [x[-1] for x in scores]
    return ordered_outputs

def getter_full_probmul_3(guesses, baseline=3):
    inf_score = np.sum([baseline-g["beam_score"] for g in guesses])
    aug_score = np.mean([np.sum([baseline-s for s in g["score_aug"]]) for g in guesses])
    return inf_score + aug_score

def score_full_probmul_3(guesses):
    return score_sum(guesses, getter_full_probmul_3)

def getter_kgmon(guesses):
    inf_score = len(guesses)
    aug_score = np.mean([np.mean(g["score_aug"]) for g in guesses])
    return inf_score - aug_score

def score_kgmon(guesses):
    return score_sum(guesses, getter_kgmon)


selection_algorithms = [
    score_full_probmul_3,
    score_kgmon,
]


class ArcDecoder:
    
    def __init__(self, dataset, n_guesses):
        self.dataset = dataset
        self.n_guesses = n_guesses
        self.decoded_results = {}

    def load_decoded_results(self, store, run_name=""):
        # [ours] tolerate a missing store, partial ".tmp" files and truncated pickles
        # (e.g. a worker killed at the hard deadline while writing).
        if not os.path.isdir(store):
            print(f"*** No decoded results in '{store}'")
            return
        n_bad = 0
        for key in sorted(os.listdir(store)):
            if key.endswith(".tmp"):
                continue
            try:
                with bz2.BZ2File(os.path.join(store, key)) as f:
                    outputs = pickle.load(f)
            except Exception as e:
                n_bad += 1
                print(f"*** Skipping unreadable result file {key}: {e!r}")
                continue
            base_key = key.split(".")[0]
            self.decoded_results[base_key] = self.decoded_results.get(base_key, {})
            for i, sample in enumerate(outputs):
                self.decoded_results[base_key][f"{key}{run_name}.out{i}"] = sample

    def run_selection_algo(self, selection_algorithm=score_kgmon):
        return {bk: selection_algorithm({k: g for k, g in v.items()}) for bk, v in self.decoded_results.items()}

    def benchmark_selection_algos(self):
        print("*** Benchmark selection algorithms...")

        labels = {}
        num_tasks_per_puzzle = {}
        num_solved_keys = 0
        num_total_keys = 0

        correct_beam_scores = []

        for basekey, basevalues in self.decoded_results.items():

            mult_key, mult_sub = basekey.split("_")
            num_tasks_per_puzzle[mult_key] = max(num_tasks_per_puzzle.get(mult_key, 0), int(mult_sub) + 1)

            labels[basekey] = correct_solution = self.dataset.replies[basekey][0]

            for subkey, sample in basevalues.items():

                solution = sample["solution"]
                beam_score = sample["beam_score"]
                aug_mean = np.mean(sample["score_aug"])

                if np.shape(correct_solution) != np.shape(solution):
                    corr_str = "bad_xy_size"
                elif np.array_equal(correct_solution, solution):
                    corr_str = "ALL_CORRECT"
                    num_solved_keys += 1
                    correct_beam_scores.append(beam_score)
                else:
                    corr_str = "bad_content"

                output_len = f"{solution.shape[0]}x{solution.shape[1]}"

                if corr_str == "ALL_CORRECT":
                    print(f"{corr_str}:{beam_score:8.5f} - {aug_mean:8.5f} {output_len:5s} [{subkey}]")
                num_total_keys += 1

        print(f" subkeys: {num_solved_keys}/{num_total_keys}")
        if correct_beam_scores:
            print(f" avg correct beam score: {np.mean(correct_beam_scores):8.5f}")
            print(f" max correct beam score: {np.max(correct_beam_scores):8.5f}")

        num_puzzles = len(num_tasks_per_puzzle)

        for selection_algorithm in selection_algorithms:
            name = selection_algorithm.__name__
            selected = self.run_selection_algo(selection_algorithm)
            correct_puzzles = {k for k, v in selected.items() if any(np.array_equal(guess, labels[k]) for guess in v[:self.n_guesses])}
            print(correct_puzzles)
            score = sum(1/num_tasks_per_puzzle[k.split("_")[0]] for k in correct_puzzles)
            print(f" acc: {score:5.1f}/{num_puzzles:3} ('{name}')")

# %% cell 5
# %%writefile arc_solver.py
from unsloth import FastLanguageModel, UnslothTrainingArguments, UnslothTrainer
from arc_loader import ArcDataset, QwenFormatter

import gc
import os
import io
import time
import torch
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from collections import defaultdict

from typing import Any, Union
from transformers import DataCollatorForLanguageModeling, TrainerCallback

# [ours] determinism / robustness helpers
import hashlib
import random
import traceback

import logging
from contextlib import redirect_stdout, redirect_stderr

from peft import get_peft_model_state_dict, set_peft_model_state_dict

import bz2
import pickle

logging.disable(logging.WARNING)

ARC_VOCAB = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "Ċ": 10,
    "<|im_end|>": 15,
}

ARC_TOKENS = list(ARC_VOCAB.values())
USER_TOKEN_ID = 11
ASSISTANT_TOKEN_ID = 12
PAD_ID = 13
EOS_ID = 15


class UnslothFixedTrainer(UnslothTrainer):

    # Issue https://github.com/unslothai/unsloth/issues/2435

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Fixed compute_loss that handles Unsloth's view tensor issue"""
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if hasattr(unwrapped_model, "_get_name") and "unsloth" in unwrapped_model._get_name().lower():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        # 🔧 KEY FIX: Clone the loss tensor before in-place operations
        if hasattr(loss, "clone"):
            loss = loss.clone()  # Converts view tensor to independent tensor
        # Now safe for DDP gradient scaling
        if self.accelerator.num_processes > 1:
            loss = loss * self.accelerator.num_processes
        return (loss, outputs) if return_outputs else loss


class QwenDataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        batch = super().torch_call(examples)
        for i in range(len(examples)):
            labels = batch["input_ids"][i].clone()
            user_start_idx = np.where(labels == USER_TOKEN_ID)[0].tolist()
            assistant_start_idx = np.where(labels == ASSISTANT_TOKEN_ID)[0].tolist()
            start_idx = sorted(user_start_idx + assistant_start_idx)
            end_idx = np.where(labels == EOS_ID)[0]
            batch["labels"][i, :] = -100
            for j, (start, end) in enumerate(zip(start_idx, end_idx)):
                assert start < end
                if j % 2 == 1:
                    start += 2
                    end += 1
                    batch["labels"][i, start:end] = labels[start:end]
        return batch


# Minimal performance patch: preserve the baseline beam set and ranking, but transfer
# only the 12 ARC-token NLL values to CPU instead of every Qwen vocabulary logit.
_ARC_TOKEN_ID_CACHE = {}


def _arc_token_ids(device):
    key = str(device)
    token_ids = _ARC_TOKEN_ID_CACHE.get(key)
    if token_ids is None:
        token_ids = torch.tensor(ARC_TOKENS, dtype=torch.long, device=device)
        _ARC_TOKEN_ID_CACHE[key] = token_ids
    return token_ids


def turbo_dfs(model, logits, max_new_tokens, max_score, scores, pos, cache, start_time, end_time) -> dict:

    n = logits.size(0)

    # Algebraically identical to: scores - logits.float().cpu().log_softmax(-1),
    # restricted to the same ARC_TOKENS used by the baseline DFS loop.
    logits_f = logits.float()
    token_ids = _arc_token_ids(logits.device)
    arc_logits = logits_f.index_select(-1, token_ids)
    nll = (
        torch.as_tensor(scores, dtype=torch.float32, device=logits.device).view(n, 1)
        + torch.logsumexp(logits_f, dim=-1, keepdim=True)
        - arc_logits
    ).cpu()

    suffixes = defaultdict(list)

    candidates = dict()

    for i in range(n):
        candidates[i] = []
        for token_idx, t in enumerate(ARC_TOKENS):
            score = nll[i, token_idx].item()
            if score < max_score:
                if t == EOS_ID:
                    suffixes[i].append((score, [t]))
                elif max_new_tokens > 1:
                    candidates[i].append((score, t))

    for i in range(n):
        candidates[i] = sorted(candidates[i], key=lambda x:x[0]) #[:5]
    
    while time.time() - start_time < 540 and time.time() < end_time:

        batch_tokens = []
        batch_scores = []
        num_alive_beams = 0

        for i in range(n):
            if len(candidates[i]) == 0:
                batch_tokens.append(PAD_ID)
                batch_scores.append(1000)
            else:
                score, t = candidates[i].pop(0)
                batch_tokens.append(t)
                batch_scores.append(score)
                num_alive_beams += 1

        if num_alive_beams == 0:
            break

        outputs = model(
            input_ids=torch.tensor(batch_tokens, device=model.device, dtype=torch.long).view(-1, 1),
            position_ids=torch.full((n, 1), pos, device=model.device),
            past_key_values=cache,
            return_dict=True,
            use_cache=True,
        )

        next_suffixes = turbo_dfs(
            model,
            logits=outputs.logits[:, -1],
            max_new_tokens=max_new_tokens-1,
            max_score=max_score,
            scores=batch_scores,
            pos=pos+1,
            cache=outputs.past_key_values,
            start_time=start_time,
            end_time=end_time,
        )

        for batch_id, beams in next_suffixes.items():
            for score, suffix_tokens in beams:
                suffix_tokens.insert(0, batch_tokens[batch_id])
                suffixes[batch_id].append((score, suffix_tokens))

    return suffixes


@torch.no_grad()
def inference_turbo_dfs(model, prefix_tokens, max_new_tokens, max_score, end_time):
    input_ids = torch.tensor(prefix_tokens, device=model.device, dtype=torch.long)
    outputs = model(input_ids=input_ids, return_dict=True, use_cache=True)
    suffixes = turbo_dfs(
        model,
        logits=outputs.logits[:, -1],
        max_new_tokens=max_new_tokens,
        max_score=max_score,
        scores=[0.0] * input_ids.size(0),
        pos=input_ids.size(1),
        cache=outputs.past_key_values,
        start_time=time.time(),
        end_time=end_time,
    )
    result = []
    for batch_id, beams in suffixes.items():
        sorted_beams = sorted(beams, key=lambda x:x[0])
        result.append((batch_id, sorted_beams))
    return result


@torch.no_grad()
def calc_scores(queries, answers, tokenizer, model):
    batch_query_tokens = []
    batch_answer_tokens = []
    batch_tokens = []
    batch_lengths = []
    for query, answer in zip(queries, answers):
        query_tokens = tokenizer.encode(query)
        answer_tokens = tokenizer.encode(answer)
        tokens = query_tokens + answer_tokens
        batch_query_tokens.append(query_tokens)
        batch_answer_tokens.append(answer_tokens)
        batch_tokens.append(tokens)
        batch_lengths.append(len(tokens))
    max_len = max(batch_lengths)
    padded_tokens = []
    for tokens in batch_tokens:
        padded = tokens + [PAD_ID] * (max_len - len(tokens))
        padded_tokens.append(padded)
    input_ids = torch.tensor(padded_tokens, device=model.device, dtype=torch.long)

    # Keep logits on GPU and gather only the target-token scores. KV cache is not
    # consumed by teacher-forced scoring, so disabling it removes redundant writes.
    outputs = model(input_ids=input_ids, return_dict=True, use_cache=False)
    batch_logits = outputs.logits.float()
    batch_log_norm = torch.logsumexp(batch_logits, dim=-1)
    result = []
    for row_id, (query_tokens, answer_tokens) in enumerate(zip(batch_query_tokens, batch_answer_tokens)):
        query_length = len(query_tokens)
        answer_length = len(answer_tokens)
        positions = torch.arange(
            query_length - 1,
            query_length - 1 + answer_length,
            device=model.device,
        )
        target_tokens = torch.tensor(answer_tokens, device=model.device, dtype=torch.long)
        answer_log_probs = (
            batch_logits[row_id, positions, target_tokens]
            - batch_log_norm[row_id, positions]
        )
        result.append(-answer_log_probs.sum().item())
    return result


# [ours] Stable, PYTHONHASHSEED-independent seed (the base used hash(bk), which changes
# between interpreter runs because str hashing is randomised).
def stable_seed(key, salt=0):
    digest = hashlib.md5(f"{key}|{salt}".encode("utf-8")).hexdigest()
    return int(digest, 16) % (1024 ** 2)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DeadlineCallback(TrainerCallback):
    """[ours] Stop test-time training once the global deadline is reached, so a task
    started just before end_time cannot run training for another ~5-15 minutes and
    push the notebook past the 12 h hard limit."""

    def __init__(self, deadline):
        self.deadline = deadline

    def on_step_end(self, args, state, control, **kwargs):
        if time.time() > self.deadline:
            control.should_training_stop = True
        return control


def check_tokenizer(tokenizer, rank):
    """[ours] Sanity-check the 12 ARC token ids (some public forks carry a mojibake
    newline key). Only the ids are used by the decoder; we log any mismatch loudly."""
    try:
        vocab = tokenizer.get_vocab()
        bad = {k: (v, vocab.get(k)) for k, v in ARC_VOCAB.items() if vocab.get(k) != v}
        nl = tokenizer.encode("\n")
        if bad or nl != [ARC_VOCAB["\u010a"]]:
            print(f"[Rank {rank}] WARNING tokenizer mismatch: {bad} newline->{nl}")
        else:
            print(f"[Rank {rank}] tokenizer check ok (newline id {nl[0]})")
    except Exception as e:
        print(f"[Rank {rank}] tokenizer check skipped: {e!r}")


def _atomic_pickle_bz2(obj, path):
    tmp = path + ".tmp"
    with bz2.BZ2File(tmp, "w") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


# [ours] pass 0 = the base configuration; pass 1 = retry with different augmentation
# seeds for tasks that crashed or produced no candidate at all (only if time remains).
PASS_CFG = {
    0: dict(train_seed=1, eval_seed=2, suffix=""),
    1: dict(train_seed=1001, eval_seed=1002, suffix=".r1"),
}
MIN_TASK_START_SECONDS = 120     # do not start a new task this close to end_time
RETRY_MIN_SECONDS = 600          # a retry needs at least this much time left
MAX_CONSECUTIVE_FAILURES = 3


def worker(rank, queue, end_time, retry_queue=None):

    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")

    peft_params = dict(
        r=256,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing=False,
        random_state=42,
        use_rslora=True,
        loftq_config=None,
    )

    train_args = dict(
        per_device_eval_batch_size=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        warmup_steps=0,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        learning_rate=5e-5,
        optim="adamw_torch",
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",
        save_strategy="no",
        eval_strategy="no",
        logging_strategy="no",
        fp16=False,
        bf16=True,
        # Disable FSDP (use standard DDP)
        fsdp="",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,
        gradient_checkpointing=False,
    )

    max_seq_length = 8192

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1",
        full_finetuning=False,
        load_in_4bit=False,
        local_files_only=True,
        use_gradient_checkpointing=False,
        max_seq_length=max_seq_length,
    )

    model = FastLanguageModel.get_peft_model(model, **peft_params)

    for name, param in model.named_parameters():
        if param.dtype == torch.float32:
            param.data = param.data.to(torch.bfloat16)

    default_weights = get_peft_model_state_dict(model, adapter_name="default")
    default_weights = {k: v.clone().detach() for k, v in default_weights.items()}

    collator = QwenDataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        mlm=False,
    )

    formatter = QwenFormatter(tokenizer=tokenizer)

    check_tokenizer(tokenizer, rank)

    max_new_tokens = formatter.max_new_tokens()

    max_score = -np.log(0.2)

    if rerun_mode:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"
    else:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"

    test_path = os.getenv("ARC_TEST_PATH") or test_path  # [ours] override for local tests

    arc_test_set = ArcDataset.from_file(test_path)

    dir_outputs = os.getenv("ARC_OUTPUT_DIR", "/kaggle/inference_outputs")
    os.makedirs(dir_outputs, exist_ok=True)

    def next_item():
        # Main queue first (keys ordered by the starter), then the retry queue.
        try:
            item = queue.get_nowait()
            if item is not None:
                return item, 0, None
        except Exception:
            pass
        if retry_queue is not None:
            try:
                item = retry_queue.get_nowait()
                if item is not None:
                    return item[0], 1, item[1]
            except Exception:
                pass
        return None, None, None

    consecutive_failures = 0

    while True:

        if time.time() > end_time - MIN_TASK_START_SECONDS:
            print(f"[Rank {rank}] stop!")
            break

        key, pass_id, est_seconds = next_item()
        if key is None:
            break

        if pass_id == 1:
            remaining = end_time - time.time()
            need = max(RETRY_MIN_SECONDS, est_seconds or 0) + 60
            if remaining < need:
                print(f"[Rank {rank}] skip retry of {key}: {remaining:.0f}s left < {need:.0f}s")
                continue
            print(f"[Rank {rank}] RETRY {key} (pass 1, est {est_seconds:.0f}s)")

        cfg = PASS_CFG[pass_id]
        start_time = time.time()
        n_written = defaultdict(int)
        timed_out = False
        failed = False

        try:
            seed_everything(stable_seed(key, pass_id))

            torch.cuda.reset_peak_memory_stats()

            load_result = set_peft_model_state_dict(
                model,
                default_weights.copy(),
                adapter_name="default",
            )

            model = FastLanguageModel.for_training(model)

            puzzle_ds = arc_test_set.change_keys([key])

            train_ds = puzzle_ds.augment(n=16, shfl_keys=True, seed=cfg["train_seed"])
            train_ds = train_ds.cut_to_len(formatter=formatter, name="text", max_len=max_seq_length)

            with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):

                trainer = UnslothFixedTrainer(
                    model=model,
                    tokenizer=tokenizer,
                    data_collator=collator,
                    train_dataset=Dataset.from_list(train_ds.as_list(formatter)),
                    dataset_text_field="text",
                    max_seq_length=max_seq_length,
                    args=UnslothTrainingArguments(**train_args),
                )
                trainer.add_callback(DeadlineCallback(end_time))

                stats = trainer.train()

                model = trainer.accelerator.unwrap_model(model, keep_fp32_wrapper=False)

                del trainer

            model = FastLanguageModel.for_inference(model)

            gc.collect()
            torch.cuda.empty_cache()

            memory_allocated = torch.cuda.max_memory_allocated() // 1024**2
            print(f"[Rank {rank}] allocated {memory_allocated}MB for training")

            torch.cuda.reset_peak_memory_stats()

            print(f"[Rank {rank}] training stats for puzzle {key}: {stats}")

            puzzle_ds_multi = puzzle_ds.split_multi_replies()

            eval_ds = puzzle_ds_multi.augment(n=2, seed=cfg["eval_seed"])
            eval_ds = eval_ds.cut_to_len(formatter=formatter, name="input", max_len=max_seq_length-max_new_tokens)

            test_id_to_subkeys = defaultdict(list)
            for subkey in sorted(eval_ds.keys):
                test_id = subkey.split(".")[0].split("_")[1]
                test_id_to_subkeys[test_id].append(subkey)

            batches = []
            for test_id, subkeys in test_id_to_subkeys.items():
                # 0: permute x 2
                # 4: rot90.rot90.permute x 2
                batch = []
                for offset in [0, 4]:
                    batch.extend(subkeys[offset:offset+2])
                batches.append(batch)
                # 2: permute.rot90 x 2
                # 6: rot90.rot90.rot90.permute x 2
                batch = []
                for offset in [2, 6]:
                    batch.extend(subkeys[offset:offset+2])
                batches.append(batch)
            for test_id, subkeys in test_id_to_subkeys.items():
                # 8: transpose.permute x 2
                # 12: transpose.rot90.rot90.permute x 2
                batch = []
                for offset in [8, 12]:
                    batch.extend(subkeys[offset:offset+2])
                batches.append(batch)
                # 10: transpose.rot90.permute x 2
                # 14: transpose.rot90.rot90.rot90.permute x 2
                batch = []
                for offset in [10, 14]:
                    batch.extend(subkeys[offset:offset+2])
                batches.append(batch)

            with torch.inference_mode():

                known_scores = {}

                for subkeys in batches:

                    spend_time = time.time() - start_time
                    if spend_time > 1200 or time.time() > end_time:
                        print(f"[Rank {rank}] timeout after {spend_time:.1f}s for puzzle {key}")
                        timed_out = True
                        break

                    print(f"[Rank {rank}] decoding {subkeys}")

                    tokens = []
                    for subkey in subkeys:
                        data = eval_ds.get(subkey, formatter)
                        tokens.append(tokenizer.encode(data["input"]))

                    dfs_result = inference_turbo_dfs(model, tokens, max_new_tokens, max_score, end_time)

                    for subkey_id, scored_beams in dfs_result:

                        subkey = subkeys[subkey_id]
                        bk = subkey.split(".")[0]
                        decoded_result = []

                        for beam_score, tokens in scored_beams:

                            # [ours] never keep scoring far past the global deadline
                            if time.time() > end_time + 60:
                                break

                            array = formatter.convert_tokens_to_array(tokens)
                            if array is None:
                                continue

                            solution = puzzle_ds_multi.invert_mod(array, subkey, inv_perm=True)

                            grid_id = (bk, tuple(map(tuple, solution)))

                            if grid_id in known_scores:
                                augmented_scores = known_scores[grid_id]
                            else:
                                print(f"[Rank {rank}] scoring {subkey} #{len(decoded_result)}")
                                aug_dataset = ArcDataset(
                                    keys=[bk],
                                    queries={bk: puzzle_ds_multi.queries.get(bk)},
                                    replies={bk: [solution.tolist()]},
                                )
                                # [ours] stable seed instead of hash(bk) (PYTHONHASHSEED dependent)
                                aug_dataset = aug_dataset.augment(seed=stable_seed(bk))
                                aug_dataset = aug_dataset.cut_to_len(formatter=formatter, name="input", max_len=max_seq_length-max_new_tokens)
                                aug_queries = []
                                aug_answers = []
                                for augmented_sample in aug_dataset.as_list(formatter):
                                    aug_queries.append(augmented_sample["input"])
                                    aug_answers.append(augmented_sample["reply"])
                                augmented_scores1 = calc_scores(aug_queries[:4], aug_answers[:4], tokenizer, model)
                                augmented_scores2 = calc_scores(aug_queries[4:], aug_answers[4:], tokenizer, model)
                                augmented_scores = augmented_scores1 + augmented_scores2
                                known_scores[grid_id] = augmented_scores

                            decoded_result.append({
                                "beam_score": beam_score,
                                "score_aug": augmented_scores,
                                "solution": solution,
                            })

                        if len(decoded_result):
                            _atomic_pickle_bz2(decoded_result, os.path.join(dir_outputs, subkey + cfg["suffix"]))
                            n_written[bk] += 1

            consecutive_failures = 0

        except Exception:
            failed = True
            consecutive_failures += 1
            print(f"[Rank {rank}] ERROR on {key} (pass {pass_id}):\n{traceback.format_exc()}")
            trainer = None  # drop a half-built trainer (optimizer state) before the next task
            try:
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass

        try:
            memory_allocated = torch.cuda.max_memory_allocated() // 1024**2
            print(f"[Rank {rank}] allocated {memory_allocated}MB for inference")
        except Exception:
            pass

        spend_time = time.time() - start_time
        n_tests = len(arc_test_set.queries[key]["test"])
        missing = [i for i in range(n_tests) if n_written.get(f"{key}_{i}", 0) == 0]
        print(f"[Rank {rank}] finished {key} in {spend_time:.1f}s (pass {pass_id}, failed={failed}, "
              f"timed_out={timed_out}, tests_without_candidates={missing})")

        if pass_id == 0 and retry_queue is not None and (failed or (missing and not timed_out)):
            retry_queue.put((key, spend_time))

        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            print(f"[Rank {rank}] {consecutive_failures} consecutive failures, worker exits")
            break

# %% cell 6
# %%writefile starter.py
import os
import sys
import time
import json
import argparse
import traceback
import torch
import torch.multiprocessing as mp


def local_worker(rank, queue, retry_queue, end_time):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    torch.set_default_device("cpu")

    # Fix Unsloth patching issue: ranks import unsloth one after the other.
    # [ours] bounded wait, so one rank dying during import cannot block the others forever.
    marker_dir = os.getenv("ARC_MARKER_DIR", "/kaggle")
    if rank > 0:
        waited = 0
        while not os.path.exists(f"{marker_dir}/worker{rank-1}") and waited < 1800:
            time.sleep(5)
            waited += 5

    try:
        from arc_solver import worker
    finally:
        with open(f"{marker_dir}/worker{rank}", "w") as f:
            f.write("Ok")

    print(f"[Rank {rank}] start!", flush=True)

    try:
        worker(rank, queue, end_time, retry_queue)
    except Exception:
        print(f"[Rank {rank}] worker crashed:\n{traceback.format_exc()}", flush=True)

    print(f"[Rank {rank}] done!", flush=True)


def task_cost(task):
    """[ours] Cheap proxy for per-task GPU time: cells seen in training (+ newlines) plus
    decode length per test output. Used to run cheap tasks first."""
    def cells(g):
        return len(g) * (len(g[0]) + 1) if len(g) else 0
    train = sum(cells(p["input"]) + cells(p["output"]) for p in task["train"])
    test = sum(cells(t["input"]) for t in task["test"])
    return train + test + len(task["test"]) * 2 * max((cells(t["input"]) for t in task["test"]), default=0)


def order_keys(data, mode):
    keys = sorted(data.keys())
    if mode == "cheap_first":
        keys = sorted(keys, key=lambda k: (task_cost(data[k]), k))
    return keys


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--end-time", type=float, default=0.0)
    parser.add_argument("--hard-deadline", type=float, default=0.0)
    args = parser.parse_args()

    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")

    if rerun_mode:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"
    else:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"

    test_path = os.getenv("ARC_TEST_PATH") or test_path  # [ours] override for local tests

    with open(test_path, "r") as f:
        data = json.load(f)

    order = os.getenv("ARC_ORDER", "cheap_first")
    eval_subset = os.getenv("ARC_EVAL_SUBSET", "smoke")   # commit mode only: smoke | all
    num_gpus = int(os.getenv("ARC_NUM_GPUS", "4"))

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    queue = manager.Queue()
    retry_queue = manager.Queue()

    n_put = 0
    for key in order_keys(data, order):
        if not rerun_mode and eval_subset == "smoke":
            if key not in ["0934a4d8", "36a08778", "981571dc", "aa4ec2a5"]:
                continue
        queue.put(key)
        n_put += 1
    print(f"[starter] queued {n_put} tasks (order={order}, gpus={num_gpus})", flush=True)

    # [ours] independent processes instead of mp.spawn: with mp.spawn one crashed rank
    # terminates all the other ranks; here the survivors keep draining the queue.
    procs = []
    for rank in range(num_gpus):
        p = ctx.Process(target=local_worker, args=(rank, queue, retry_queue, args.end_time))
        p.start()
        procs.append(p)

    hard_deadline = args.hard_deadline or (args.end_time + 180 if args.end_time else 0)
    while any(p.is_alive() for p in procs):
        for p in procs:
            p.join(timeout=5)
        if hard_deadline and time.time() > hard_deadline:
            print("[starter] hard deadline reached, terminating workers", flush=True)
            for p in procs:
                if p.is_alive():
                    p.terminate()
            time.sleep(10)
            for p in procs:
                if p.is_alive():
                    p.kill()
            break

    print(f"[starter] exit codes: {[p.exitcode for p in procs]}", flush=True)
    try:
        manager.shutdown()
    except Exception:
        pass

# %% cell 7
# %%writefile arc_dsl.py
"""Small, conservative CPU program search for ARC tasks.

Only programs that reproduce EVERY train pair exactly are kept. Each kept program is
applied to the test inputs, and the distinct predictions are returned in order of
program simplicity. The notebook uses these ONLY to fill empty or duplicate attempt
slots; they never replace a model attempt_1.

Pure numpy + stdlib, no external dependencies. Safe to run in a background process.

Usage as a script:
    python arc_dsl.py CHALLENGES_JSON OUT_JSON [--time-limit SECONDS] [--per-task SECONDS]
writes {task_id: [[cand_grid, ...] per test input]} plus a "_programs" map with the names
of the programs that produced each candidate.
"""
import json
import sys
import time
import signal
import itertools
import numpy as np

# ----------------------------------------------------------------------------- helpers

MAX_SIDE = 30


def _valid(g):
    return (
        isinstance(g, np.ndarray)
        and g.ndim == 2
        and 1 <= g.shape[0] <= MAX_SIDE
        and 1 <= g.shape[1] <= MAX_SIDE
        and g.dtype.kind in "iu"
        and g.min() >= 0
        and g.max() <= 9
    )


DIHEDRAL = [
    ("id", lambda g: g),
    ("rot90", lambda g: np.rot90(g, 1)),
    ("rot180", lambda g: np.rot90(g, 2)),
    ("rot270", lambda g: np.rot90(g, 3)),
    ("flipud", lambda g: g[::-1, :]),
    ("fliplr", lambda g: g[:, ::-1]),
    ("transpose", lambda g: g.T),
    ("antitranspose", lambda g: np.rot90(g, 2).T),
]


def most_common_color(g):
    return int(np.bincount(g.ravel(), minlength=10).argmax())


def components(g, bg, diag, multicolor):
    """Connected components of non-bg cells. Returns list of (mask, colors_set)."""
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    if diag:
        nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for sy in range(h):
        for sx in range(w):
            if seen[sy, sx] or g[sy, sx] == bg:
                continue
            col = g[sy, sx]
            stack = [(sy, sx)]
            seen[sy, sx] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in nbrs:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx]:
                        v = g[ny, nx]
                        if v == bg:
                            continue
                        if not multicolor and v != col:
                            continue
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            mask = np.zeros((h, w), dtype=bool)
            ys, xs = zip(*cells)
            mask[list(ys), list(xs)] = True
            comps.append(mask)
    return comps


def bbox(mask):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


# ----------------------------------------------------------------------------- base transforms
# Each "family" is a generator taking the list of train (inp, out) pairs and yielding
# (name, complexity, fn) with fn: np.ndarray -> np.ndarray | None.  Params that must be
# learned from data are derived from the train pairs; everything is verified afterwards.


def fam_dihedral(pairs):
    for i, (name, f) in enumerate(DIHEDRAL):
        yield name, 1 + (i > 0), f


def _ratio(pairs, num):
    """Consistent integer scale factors (out = in * k) or (in = out * k) if num=False."""
    rs = set()
    for a, b in pairs:
        if num:
            if b.shape[0] % a.shape[0] or b.shape[1] % a.shape[1]:
                return None
            rs.add((b.shape[0] // a.shape[0], b.shape[1] // a.shape[1]))
        else:
            if a.shape[0] % b.shape[0] or a.shape[1] % b.shape[1]:
                return None
            rs.add((a.shape[0] // b.shape[0], a.shape[1] // b.shape[1]))
    return rs.pop() if len(rs) == 1 else None


def fam_upscale(pairs):
    r = _ratio(pairs, True)
    if r and r != (1, 1):
        fy, fx = r
        yield f"upscale{fy}x{fx}", 2, lambda g, fy=fy, fx=fx: np.kron(g, np.ones((fy, fx), dtype=g.dtype))


def fam_downscale(pairs):
    r = _ratio(pairs, False)
    if not r or r == (1, 1):
        return
    fy, fx = r

    def mk(mode):
        def f(g):
            h, w = g.shape
            if h % fy or w % fx:
                return None
            blocks = g.reshape(h // fy, fy, w // fx, fx).transpose(0, 2, 1, 3).reshape(h // fy, w // fx, fy * fx)
            out = np.zeros((h // fy, w // fx), dtype=g.dtype)
            bg = most_common_color(g)
            for y in range(out.shape[0]):
                for x in range(out.shape[1]):
                    b = blocks[y, x]
                    if mode == "uniform":
                        if (b != b[0]).any():
                            return None
                        out[y, x] = b[0]
                    elif mode == "majority":
                        cnt = np.bincount(b, minlength=10)
                        if (cnt == cnt.max()).sum() > 1:
                            return None
                        out[y, x] = cnt.argmax()
                    elif mode == "anyfg":
                        fg = b[b != bg]
                        if len(fg) == 0:
                            out[y, x] = bg
                        else:
                            cnt = np.bincount(fg, minlength=10)
                            if (cnt == cnt.max()).sum() > 1:
                                return None
                            out[y, x] = cnt.argmax()
            return out
        return f

    for mode in ("uniform", "majority", "anyfg"):
        yield f"downscale{fy}x{fx}_{mode}", 3, mk(mode)


def fam_tile_dihedral(pairs):
    """out is an (a x b) arrangement of dihedral copies of the input (tiling / mirroring)."""
    r = _ratio(pairs, True)
    if not r or r == (1, 1):
        return
    a, b = r
    if a * b > 16:
        return
    layout = []
    for i in range(a):
        for j in range(b):
            ok_names = None
            for inp, out in pairs:
                h, w = inp.shape
                block = out[i * h:(i + 1) * h, j * w:(j + 1) * w]
                names = {n for n, f in DIHEDRAL if f(inp).shape == block.shape and np.array_equal(f(inp), block)}
                ok_names = names if ok_names is None else ok_names & names
                if not ok_names:
                    return
            # deterministic pick: first in DIHEDRAL order
            layout.append(next(n for n, _ in DIHEDRAL if n in ok_names))
    fns = dict(DIHEDRAL)

    def f(g, layout=tuple(layout)):
        h, w = g.shape
        out = np.zeros((a * h, b * w), dtype=g.dtype)
        for idx, n in enumerate(layout):
            t = fns[n](g)
            if t.shape != (h, w):
                return None
            i, j = divmod(idx, b)
            out[i * h:(i + 1) * h, j * w:(j + 1) * w] = t
        return out

    yield f"tile{a}x{b}[" + ",".join(layout) + "]", 3, f


def fam_fractal(pairs):
    """out = kron(mask(in), in): the classic self-similar tiling."""
    ok = all(b.shape == (a.shape[0] ** 2, a.shape[1] ** 2) for a, b in pairs)
    if not ok:
        return

    def mk(inv):
        def f(g):
            if g.shape[0] ** 2 > MAX_SIDE or g.shape[1] ** 2 > MAX_SIDE:
                return None
            bg = 0
            m = (g != bg) if not inv else (g == bg)
            return np.kron(m.astype(g.dtype), g)
        return f

    yield "fractal", 3, mk(False)
    yield "fractal_inv", 3, mk(True)


def _crop(g, bb):
    if bb is None:
        return None
    y0, y1, x0, x1 = bb
    return g[y0:y1, x0:x1]


def fam_crop(pairs):
    if all(a.shape == b.shape for a, b in pairs):
        return
    for bgmode in ("zero", "common"):
        def f(g, bgmode=bgmode):
            bg = 0 if bgmode == "zero" else most_common_color(g)
            return _crop(g, bbox(g != bg))
        yield f"crop_nonbg_{bgmode}", 2, f
    for c in range(10):
        if not all((a == c).any() for a, _ in pairs):
            continue

        def fc(g, c=c):
            return _crop(g, bbox(g == c))

        def fci(g, c=c):
            bb = bbox(g == c)
            if bb is None:
                return None
            y0, y1, x0, x1 = bb
            if y1 - y0 < 3 or x1 - x0 < 3:
                return None
            return g[y0 + 1:y1 - 1, x0 + 1:x1 - 1]

        yield f"crop_color{c}", 3, fc
        yield f"crop_color{c}_inner", 3, fci


_OBJ_KEYS = {
    "largest": lambda g, m: m.sum(),
    "smallest": lambda g, m: -m.sum(),
    "bbox_largest": lambda g, m: _bbarea(m),
    "bbox_smallest": lambda g, m: -_bbarea(m),
    "most_colors": lambda g, m: len(np.unique(g[m])),
    "fewest_colors": lambda g, m: -len(np.unique(g[m])),
    "topmost": lambda g, m: -bbox(m)[0],
    "bottommost": lambda g, m: bbox(m)[1],
    "leftmost": lambda g, m: -bbox(m)[2],
    "rightmost": lambda g, m: bbox(m)[3],
}


def _bbarea(m):
    y0, y1, x0, x1 = bbox(m)
    return (y1 - y0) * (x1 - x0)


def fam_objects(pairs):
    if all(a.shape == b.shape for a, b in pairs):
        return
    for bgmode, diag, multi in itertools.product(("zero", "common"), (False, True), (True, False)):
        for kname, kfn in _OBJ_KEYS.items():
            for outmode in ("bbox", "masked"):
                def f(g, bgmode=bgmode, diag=diag, multi=multi, kfn=kfn, outmode=outmode):
                    bg = 0 if bgmode == "zero" else most_common_color(g)
                    comps = components(g, bg, diag, multi)
                    if len(comps) < 2 or len(comps) > 60:
                        return None
                    vals = [kfn(g, m) for m in comps]
                    best = max(vals)
                    if sum(1 for v in vals if v == best) != 1:
                        return None  # ambiguous: refuse rather than guess
                    m = comps[vals.index(best)]
                    y0, y1, x0, x1 = bbox(m)
                    if outmode == "bbox":
                        return g[y0:y1, x0:x1].copy()
                    out = np.full((y1 - y0, x1 - x0), bg, dtype=g.dtype)
                    sub = m[y0:y1, x0:x1]
                    out[sub] = g[y0:y1, x0:x1][sub]
                    return out
                yield f"object[{bgmode},{'8' if diag else '4'}conn,{'multi' if multi else 'mono'},{kname},{outmode}]", 4, f
        # object whose colour is unique among objects (e.g. the odd one out)
        def fu(g, bgmode=bgmode, diag=diag, multi=multi):
            bg = 0 if bgmode == "zero" else most_common_color(g)
            comps = components(g, bg, diag, multi)
            if len(comps) < 3 or len(comps) > 60:
                return None
            sigs = [tuple(sorted(np.unique(g[m]).tolist())) for m in comps]
            cnt = {s: sigs.count(s) for s in sigs}
            uniq = [i for i, s in enumerate(sigs) if cnt[s] == 1]
            if len(uniq) != 1:
                return None
            y0, y1, x0, x1 = bbox(comps[uniq[0]])
            return g[y0:y1, x0:x1].copy()
        yield f"object[{bgmode},{'8' if diag else '4'}conn,{'multi' if multi else 'mono'},unique_colors,bbox]", 4, fu

        def fs(g, bgmode=bgmode, diag=diag, multi=multi):
            bg = 0 if bgmode == "zero" else most_common_color(g)
            comps = components(g, bg, diag, multi)
            if len(comps) < 3 or len(comps) > 60:
                return None
            crops = []
            for m in comps:
                y0, y1, x0, x1 = bbox(m)
                crops.append((m[y0:y1, x0:x1].tobytes(), m[y0:y1, x0:x1].shape))
            cnt = {s: crops.count(s) for s in crops}
            uniq = [i for i, s in enumerate(crops) if cnt[s] == 1]
            if len(uniq) != 1:
                return None
            y0, y1, x0, x1 = bbox(comps[uniq[0]])
            return g[y0:y1, x0:x1].copy()
        yield f"object[{bgmode},{'8' if diag else '4'}conn,{'multi' if multi else 'mono'},unique_shape,bbox]", 4, fs


def _split_parts(g, n, axis):
    """Split g into n equal parts along axis, allowing 1-wide uniform separator lines."""
    size = g.shape[axis]
    for sep in (0, 1):
        tot = size - sep * (n - 1)
        if tot <= 0 or tot % n:
            continue
        p = tot // n
        parts = []
        seps = []
        for i in range(n):
            s = i * (p + sep)
            parts.append(g[s:s + p] if axis == 0 else g[:, s:s + p])
            if sep and i < n - 1:
                line = g[s + p] if axis == 0 else g[:, s + p]
                seps.append(line)
        if sep:
            # separators must be uniform lines of one colour
            if not all((ln == ln[0]).all() for ln in seps):
                continue
        return parts
    return None


_BOOL_OPS = {
    "and": lambda a, b: a & b,
    "or": lambda a, b: a | b,
    "xor": lambda a, b: a ^ b,
    "nor": lambda a, b: ~(a | b),
    "a_not_b": lambda a, b: a & ~b,
    "b_not_a": lambda a, b: b & ~a,
    "nand": lambda a, b: ~(a & b),
}


def fam_parts(pairs):
    # Candidate split: n parts along an axis such that part shape == output shape
    for n, axis in itertools.product((2, 3, 4), (0, 1)):
        good = True
        for a, b in pairs:
            ps = _split_parts(a, n, axis)
            if ps is None or ps[0].shape != b.shape:
                good = False
                break
        if not good:
            continue
        # boolean ops on two parts: learn (true_colour, false_colour)
        if n == 2:
            for opname, op in _BOOL_OPS.items():
                tc = fc = None
                ok = True
                for a, b in pairs:
                    pa, pb = _split_parts(a, 2, axis)
                    bga = 0
                    m = op(pa != bga, pb != bga)
                    tv = np.unique(b[m])
                    fv = np.unique(b[~m])
                    if len(tv) > 1 or len(fv) > 1:
                        ok = False
                        break
                    if len(tv):
                        if tc is not None and tc != tv[0]:
                            ok = False
                            break
                        tc = int(tv[0])
                    if len(fv):
                        if fc is not None and fc != fv[0]:
                            ok = False
                            break
                        fc = int(fv[0])
                if not ok or tc is None or fc is None or tc == fc:
                    continue

                def f(g, op=op, tc=tc, fc=fc, axis=axis):
                    ps = _split_parts(g, 2, axis)
                    if ps is None:
                        return None
                    m = op(ps[0] != 0, ps[1] != 0)
                    return np.where(m, tc, fc).astype(g.dtype)
                yield f"parts2_axis{axis}_{opname}_t{tc}_f{fc}", 3, f
        # overlay of parts with a priority order (non-zero cells of earlier parts win)
        for order in itertools.permutations(range(n)):
            def fo(g, order=order, n=n, axis=axis):
                ps = _split_parts(g, n, axis)
                if ps is None:
                    return None
                out = np.zeros_like(ps[0])
                for i in reversed(order):
                    out = np.where(ps[i] != 0, ps[i], out)
                return out
            yield f"parts{n}_axis{axis}_overlay{''.join(map(str, order))}", 4, fo


def fam_symmetry_complete(pairs):
    """Fill cells of a 'hole' colour using the grid's own mirror/rotational symmetry."""
    if not all(a.shape == b.shape for a, b in pairs):
        return
    diffs = set()
    for a, b in pairs:
        d = a != b
        if not d.any():
            return
        vals = np.unique(a[d])
        if len(vals) != 1:
            return
        diffs.add(int(vals[0]))
    if len(diffs) != 1:
        return
    hole = diffs.pop()

    def f(g, hole=hole):
        out = g.copy()
        m = out == hole
        if not m.any():
            return None
        cands = [g[::-1, :], g[:, ::-1], np.rot90(g, 2)]
        if g.shape[0] == g.shape[1]:
            cands += [g.T, np.rot90(g, 2).T, np.rot90(g, 1), np.rot90(g, 3)]
        for _ in range(3):
            for c in cands:
                fill = m & (c != hole)
                out[fill] = c[fill]
                m = out == hole
            cands = [out[::-1, :], out[:, ::-1], np.rot90(out, 2)] + (
                [out.T, np.rot90(out, 2).T, np.rot90(out, 1), np.rot90(out, 3)] if g.shape[0] == g.shape[1] else [])
        if m.any():
            return None
        return out

    yield f"symfill{hole}", 4, f


FAMILIES = [
    fam_dihedral,
    fam_upscale,
    fam_downscale,
    fam_tile_dihedral,
    fam_fractal,
    fam_crop,
    fam_objects,
    fam_parts,
    fam_symmetry_complete,
]


# ----------------------------------------------------------------------------- composition + verification

def _learn_cmap(preds, outs):
    """Colour mapping m with m[pred] == out, consistent over all pairs. None if impossible."""
    m = {}
    for p, o in zip(preds, outs):
        if p.shape != o.shape:
            return None
        for pc, oc in zip(p.ravel().tolist(), o.ravel().tolist()):
            if m.setdefault(pc, oc) != oc:
                return None
    return m


def _apply_cmap(g, m):
    vals = np.unique(g).tolist()
    if any(v not in m for v in vals):
        return None  # unseen colour: refuse
    lut = np.arange(10)
    for k, v in m.items():
        lut[k] = v
    return lut[g]


def _safe(fn, g):
    try:
        r = fn(g)
    except Exception:
        return None
    if r is None:
        return None
    r = np.asarray(r)
    if r.dtype.kind not in "iu":
        return None
    r = r.astype(np.int64)
    return r if _valid(r) else None


def find_programs(task, deadline=None):
    """Return list of (name, complexity, fn) reproducing all train pairs, simplest first."""
    pairs = [(np.array(p["input"], dtype=np.int64), np.array(p["output"], dtype=np.int64)) for p in task["train"]]
    found = []
    for fam in FAMILIES:
        try:
            gen = list(fam(pairs))
        except Exception:
            continue
        for name, cx, fn in gen:
            if deadline is not None and time.time() > deadline:
                return sorted(found, key=lambda x: x[1])
            preds = [_safe(fn, a) for a, _ in pairs]
            if any(p is None for p in preds):
                continue
            if all(p.shape == b.shape and np.array_equal(p, b) for p, (_, b) in zip(preds, pairs)):
                found.append((name, cx, fn))
                continue
            # geometric post-transform and/or colour map on top of the base program
            for gname, gf in DIHEDRAL:
                gp = [gf(p) for p in preds]
                if gname != "id" and all(p.shape == b.shape and np.array_equal(p, b) for p, (_, b) in zip(gp, pairs)):
                    found.append((f"{name}|{gname}", cx + 1, (lambda g, fn=fn, gf=gf: (lambda r: None if r is None else gf(r))(_safe(fn, g)))))
                    continue
                m = _learn_cmap(gp, [b for _, b in pairs])
                if m is None or all(k == v for k, v in m.items()):
                    continue
                found.append((f"{name}|{gname}|cmap{sorted(m.items())}", cx + 2,
                              (lambda g, fn=fn, gf=gf, m=m: (lambda r: None if r is None else _apply_cmap(gf(r), m))(_safe(fn, g)))))
    found.sort(key=lambda x: x[1])  # stable: family order then complexity
    return found


def solve_task(task, max_candidates=2, per_task_seconds=10.0):
    """Returns (predictions, program_names): predictions[t] is a list (<= max_candidates)
    of distinct grids (lists) for test input t, possibly empty."""
    deadline = time.time() + per_task_seconds
    progs = find_programs(task, deadline=deadline)
    preds, names = [], []
    for t in task["test"]:
        g = np.array(t["input"], dtype=np.int64)
        seen, cands, cn = set(), [], []
        for name, _, fn in progs:
            r = _safe(fn, g)
            if r is None:
                continue
            key = (r.shape, r.tobytes())
            if key in seen:
                continue
            seen.add(key)
            cands.append(r.tolist())
            cn.append(name)
            if len(cands) >= max_candidates:
                break
        preds.append(cands)
        names.append(cn)
    return preds, names


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def solve_all(challenges, time_limit=1800.0, per_task=10.0, verbose=False):
    t_end = time.time() + time_limit
    out, progs = {}, {}
    use_alarm = hasattr(signal, "SIGALRM")
    if use_alarm:
        signal.signal(signal.SIGALRM, _alarm)
    for tid in sorted(challenges):
        if time.time() > t_end:
            break
        try:
            if use_alarm:
                signal.alarm(int(per_task * 2) + 1)
            p, n = solve_task(challenges[tid], per_task_seconds=per_task)
        except BaseException as e:  # includes _Timeout
            if isinstance(e, KeyboardInterrupt):
                raise
            p, n = [[] for _ in challenges[tid]["test"]], [[] for _ in challenges[tid]["test"]]
        finally:
            if use_alarm:
                signal.alarm(0)
        if any(len(x) for x in p):
            out[tid] = p
            progs[tid] = n
            if verbose:
                print(f"[dsl] {tid}: {n}", flush=True)
    return out, progs


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("challenges")
    ap.add_argument("out")
    ap.add_argument("--time-limit", type=float, default=1800.0)
    ap.add_argument("--per-task", type=float, default=10.0)
    args = ap.parse_args()
    with open(args.challenges) as f:
        ch = json.load(f)
    t0 = time.time()
    preds, progs = solve_all(ch, time_limit=args.time_limit, per_task=args.per_task, verbose=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"predictions": preds, "programs": progs}, f)
    import os
    os.replace(tmp, args.out)
    print(f"[dsl] wrote {len(preds)} tasks with candidates in {time.time() - t0:.1f}s", flush=True)

# %% cell 8
# %%writefile arc_submit.py
"""[ours] Submission assembly: attempt selection, verified-program fill, format validation.

Pure json + numpy (no torch / transformers), so it can be unit-tested on CPU and can
always run, even if every GPU worker crashed.

Slot policy for every test output (task_id, i):
  1. model candidates, in the base selection order (score_kgmon), de-duplicated and
     sanitised -> attempt_1, attempt_2   (the model's attempt_1 is NEVER replaced)
  2. still-empty slots are filled with CPU program-search candidates (programs that
     reproduce every train pair), skipping grids already used
  3. anything still empty gets a cheap guess (the test input itself), then [[0]]
"""
import json
import os

import numpy as np

PLACEHOLDER = [[0]]


def sanitize_grid(g):
    """Return g as a list of lists of ints in 0..9, rectangular, 1..30 per side; else None."""
    try:
        a = np.asarray(g)
        if a.dtype == object or a.ndim != 2:
            return None
        if not (1 <= a.shape[0] <= 30 and 1 <= a.shape[1] <= 30):
            return None
        if a.dtype.kind == "f":
            if not np.all(np.equal(np.mod(a, 1), 0)):
                return None
        elif a.dtype.kind not in "iub":
            return None
        a = a.astype(np.int64)
        if a.min() < 0 or a.max() > 9:
            return None
        return a.tolist()
    except Exception:
        return None


def _key(g):
    return json.dumps(g, separators=(",", ":"))


def choose_attempts(model_cands, dsl_cands=(), fallback_cands=(), n=2):
    """Pick n distinct grids. Returns (attempts, sources)."""
    chosen, sources, seen = [], [], set()
    for src, cands in (("model", model_cands), ("dsl", dsl_cands), ("fallback", fallback_cands)):
        for c in cands or ():
            if len(chosen) >= n:
                break
            g = sanitize_grid(c)
            if g is None:
                continue
            k = _key(g)
            if k in seen:
                continue
            seen.add(k)
            chosen.append(g)
            sources.append(src)
    while len(chosen) < n:
        chosen.append(PLACEHOLDER)
        sources.append("placeholder")
    return chosen, sources


def build_submission(challenges, model_ranked=None, dsl_preds=None):
    """challenges: {task_id: task}; model_ranked: {f"{tid}_{i}": [grid, ...]} ranked;
    dsl_preds: {tid: [[grid, ...] per test input]}. Returns (submission, stats)."""
    model_ranked = model_ranked or {}
    dsl_preds = dsl_preds or {}
    submission, stats = {}, {}
    for tid in sorted(challenges):
        tests = challenges[tid].get("test", [])
        entries = []
        for i, t in enumerate(tests):
            model_c = model_ranked.get(f"{tid}_{i}", [])
            try:
                dsl_c = dsl_preds.get(tid, [])[i]
            except Exception:
                dsl_c = []
            fb = [t.get("input")] if isinstance(t, dict) else []
            attempts, sources = choose_attempts(model_c, dsl_c, fb)
            entries.append({"attempt_1": attempts[0], "attempt_2": attempts[1]})
            for slot, s in zip(("attempt_1", "attempt_2"), sources):
                stats[f"{slot}:{s}"] = stats.get(f"{slot}:{s}", 0) + 1
        if not entries:  # defensive: a task without test inputs still needs an entry
            entries = [{"attempt_1": PLACEHOLDER, "attempt_2": PLACEHOLDER}]
        submission[tid] = entries
    return submission, stats


def validate_submission(submission, challenges):
    """Return a list of problems (empty list == valid)."""
    problems = []
    for tid, task in challenges.items():
        if tid not in submission:
            problems.append(f"missing task {tid}")
            continue
        entries = submission[tid]
        n = max(1, len(task.get("test", [])))
        if not isinstance(entries, list) or len(entries) != n:
            problems.append(f"{tid}: expected {n} entries")
            continue
        for i, e in enumerate(entries):
            for slot in ("attempt_1", "attempt_2"):
                if not isinstance(e, dict) or slot not in e:
                    problems.append(f"{tid}[{i}] missing {slot}")
                elif sanitize_grid(e[slot]) != e[slot]:
                    problems.append(f"{tid}[{i}] {slot} is not a valid grid")
    for tid in submission:
        if tid not in challenges:
            problems.append(f"extra task {tid}")
    return problems


def write_json_atomic(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def load_dsl_predictions(path):
    try:
        with open(path) as f:
            d = json.load(f)
        return d.get("predictions", {}) or {}
    except Exception as e:
        print(f"*** No program-search predictions ({e!r})")
        return {}


def write_placeholder(challenges_path, out_path="submission.json"):
    """Valid submission before any GPU work (test input as guess, [[0]] otherwise)."""
    with open(challenges_path) as f:
        challenges = json.load(f)
    sub, _ = build_submission(challenges)
    assert not validate_submission(sub, challenges)
    write_json_atomic(sub, out_path)
    return sub


def score_submission(submission, solutions):
    """Official metric: per task, mean over test outputs of any-attempt exact match."""
    total = 0.0
    for tid, sols in solutions.items():
        for i, s in enumerate(sols):
            e = submission.get(tid, [])
            if i < len(e) and (e[i].get("attempt_1") == s or e[i].get("attempt_2") == s):
                total += 1.0 / len(sols)
    return total

# %% cell 9
# [ours] 1) write a valid placeholder submission.json immediately, so a later failure
#    still leaves a well-formed file; 2) start the CPU program search in the background
#    (niced, uses otherwise idle CPU cores while the GPUs train).
import sys, subprocess, traceback
from arc_submit import write_placeholder

try:
    write_placeholder(TEST_PATH, "submission.json")
    print("placeholder submission.json written")
except Exception:
    traceback.print_exc()

dsl_proc = None
try:
    dsl_cmd = [sys.executable, "arc_dsl.py", TEST_PATH, "dsl_predictions.json",
               "--time-limit", str(DSL_TIME_LIMIT), "--per-task", "10"]
    dsl_log = open("dsl.log", "w")
    try:
        dsl_proc = subprocess.Popen(["nice", "-n", "19"] + dsl_cmd, stdout=dsl_log, stderr=subprocess.STDOUT)
    except FileNotFoundError:
        dsl_proc = subprocess.Popen(dsl_cmd, stdout=dsl_log, stderr=subprocess.STDOUT)
    print("program search started, pid", dsl_proc.pid)
except Exception:
    traceback.print_exc()

# %% cell 10
# [ours] Same launch as the base (`!... python starter.py --end-time ...`) plus:
#  - PYTHONHASHSEED=0 for reproducible runs,
#  - the starter terminates its workers at end_time + 180 s,
#  - and this cell kills the whole process group at end_time + 240 s as a last resort,
#    so the final cell always gets to write submission.json within the 12 h limit.
import os, sys, shutil, signal, subprocess, threading, time

run_env = dict(
    os.environ,
    PYTHONHASHSEED="0",
    UNSLOTH_DISABLE_STATISTICS="1",
    TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas",
    OMP_NUM_THREADS="12",
    ARC_ORDER=ARC_ORDER,
    ARC_EVAL_SUBSET=ARC_EVAL_SUBSET,
)
python_bin = shutil.which("python") or sys.executable  # same interpreter as the base's `!python`
starter = subprocess.Popen(
    [python_bin, "-u", "starter.py", "--end-time", str(global_end_time),
     "--hard-deadline", str(global_end_time + 180)],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    env=run_env, start_new_session=True,
)

def _pump(stream):
    for line in stream:
        print(line, end="", flush=True)

pump = threading.Thread(target=_pump, args=(starter.stdout,), daemon=True)
pump.start()

kill_at = global_end_time + 240
while starter.poll() is None and time.time() < kill_at:
    time.sleep(10)
if starter.poll() is None:
    print("*** starter still running at the kill deadline: killing its process group")
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(starter.pid, sig)
        except Exception as e:
            print("killpg:", e)
        time.sleep(10)
pump.join(timeout=10)
print("starter exit code:", starter.poll())

# %% cell 11
import os
# [ours] Assemble submission.json. Every step is guarded: whatever fails, a valid file
# covering every test output with 2 attempts is written.
import json, traceback
from arc_submit import (build_submission, validate_submission, write_json_atomic,
                        load_dsl_predictions, score_submission)

with open(TEST_PATH) as f:
    challenges = json.load(f)

model_ranked, data, decoder = {}, None, None
try:
    from arc_loader import ArcDataset
    from arc_decoder import ArcDecoder
    data = ArcDataset.from_file(TEST_PATH)
    if not RERUN_MODE:
        data = data.load_replies(SOLUTIONS_PATH)
    decoder = ArcDecoder(data.split_multi_replies(), n_guesses=2)
    decoder.load_decoded_results("/kaggle/inference_outputs")
    # [ours v2] selector: score_full_probmul_3 (medvax reports 29.17 vs 28.33 for
    # score_kgmon on the 120-task eval); override with ARC_SELECTOR=score_kgmon.
    import arc_decoder as _ad
    _selector = getattr(_ad, os.environ.get("ARC_SELECTOR", "score_kgmon"), _ad.score_kgmon)
    print("*** selection algorithm:", _selector.__name__)
    model_ranked = decoder.run_selection_algo(_selector)
except Exception:
    traceback.print_exc()

_dsl = globals().get("dsl_proc")
if _dsl is not None and _dsl.poll() is None:
    print("*** program search still running: stopping it")
    try:
        _dsl.kill()
    except Exception:
        pass
dsl_preds = load_dsl_predictions("dsl_predictions.json")
print(f"*** program search produced candidates for {len(dsl_preds)} tasks")

submission, stats = build_submission(challenges, model_ranked, dsl_preds)
problems = validate_submission(submission, challenges)
print("*** slot sources:", json.dumps(stats, sort_keys=True))
if problems:
    print("*** VALIDATION PROBLEMS:", problems[:20])
    submission, _ = build_submission(challenges)  # last resort, valid by construction
write_json_atomic(submission, "submission.json")
print(f"*** wrote submission.json with {len(submission)} tasks")

if not RERUN_MODE:
    with open(SOLUTIONS_PATH) as f:
        solutions = json.load(f)
    model_only, _ = build_submission(challenges, model_ranked, None)
    print("*** Score (model only, all eval tasks):", score_submission(model_only, solutions))
    print("*** Score (final, all eval tasks):", score_submission(submission, solutions))
    try:
        decoder.benchmark_selection_algos()
        with open("submission.json", "r") as f:
            reload_submission = json.load(f)
        print("*** Reload score:", data.validate_submission(reload_submission))
    except Exception:
        traceback.print_exc()
    # keep the raw candidate pools for CPU-side selection experiments
    try:
        import shutil
        shutil.make_archive("inference_outputs", "gztar", "/kaggle/inference_outputs")
    except Exception:
        traceback.print_exc()
