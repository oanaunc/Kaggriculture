"""CPU stand-ins for unsloth / peft / datasets and a fake model, so the real worker loop,
result pickles, decoder selection and submission assembly can run end-to-end on CPU.

Fake "model": every task is the identity task (output == input), which is equivariant
under all NVARC augmentations; the fake DFS returns the (augmented) query grid as the
best beam plus a view-dependent wrong beam. Behaviour per task is scripted via SCRIPT.
"""
import sys
import types

import numpy as np

SPECIAL = [("<|im_start|>", 14), ("<|im_end|>", 15), ("assistant", 12), ("user", 11)]
INV = {v: k for k, v in SPECIAL}
INV[10] = "\n"

# task_id -> list of behaviours per attempt: "ok", "raise_train", "raise_dfs", "empty"
SCRIPT = {}
STATE = {"current": None, "attempts": {}, "callbacks_seen": 0, "stopped_by_deadline": 0}


class FakeTokenizer:
    def encode(self, text):
        ids, i = [], 0
        while i < len(text):
            for s, t in SPECIAL:
                if text.startswith(s, i):
                    ids.append(t)
                    i += len(s)
                    break
            else:
                c = text[i]
                ids.append(10 if c == "\n" else int(c))
                i += 1
        return ids

    def decode(self, ids):
        return "".join(INV.get(t, str(t)) for t in ids)

    def get_vocab(self):
        v = {str(d): d for d in range(10)}
        v["Ċ"] = 10
        v["<|im_end|>"] = 15
        return v


class FakeModel:
    device = "cpu"

    def named_parameters(self):
        return []


def _behaviour():
    tid = STATE["current"]
    n = STATE["attempts"].get(tid, 1)
    plan = SCRIPT.get(tid, ["ok"])
    return plan[min(n, len(plan)) - 1]


class _Control:
    should_training_stop = False


class FakeTrainer:
    def __init__(self, model=None, train_dataset=None, **kw):
        self.model = model
        self.train_dataset = train_dataset
        self.callbacks = []
        self.accelerator = types.SimpleNamespace(unwrap_model=lambda m, keep_fp32_wrapper=False: m)
        self.label_smoother = None

    def add_callback(self, cb):
        self.callbacks.append(cb)

    def train(self):
        tid = self.train_dataset[0]["key"].split(".")[0]
        STATE["current"] = tid
        STATE["attempts"][tid] = STATE["attempts"].get(tid, 0) + 1
        if _behaviour() == "raise_train":
            raise RuntimeError(f"CUDA out of memory (fake) on {tid}")
        control = _Control()
        for step in range(3):
            for cb in self.callbacks:
                STATE["callbacks_seen"] += 1
                cb.on_step_end(None, None, control)
            if control.should_training_stop:
                STATE["stopped_by_deadline"] += 1
                break
        return "fake-stats"


def _module(name):
    import importlib.machinery
    m = types.ModuleType(name)
    m.__spec__ = importlib.machinery.ModuleSpec(name, None)
    return m


def install_stubs():
    import transformers.trainer_callback  # noqa: F401  (import before peft is stubbed)
    unsloth = _module("unsloth")

    class FastLanguageModel:
        @staticmethod
        def from_pretrained(**kw):
            return FakeModel(), FakeTokenizer()

        @staticmethod
        def get_peft_model(model, **kw):
            return model

        @staticmethod
        def for_training(model):
            return model

        @staticmethod
        def for_inference(model):
            return model

    unsloth.FastLanguageModel = FastLanguageModel
    unsloth.UnslothTrainingArguments = lambda **kw: dict(kw)
    unsloth.UnslothTrainer = FakeTrainer
    sys.modules["unsloth"] = unsloth

    peft = _module("peft")
    peft.get_peft_model_state_dict = lambda model, adapter_name=None: {}
    peft.set_peft_model_state_dict = lambda model, sd, adapter_name=None: None
    sys.modules["peft"] = peft

    datasets = _module("datasets")
    datasets.Dataset = types.SimpleNamespace(from_list=lambda lst: lst)
    sys.modules["datasets"] = datasets

    import torch
    torch.cuda.reset_peak_memory_stats = lambda *a, **k: None
    torch.cuda.max_memory_allocated = lambda *a, **k: 0
    torch.cuda.empty_cache = lambda *a, **k: None


def _last_query_grid(tokens):
    i = len(tokens) - 1 - tokens[::-1].index(11)  # last "user"
    j = i + 2  # skip "user", "\n"
    k = tokens.index(15, j)
    rows, row = [], []
    for t in tokens[j:k]:
        if t == 10:
            rows.append(row)
            row = []
        else:
            row.append(t)
    rows.append(row)
    return np.array(rows)


def fake_inference_turbo_dfs(model, prefix_tokens, max_new_tokens, max_score, end_time):
    b = _behaviour()
    if b == "raise_dfs":
        raise RuntimeError("fake DFS failure")
    result = []
    for bid, toks in enumerate(prefix_tokens):
        if b == "empty":
            result.append((bid, []))
            continue
        g = _last_query_grid(toks)
        good = FakeTokenizer().encode("\n".join("".join(map(str, r)) for r in g)) + [15]
        wrong = g.copy()
        wrong[0, 0] = (wrong[0, 0] + 1) % 10
        bad = FakeTokenizer().encode("\n".join("".join(map(str, r)) for r in wrong)) + [15]
        result.append((bid, [(0.1, good), (1.2, bad)]))
    return result


def fake_calc_scores(queries, answers, tokenizer, model):
    return [0.5 for _ in queries]
