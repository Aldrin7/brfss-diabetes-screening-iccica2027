import os
import json
import random
import numpy as np

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)

def load_json(path):
    with open(path) as f:
        return json.load(f)
