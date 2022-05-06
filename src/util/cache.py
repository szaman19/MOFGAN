import json
from collections import deque
from pathlib import Path
from typing import List

_cache_data = {}
_cache_path = Path('cache.json')
if _cache_path.exists():
    with _cache_path.open('r') as _cache_file:
        _cache_data = json.load(_cache_file)


def get(key: List[str]):
    current = _cache_data
    for k in key:
        if k in current:
            current = current[k]
        else:
            return None
    print(f"Loaded: '{key[0]}' from cache!")
    return current


def store(key: List[str], value):
    current = _cache_data
    key = deque(key)
    while key:
        k = key.popleft()
        if key:
            if k in current:
                current = current[k]
            else:
                current[k] = {}
                current = current[k]
        else:
            current[k] = value
            with open(_cache_path, 'w+') as f:
                json.dump(_cache_data, f, indent='\t')
