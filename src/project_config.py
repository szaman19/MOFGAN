import json
from pathlib import Path
from typing import NamedTuple

import main

ROOT_PATH = Path(main.__file__).parent.parent
RESOURCE_PATH = ROOT_PATH / 'resources'


class _SFTPConfig(NamedTuple):
    host: str
    user: str
    key: str
    root: str


class _WBConfig(NamedTuple):
    user: str


class _LocalConfig(NamedTuple):
    root: str
    sample_save_path: str


with open(RESOURCE_PATH / 'config.json') as f:
    data = json.load(f)
    sftp = _SFTPConfig(**data['sftp'])
    wandb = _WBConfig(**data['wandb'])
    local = _LocalConfig(**data['local'])
    print("LOADED CONFIG!")
