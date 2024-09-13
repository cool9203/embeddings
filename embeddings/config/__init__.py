# coding: utf-8

from functools import partial
from pathlib import Path

from . import _config

env_file_paths = ["./.env.toml", "./.setting.toml"]
config = dict()
for env_file_path in env_file_paths:
    if Path(env_file_path).is_file() and Path(env_file_path).exists():
        print(f"Load '{env_file_path}' success")
        config = _config.load_config(env_file_path)
        break

config_nested_get = partial(_config.dict_nested_get, config)
