# coding: utf-8

import importlib
import pkgutil
from pathlib import Path
from typing import TYPE_CHECKING, List

import embeddings

__all__: List[str] = []

PACKAGE = embeddings
FOLDER_NAME: str = "backend"


if TYPE_CHECKING:
    from . import _base as base
    from ._gensim import gensim
    from ._llama_cpp import llama_cpp
    from ._sentence_transformer import sentence_transformer


# Dynamic import model
for finder, name, is_pkg in pkgutil.iter_modules([str(Path(PACKAGE.__path__[0], FOLDER_NAME))]):
    try:
        module_name = name if name[0] not in ["_"] else name[1:]  # Alias name to not have "_"

        if module_name not in globals():
            module = importlib.import_module(f".{name}", f"{PACKAGE.__name__}.{FOLDER_NAME}")  # Import
            module = getattr(module, module_name, module)
            globals().update({module_name: module})  # Set alias name to this module scope
            __all__.append(module_name)  # Update to __all__

    except Exception as e:
        print(f"Can not import '{name}'.")
        print(e)
