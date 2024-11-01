# coding: utf-8

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Union

import numpy as np
from huggingface_hub.errors import HFValidationError
from llama_cpp import Llama

from ._base import EmbeddingModel, EmbeddingResult, EmbeddingResults

huggingface_hub_version = version("huggingface_hub")
huggingface_hub_version = [int(v) for v in huggingface_hub_version.split(".")]
if huggingface_hub_version[1] <= 24 and huggingface_hub_version[2] <= 7:
    from huggingface_hub.utils._errors import RepositoryNotFoundError
else:
    from huggingface_hub.errors import RepositoryNotFoundError


class llama_cpp(EmbeddingModel):
    def __init__(
        self,
        model: Llama,
        tokenizer=None,
    ) -> None:
        super().__init__(
            model=model,
            tokenizer=tokenizer,
        )

        if TYPE_CHECKING:
            self.model: Llama

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Union[str, Path],
        tokenizer=None,
        filename: str = None,
        **kwds: Dict[str, Any],
    ) -> llama_cpp:
        try:
            model = Llama.from_pretrained(
                repo_id=model_name_or_path,
                filename=filename,
                embedding=True,
                **kwds,
            )
        except (HFValidationError, RepositoryNotFoundError) as e:
            raise TypeError("Variable `model_name_or_path` not a folder, and not exist in huggingface.") from e

        return llama_cpp(model, tokenizer)

    def encodes(
        self,
        sentences: List[str],
        **kwds: Dict[str, Any],
    ) -> EmbeddingResults:
        if not isinstance(sentences, list):
            raise TypeError(f"Variable 'sentences' type should be List[str], but got {type(sentences)}.")

        results = list()
        _embeddings = self.model.create_embedding(sentences)
        used_tokens = _embeddings["usage"]["total_tokens"]
        for embedding_data in _embeddings["data"]:
            results.append(
                EmbeddingResult(
                    vector=np.array(embedding_data["embedding"], dtype="float64"),
                    used_tokens=used_tokens if used_tokens is not None else 0,
                )
            )
            used_tokens = None
        return results

    def encode(
        self,
        sentence: str,
        **kwds: Dict[str, Any],
    ) -> EmbeddingResult:
        if not isinstance(sentence, str):
            raise TypeError(f"Variable 'sentence' type should be str, but got {type(sentence)}.")

        results = self.encodes([sentence], **kwds)
        return results[0]
