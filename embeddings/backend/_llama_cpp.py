# coding: utf-8

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Union

import llama_cpp as _llama_cpp
import numpy as np

from ._base import EmbeddingModel, EmbeddingResult, EmbeddingResults


class llama_cpp(EmbeddingModel):
    def __init__(
        self,
        model: _llama_cpp.Llama,
        tokenizer=None,
    ) -> None:
        super().__init__(
            model=model,
            tokenizer=None,
        )

        if TYPE_CHECKING:
            self.model: _llama_cpp.Llama

    @classmethod
    def from_pretrained(
        cls,
        path: Union[str, Path],
        tokenizer=None,
        **kwds: Dict[str, Any],
    ) -> llama_cpp:
        if not Path(path).is_dir():
            raise NotADirectoryError("Variable `path` not a folder.")
        model = _llama_cpp.Llama(model_path=path, embedding=True)
        return llama_cpp(model, None)

    def encodes(
        self,
        sentences: List[str],
        **kwds: Dict[str, Any],
    ) -> EmbeddingResults:
        if not isinstance(sentences, list):
            raise TypeError(f"Variable 'sentences' type should be List[str], but got {type(sentences)}.")

        results = list()
        _embeddings = self.model.create_embedding(sentences)
        for _embedding in _embeddings.data:
            results.append(
                EmbeddingResult(
                    vector=np.array(_embedding, dtype="float64"),
                    used_tokens=0,
                )
            )
        return

    def encode(
        self,
        sentence: str,
        **kwds: Dict[str, Any],
    ) -> EmbeddingResult:
        if not isinstance(sentence, str):
            raise TypeError(f"Variable 'sentence' type should be str, but got {type(sentence)}.")

        results = self.encodes([sentence], **kwds)
        return EmbeddingResult(
            vector=np.array(results[0], dtype="float32"),
            used_tokens=sum([result.used_tokens for result in results]),
        )
