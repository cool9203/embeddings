# coding: utf-8

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from ._base import EmbeddingModel, EmbeddingResult, EmbeddingResults

logger = logging.getLogger(
    __name__,
)


class sentence_transformer(EmbeddingModel):
    def __init__(
        self,
        model: Any,
    ) -> None:
        super().__init__(
            model=model,
        )

        if TYPE_CHECKING:
            self.model: SentenceTransformer

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        **kwds: Dict[str, Any],
    ) -> EmbeddingModel:
        model = SentenceTransformer(
            model_name_or_path=model_name_or_path,
            **kwds,
        )
        return sentence_transformer(
            model=model,
        )

    def encodes(
        self,
        sentences: List[str],
        normalize: bool = True,
        **kwds: Dict[str, Any],
    ) -> EmbeddingResults:
        if not isinstance(sentences, list):
            raise TypeError(f"Variable 'sentences' type should be List[str], but got {type(sentences)}.")

        results = list()
        for sentence in sentences:
            results.append(
                self.encode(
                    sentence=sentence,
                    normalize=normalize,
                    **kwds,
                )
            )
        return results

    def encode(
        self,
        sentence: str,
        normalize: bool = True,
        **kwds: Dict[str, Any],
    ) -> EmbeddingResult:
        if not isinstance(sentence, str):
            raise TypeError(f"Variable 'sentence' type should be str, but got {type(sentence)}.")

        return EmbeddingResult(
            vector=np.array(self.model.encode(sentences=sentence, normalize_embeddings=normalize), dtype="float64"),
            used_tokens=len(sentence),
        )
