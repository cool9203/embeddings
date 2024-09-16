# coding: utf-8

import logging
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np
from huggingface_hub.errors import HFValidationError
from huggingface_hub.utils._errors import RepositoryNotFoundError
from sentence_transformers import SentenceTransformer

from ._base import EmbeddingModel, EmbeddingResult, EmbeddingResults

logger = logging.getLogger(
    __name__,
)


class sentence_transformer(EmbeddingModel):
    def __init__(
        self,
        model: Any,
        tokenizer=None,
    ) -> None:
        super().__init__(
            model=model,
            tokenizer=tokenizer,
        )

        if TYPE_CHECKING:
            self.model: SentenceTransformer

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        tokenizer=None,
        **kwds: Dict[str, Any],
    ) -> EmbeddingModel:
        try:
            if not model_name_or_path:
                raise ValueError
            model = SentenceTransformer(
                model_name_or_path=model_name_or_path,
                **kwds,
            )
        except (HFValidationError, RepositoryNotFoundError, OSError, ValueError) as e:
            raise TypeError("Variable `model_name_or_path` not a folder, and not exist in huggingface.") from e

        return sentence_transformer(
            model=model,
            tokenizer=tokenizer,
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
            vector=np.array(
                self.model.encode(
                    sentences=sentence,
                    normalize_embeddings=normalize,
                    **kwds,
                ),
                dtype="float64",
            ),
            used_tokens=len(sentence),
        )
