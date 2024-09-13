# coding: utf-8

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Union

import numpy as np
from InstructorEmbedding import INSTRUCTOR

from ._base import EmbeddingModel, EmbeddingResult, EmbeddingResults

logger = logging.getLogger(
    __name__,
)


# Reference from langchain: https://github.com/langchain-ai/langchain/blob/b01a443ee525e274335f475a849a1681240ff249/libs/langchain/langchain/embeddings/huggingface.py#L11-L12
DEFAULT_EMBED_INSTRUCTION = "Represent the document for retrieval: "
DEFAULT_QUERY_INSTRUCTION = "Represent the question for retrieving supporting documents: "


class instructor_embedding(EmbeddingModel):
    def __init__(
        self,
        model: Any,
    ) -> None:
        super().__init__(
            model=model,
        )

        if TYPE_CHECKING:
            self.model: INSTRUCTOR

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        **kwds: Dict[str, Any],
    ) -> EmbeddingModel:
        model = INSTRUCTOR(
            model_name_or_path=model_name_or_path,
        )
        return instructor_embedding(
            model=model,
        )

    def encodes(
        self,
        sentences: List[str],
        normalize: bool = True,
        doc: bool = False,
        **kwds: Dict[str, Any],
    ) -> EmbeddingResults:
        if not isinstance(sentences, str):
            raise TypeError(f"Variable 'sentences' type should be List[str], but got {type(sentences)}.")

        results = list()
        for sentence in sentences:
            results.append(
                self.encode(
                    sentence=sentence,
                    normalize=normalize,
                    doc=doc,
                    **kwds,
                )
            )
        return results

    def encode(
        self,
        sentence: str,
        normalize: bool = True,
        doc: bool = False,
        **kwds: Dict[str, Any],
    ) -> EmbeddingResult:
        if not isinstance(sentence, str):
            raise TypeError(f"Variable 'sentence' type should be str, but got {type(sentence)}.")

        sentence = [[DEFAULT_EMBED_INSTRUCTION if doc else DEFAULT_QUERY_INSTRUCTION, sentence]]
        return EmbeddingResult(
            vector=np.array(self.model.encode(sentences=sentence, normalize_embeddings=normalize)[0], dtype="float64"),
            used_tokens=len(sentence),
        )
