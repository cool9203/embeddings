# coding: utf-8
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Union

__all__: Tuple[str] = (
    "EmbeddingModel",
    "EmbeddingResult",
    "EmbeddingResults",
    "Vector",
    "Vectors",
)


class EmbeddingModel(abc.ABC):
    def __init__(
        self,
        **kwds,
    ) -> None:
        for k, v in kwds.items():
            setattr(self, k, v)

    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    @model.setter
    def model(
        self,
        model,
    ) -> None:
        self._model = model

    @tokenizer.setter
    def tokenizer(
        self,
        tokenizer,
    ) -> None:
        self._tokenizer = tokenizer

    @abc.abstractmethod
    def encode(self) -> EmbeddingResult:
        raise NotImplementedError

    @abc.abstractmethod
    def encodes(self) -> EmbeddingResults:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_pretrained(
        cls,
        *args,
        **kwds,
    ) -> EmbeddingModel:
        raise NotImplementedError


@dataclass
class EmbeddingResult:
    vector: Vector
    used_tokens: int


Vector = Union[Sequence[float], Sequence[int]]
Vectors = List[Vector]
EmbeddingResults = List[EmbeddingResult]
