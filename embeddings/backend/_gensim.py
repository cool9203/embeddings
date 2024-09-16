# coding: utf-8

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Union

import gensim as _gensim
import jieba
import numpy as np
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError
from huggingface_hub.utils._errors import RepositoryNotFoundError

from ._base import EmbeddingModel, EmbeddingResult, EmbeddingResults

logger = logging.getLogger(
    __name__,
)


class gensim(EmbeddingModel):
    def __init__(
        self,
        model: _gensim.models.KeyedVectors,
        tokenizer: jieba,
    ) -> None:
        super().__init__(
            model=model,
            tokenizer=tokenizer,
        )

        if TYPE_CHECKING:
            self.model: _gensim.models.KeyedVectors
            self.tokenizer: jieba

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: Union[str, Path],
        tokenizer: jieba = None,
        **kwds: Dict[str, Any],
    ) -> gensim:
        jieba_dict_path = kwds.pop("jieba_dict_path", None) if not tokenizer else None
        if not Path(model_name_or_path).is_dir():
            try:
                model_name_or_path = snapshot_download(repo_id=model_name_or_path, **kwds)
            except (RepositoryNotFoundError, HFValidationError) as e:
                raise TypeError("Variable `model_name_or_path` not a folder, and not exist in huggingface.") from e

        try:
            model = _gensim.models.KeyedVectors.load(str(Path(model_name_or_path, "model.kv")))
        except FileNotFoundError as e:
            raise TypeError("model backend not is gensim") from e

        tokenizer = jieba if not tokenizer else tokenizer

        if jieba_dict_path and Path(jieba_dict_path).exists():  # pragma: no cover
            tokenizer.set_dictionary(jieba_dict_path)
        tokenizer.initialize()
        return gensim(model, tokenizer)

    def get_vectors(
        self,
        words: List[str],
        norm: bool = False,
        oov_ignore: bool = True,
        oov_vector: List[float] = None,
        **kwds: Dict[str, Any],
    ) -> EmbeddingResults:
        if oov_vector is None:
            oov_vector = np.zeros(self.model.vector_size)
        if not isinstance(words, list):
            raise TypeError(f"Variable 'words' type should be List[str], but got {type(words)}.")

        results = list()
        for word in words:
            result = self.get_vector(
                word=word,
                norm=norm,
                **kwds,
            )
            if result.vector is None and not oov_ignore:
                result.vector = oov_vector
                results.append(result)
            elif result.vector is not None:
                results.append(result)
        return results

    def get_vector(
        self,
        word: str,
        norm: bool = False,
        **kwds: Dict[str, Any],
    ) -> EmbeddingResult:
        if not isinstance(word, str):
            raise TypeError(f"Variable 'word' type should be str, but got {type(word)}.")

        word_vec = None
        try:
            word_vec = self.model.get_vector(word, norm=norm)
        except KeyError:
            logger.debug(f"Get oov word: '{word}'")

        return EmbeddingResult(
            vector=word_vec,
            used_tokens=1,
        )

    def encodes(
        self,
        sentences: List[str],
        **kwds: Dict[str, Any],
    ) -> EmbeddingResults:
        if not isinstance(sentences, list):
            raise TypeError(f"Variable 'sentences' type should be List[str], but got {type(sentences)}.")

        results = list()
        for sentence in sentences:
            results.append(self.encode(sentence=sentence, **kwds))
        return results

    def encode(
        self,
        sentence: str,
        **kwds: Dict[str, Any],
    ) -> EmbeddingResult:
        if not isinstance(sentence, str):
            raise TypeError(f"Variable 'sentence' type should be str, but got {type(sentence)}.")

        sentence_seg = [word for word in self.tokenizer.cut(sentence, cut_all=False)]
        logger.debug(f"sentence_seg: {sentence_seg}")
        results = self.get_vectors(sentence_seg, **kwds)
        logger.debug(f"results: {results}")
        sentence_vec = np.array([result.vector for result in results], dtype="float32").mean(0)
        logger.debug(f"sentence_vec: {sentence_vec}")
        return EmbeddingResult(
            vector=np.array(sentence_vec, dtype="float32"),
            used_tokens=sum([result.used_tokens for result in results]),
        )
