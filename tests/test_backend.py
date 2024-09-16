#! python3
# coding: utf-8
# write reference: https://docs.python.org/zh-tw/3/library/unittest.html

import unittest

import jieba

from embeddings import backend


class Test(unittest.TestCase):
    def test_gensim(self):
        # Test error model path
        with self.assertRaises(TypeError):
            _ = backend.gensim.from_pretrained("./not_is_dir", None)

        with self.assertRaises(TypeError):
            _ = backend.gensim.from_pretrained("./tests", None)

        with self.assertRaises(TypeError):
            _ = backend.gensim.from_pretrained("", None)

        model = backend.gensim.from_pretrained(model_name_or_path="./model/word2vec")

        # Test pass error type
        with self.assertRaises(TypeError):
            _ = model.get_vector([])

        with self.assertRaises(TypeError):
            _ = model.get_vectors("")

        with self.assertRaises(TypeError):
            _ = model.encode([])

        with self.assertRaises(TypeError):
            _ = model.encodes("")

        # Test tokenizer is jieba
        self.assertEqual(model.tokenizer, jieba)

        # Test get_vector
        result = model.get_vector("好")
        self.assertIsInstance(result, backend.base.EmbeddingResult)
        self.assertEqual(result.used_tokens, 1)

        # Test get_vectors with oov_ignore=True
        result = model.get_vectors(["好", " "])
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], backend.base.EmbeddingResult)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].used_tokens, 1)

        # Test get_vectors with oov_ignore=False
        result = model.get_vectors(["好", " "], oov_ignore=False)
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], backend.base.EmbeddingResult)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].used_tokens, 1)
        self.assertEqual(result[1].used_tokens, 1)

        # Test encode with oov_ignore=True
        result = model.encode("好 ")
        self.assertIsInstance(result, backend.base.EmbeddingResult)
        self.assertEqual(result.used_tokens, 1)

        # Test encode with oov_ignore=False
        result = model.encode("好 ", oov_ignore=False)
        self.assertIsInstance(result, backend.base.EmbeddingResult)
        self.assertEqual(result.used_tokens, 2)

        # Test encodes with oov_ignore=True
        result = model.encodes(["好 "])
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], backend.base.EmbeddingResult)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].used_tokens, 1)

        # Test encodes with oov_ignore=False
        result = model.encodes(["好 "], oov_ignore=False)
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], backend.base.EmbeddingResult)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].used_tokens, 2)

    def test_llama_cpp(self):
        # Test error model path
        with self.assertRaises(TypeError):
            _ = backend.llama_cpp.from_pretrained("./not_is_dir", None)

        with self.assertRaises(TypeError):
            _ = backend.llama_cpp.from_pretrained("./tests", None)

        with self.assertRaises(TypeError):
            _ = backend.llama_cpp.from_pretrained("", None)

        model = backend.llama_cpp.from_pretrained(
            model_name_or_path="Qwen/Qwen2-0.5B-Instruct-GGUF", filename="qwen2-0_5b-instruct-q2_k.gguf"
        )

        with self.assertRaises(TypeError):
            _ = model.encode([])

        with self.assertRaises(TypeError):
            _ = model.encodes("")

        # Test encode
        result = model.encode("好")
        self.assertIsInstance(result, backend.base.EmbeddingResult)

        # Test encodes
        result = model.encodes(["好", "喔"])
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], backend.base.EmbeddingResult)
        self.assertEqual(len(result), 2)

    def test_sentence_transformer(self):
        # Test error model path
        with self.assertRaises(TypeError):
            _ = backend.sentence_transformer.from_pretrained("./not_is_dir", None)

        with self.assertRaises(TypeError):
            _ = backend.sentence_transformer.from_pretrained("./tests", None)

        with self.assertRaises(TypeError):
            _ = backend.sentence_transformer.from_pretrained("", None)

        model = backend.sentence_transformer.from_pretrained(model_name_or_path="DMetaSoul/Dmeta-embedding")

        with self.assertRaises(TypeError):
            _ = model.encode([])

        with self.assertRaises(TypeError):
            _ = model.encodes("")

        # Test encode
        result = model.encode("好")
        self.assertIsInstance(result, backend.base.EmbeddingResult)

        # Test encodes
        result = model.encodes(["好", "喔"])
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], backend.base.EmbeddingResult)
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
    input("press Enter to continue...")
