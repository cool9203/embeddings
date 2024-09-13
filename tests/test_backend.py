#! python3
# coding: utf-8
# write reference: https://docs.python.org/zh-tw/3/library/unittest.html

import unittest

import jieba

from embeddings import backend, config


class Test(unittest.TestCase):
    def test_word2vec(self):
        # Test error model path
        with self.assertRaises(NotADirectoryError):
            _ = backend.word2vec.from_pretrained("./not_is_dir", None)

        with self.assertRaises(FileNotFoundError):
            _ = backend.word2vec.from_pretrained("", None)

        model_setting = config.config_nested_get("model.word2vec")
        model = backend.word2vec.from_pretrained(**model_setting)

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
        self.assertIsInstance(result, backend.types.EmbeddingResult)
        self.assertEqual(result.used_tokens, 1)

        # Test get_vectors with oov_ignore=True
        result = model.get_vectors(["好", " "])
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], backend.types.EmbeddingResult)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].used_tokens, 1)

        # Test get_vectors with oov_ignore=False
        result = model.get_vectors(["好", " "], oov_ignore=False)
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], backend.types.EmbeddingResult)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].used_tokens, 1)
        self.assertEqual(result[1].used_tokens, 1)

        # Test encode with oov_ignore=True
        result = model.encode("好 ")
        self.assertIsInstance(result, backend.types.EmbeddingResult)
        self.assertEqual(result.used_tokens, 1)

        # Test encode with oov_ignore=False
        result = model.encode("好 ", oov_ignore=False)
        self.assertIsInstance(result, backend.types.EmbeddingResult)
        self.assertEqual(result.used_tokens, 2)

        # Test encodes with oov_ignore=True
        result = model.encodes(["好 "])
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], backend.types.EmbeddingResult)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].used_tokens, 1)

        # Test encodes with oov_ignore=False
        result = model.encodes(["好 "], oov_ignore=False)
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], backend.types.EmbeddingResult)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].used_tokens, 2)


if __name__ == "__main__":
    unittest.main()
    input("press Enter to continue...")
