#! python3
# coding: utf-8
# write reference: https://docs.python.org/zh-tw/3/library/unittest.html


import unittest

base_url = "http://localhost:8000"
api_key = "empty"


class Test(unittest.TestCase):
    def test_use_openai_latest(self):
        import openai

        all_embeddings_dimensions = dict()
        openai_version = tuple([int(s) for s in openai.__version__.split(".")])
        if openai_version[0] == 1:
            client = openai.Client(
                api_key=api_key,
                base_url=base_url,
            )
            response = client.embeddings.create(
                model="word2vec",
                timeout=30,
                input="test content",
            )
            all_embeddings_dimensions["word2vec"] = len(response.model_dump().get("data")[0].get("embedding"))

        elif openai_version[0] == 0:
            response = openai.Embedding.create(
                model="word2vec",
                request_timeout=30,
                api_key=api_key,
                api_base=base_url,
                input="test content",
            )
            all_embeddings_dimensions["word2vec"] = len(response.get("data")[0].get("embedding"))
        print(all_embeddings_dimensions)


if __name__ == "__main__":
    unittest.main()
    input("press Enter to continue...")
