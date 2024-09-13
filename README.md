# embeddings

A embeddings server lib.

## python 環境

```
python>=3.7.9
```

## 安裝

```bash
git clone https://gitlab-devops.iii.org.tw/root/embeddings.git
cd embeddings
pip install -e .

# 其他版本
pip install -e .[all] # 全部安裝
pip install -e .[word2vec] # only word2vec
pip install -e .[sentence-transformers] # only sentence-transformers
pip install -e .[st] # only sentence-transformers
```

## 啟動伺服器

```bash
python embeddings/server.py
```

or

```bash
python -m embeddings.server
```

### 測試網址

http://localhost:8000/

### api swagger

http://localhost:8000/docs

## unit test

```bash
pip install tox
tox
```


```bash
. ./script/unittest.sh
```

or

```bash
coverage run -m unittest
coverage report -m
```

### coverage report

```
----------Start unittest----------
..
----------------------------------------------------------------------
Ran 3 tests in 3.915s

OK
----------End unittest----------
----------Code coverage----------
Name                                        Stmts   Miss   Cover   Missing
--------------------------------------------------------------------------
embeddings\model\_sentence_transformer.py      22     11  50.00%   20, 36, 43-49, 56-59
--------------------------------------------------------------------------
TOTAL                                         108     11  89.81%

2 files skipped due to complete coverage.
Coverage failure: total of 89.81 is less than fail-under=90.00
```

## Reference
