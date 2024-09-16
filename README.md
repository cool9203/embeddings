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

> [!IMPORTANT]
> 需要先啟動伺服器, 連帶測試 openai package 可以正常使用

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
Ran 5 tests in 25.680s

OK
----------End unittest----------
----------Code coverage----------
Name                                          Stmts   Miss   Cover   Missing
----------------------------------------------------------------------------
embeddings/backend/_instructor_embedding.py      28     13  53.57%   27, 40-43, 54-67, 76-80
embeddings/config/_config.py                     59     36  38.98%   24-35, 49-73, 83, 86-88
----------------------------------------------------------------------------
TOTAL                                           249     49  80.32%
```

## Reference
