# coding: utf-8

import base64
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Union
from urllib.parse import unquote_plus

import numpy as np
from fastapi import Body, FastAPI, HTTPException, Path
from typing_extensions import Annotated

import embeddings._error as error
from embeddings import backend as _backend
from embeddings import config
from embeddings._types import EmbeddingDataElement, EmbeddingResponse, EmbeddingUsageResult, Model, ModelListResponse
from embeddings.backend._base import EmbeddingResult, EmbeddingResults

# Example for api format
__api_url_format = "http://hostname/engines/{engines}/{resource}"

logger = logging.getLogger(__name__)


# Same use model
__model: Dict[str, Dict[str, Union[str, float, _backend.base.EmbeddingModel]]] = dict()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start event - load model
    for backend_name, model_info in config.config_nested_get("model").items():
        for model_name, model_setting in model_info.items():
            await load_model(
                model_name=model_name,
                backend=backend_name,
                parameters=model_setting,
            )
    yield
    # Shutdown event


app = FastAPI(
    lifespan=lifespan,
)


@app.get("/models")
async def get_models():
    model_list = list()
    for model_name, model_info in __model.items():
        model_list.append(
            Model(
                id=model_name,
                created=model_info.get("create_time"),
                object="model",
                owned_by=model_info.get("backend"),
            )
        )
    return ModelListResponse(
        object="list",
        data=model_list,
    )


@app.post("/load_model")
async def load_model(
    model_name: Annotated[str, Body()],
    backend: Annotated[str, Body()],
    parameters: Annotated[Dict[str, Any], Body()] = {},
):
    model_fn: _backend.base.EmbeddingModel = getattr(_backend, backend)
    __model[model_name] = {
        "model": model_fn.from_pretrained(**parameters),
        "backend": backend,
        "create_time": int(datetime.now().timestamp()),
        "status": "loaded",
    }
    logger.info(f"Load '{model_name}' from '{backend}' success.")
    return {"status": True}


@app.post(
    "/engines/{engine}/embeddings",
    response_model=EmbeddingResponse,
)
async def embedding_method_1(
    engine: Annotated[str, Path()],
    input: Annotated[Union[str, List[str]], Body()],
):
    return await _embedding(
        model=engine,
        text=input,
    )


@app.post(
    "/embeddings",
    response_model=EmbeddingResponse,
)
async def embedding_method_2(
    input: Annotated[Union[str, List[str]], Body()],
    engine: Annotated[str, Body()] = None,
    model: Annotated[str, Body()] = None,
):
    if not engine and not model:
        raise HTTPException(status_code=422, detail="Parameter 'engine' and 'model' must choice one.")

    return await _embedding(
        model=engine if engine else model,
        text=input,
    )


async def _embedding(
    model: str,
    text: Union[str, List[str]],
):
    model = unquote_plus(model)
    if model not in __model:
        raise HTTPException(
            status_code=error.HTTP_CODE_INVALID_REQUEST,
            detail=f"Not support model '{model}'.",
        )

    text = [text] if isinstance(text, str) else text

    try:
        data: EmbeddingResults = __model.get(model).get("model").encodes(text)
    except Exception as e:
        logger.error(e)
        raise HTTPException(
            status_code=error.HTTP_CODE_TRY_AGAIN,
            detail="Unknown error, please try again.",
        )

    for i in range(len(data)):
        if isinstance(data[i], dict):
            data[i] = EmbeddingDataElement(
                object="embedding",
                index=i,
                **data[i],
            )
        elif isinstance(data[i], EmbeddingDataElement):
            data[i] = EmbeddingDataElement(
                object=data[i].object,
                index=i,
                embedding=data[i].embedding,
            )
        elif isinstance(data[i], EmbeddingResult):
            data[i] = EmbeddingDataElement(
                object="embedding",
                index=i,
                embedding=data[i].vector,
                prompt_tokens=data[i].used_tokens,
            )
        else:
            raise TypeError(f"Not support type for '{type(data[i])}'.")
    prompt_tokens = sum([d.prompt_tokens for d in data])

    # Use base64
    for i in range(len(data)):
        if isinstance(data[i].embedding, (list, np.ndarray)):
            data[i].embedding = str(
                base64.b64encode(
                    np.array(data[i].embedding, dtype="float32").tobytes(),
                )
            )[2:-1]

    return EmbeddingResponse(
        object=type(data).__name__,
        data=data,
        model=model,
        usage=EmbeddingUsageResult(
            prompt_tokens=prompt_tokens,
            completion_tokens=0,
            total_tokens=prompt_tokens,
        ),
    )


def main() -> None:
    import uvicorn

    server_setting = config.config_nested_get("server")
    uvicorn.run(app=app, **server_setting)


if __name__ == "__main__":
    main()
