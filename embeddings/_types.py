# coding: utf-8
from __future__ import annotations

from typing import List, Literal, Union

from pydantic import BaseModel, Field


class Model(BaseModel):
    id: str = Field(
        ...,
        title="模型名稱",
        description="模型名稱",
        examples=["llama-3", "gpt-4o-mini", "openai-o1-mini", "word2vec"],
    )

    created: int = Field(
        ...,
        title="模型建立時間",
        description="模型建立時間",
        examples=[0],
    )

    object: Literal["model"] = Field(
        ...,
        title="該object名稱",
        description="該object名稱",
        examples=["model"],
    )

    owned_by: str = Field(
        ...,
        title="模型所有者",
        description="模型所有者",
        examples=["llama_cpp", "openai"],
    )


class ModelListResponse(BaseModel):
    object: Literal["list"] = Field(
        ...,
        title="type",
        description="type",
        examples=["list"],
    )
    data: List[Model] = Field(
        ...,
        title="模型資料列表",
        description="模型資料列表",
        examples=[[{property_name: info.examples[0] for property_name, info in Model.model_fields.items() if info.examples}]],
    )


class EmbeddingDataElement(BaseModel):
    object: Literal["embedding"] = Field(
        ...,
        title="該次呼叫服務名稱",
        description="該次呼叫服務名稱",
        examples=["embedding"],
    )
    index: int = Field(
        ...,
        title="",
        description="",
        examples=[""],
    )
    embedding: Union[List[float], str] = Field(
        ...,
        title="",
        description="",
        examples=[""],
    )  # str 等於 base64.encode 過
    prompt_tokens: int = Field(
        ...,
        title="",
        description="",
        examples=[""],
    )  # openai 沒有 when 2023/10/06


class EmbeddingUsageResult(BaseModel):
    prompt_tokens: int = Field(
        ...,
        title="",
        description="",
        examples=[128],
    )
    completion_tokens: int = Field(
        ...,
        title="",
        description="",
        examples=[128],
    )  # openai.embeddings 沒有, 但 openai.chat_completion 有
    total_tokens: int = Field(
        ...,
        title="",
        description="",
        examples=[1024],
    )


class EmbeddingResponse(BaseModel):
    object: Literal["list"] = Field(
        ...,
        title="",
        description="",
        examples=[""],
    )
    data: List[EmbeddingDataElement] = Field(
        ...,
        title="",
        description="",
        examples=[
            [
                {
                    property_name: info.examples[0]
                    for property_name, info in EmbeddingDataElement.model_fields.items()
                    if info.examples
                }
            ]
        ],
    )
    model: str = Field(
        ...,
        title="",
        description="",
        examples=[""],
    )
    usage: EmbeddingUsageResult = Field(
        ...,
        title="",
        description="",
        examples=[
            [
                {
                    property_name: info.examples[0]
                    for property_name, info in EmbeddingUsageResult.model_fields.items()
                    if info.examples
                }
            ]
        ],
    )

    # class Config:
    #     populate_by_name = True
    #     arbitrary_types_allowed = True
