from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator


class ExtractRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str = Field(min_length=1, max_length=255)
    download_url: HttpUrl


class ExtractionBlock(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)

    type: Literal[
        "heading",
        "paragraph",
        "list",
        "table",
        "figure_caption",
        "equation",
        "code",
        "reference",
    ]
    text: str = Field(min_length=1)
    level: int | None = Field(default=None, ge=1, le=6)
    caption: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_level(self):
        if self.type == "heading" and self.level is None:
            raise ValueError("heading blocks require level")
        if self.type != "heading" and self.level is not None:
            raise ValueError("level is only valid for heading blocks")
        return self


class ExtractionManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    blocks: list[ExtractionBlock] = Field(min_length=1)
    images: list[str] = Field(default_factory=list)


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system: str = Field(default="", max_length=8000)
    prompt: str = Field(min_length=1, max_length=48000)

    @field_validator("system")
    @classmethod
    def strip_system(cls, value: str) -> str:
        return value.strip()

    @field_validator("prompt")
    @classmethod
    def strip_prompt(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("prompt must not be empty")
        return value


class GenerateResponse(BaseModel):
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    response: str = Field(min_length=1)
    done: bool


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=12000)

    @field_validator("text")
    @classmethod
    def strip_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must not be empty")
        return value


class EmbeddingResponse(BaseModel):
    embedding: list[float]


class BatchEmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    texts: list[str] = Field(min_length=1, max_length=64)

    @field_validator("texts")
    @classmethod
    def strip_texts(cls, values: list[str]) -> list[str]:
        stripped = [value.strip() for value in values]
        if any(not value for value in stripped):
            raise ValueError("texts must not contain empty values")
        return stripped


class BatchEmbeddingResponse(BaseModel):
    embeddings: list[list[float]]
