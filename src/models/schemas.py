from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    image_url: str | None = None
    thread_id: str = Field(min_length=1, max_length=200)

    @field_validator("message", "thread_id", mode="before")
    @classmethod
    def trim_required_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class KnowledgeDocumentRequest(BaseModel):
    source: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1, max_length=100_000)

    @field_validator("source", "content", mode="before")
    @classmethod
    def trim_required_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value