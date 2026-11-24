from pydantic import BaseModel, Field


class TrainingSample(BaseModel):
    text: str = Field(min_length=1)

    category: str
    component: str
    cause: str
    source: str
    source_group: str
    synthetic: bool = False
    schema_version: str = "1.0"
