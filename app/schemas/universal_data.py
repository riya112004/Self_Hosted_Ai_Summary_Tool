import warnings
from typing import Any

from pydantic import BaseModel, ConfigDict

warnings.filterwarnings("ignore", category=UserWarning, message='.*shadows an attribute.*')


class Metadata(BaseModel):
    source_type: str  # "json", "csv", "database", "text"
    row_count: int


class ColumnInfo(BaseModel):
    name: str
    dtype: str  # "string", "integer", "float", "boolean", "unknown"


class Schema(BaseModel):
    columns: list[ColumnInfo]


class UDR(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    metadata: Metadata
    schema: Schema
    statistics: dict[str, Any] = {}
    records: list[dict[str, Any]] = []