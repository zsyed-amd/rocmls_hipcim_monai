"""Pydantic response/request models. Extended per phase."""

from pydantic import BaseModel


class SystemInfo(BaseModel):
    gpu: str
    cpu: str
    versions: dict[str, str]
