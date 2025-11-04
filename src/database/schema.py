"""Data models and schema definitions"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum


class SourceType(str, Enum):
    """Source type for corpus documents"""
    EMAIL = "email"
    CHAT = "chat"
    DOCUMENT = "document"
    CODE = "code"
    NOTE = "note"


class CorpusDocument(BaseModel):
    """Document schema for corpus storage"""
    id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None

    @property
    def timestamp(self) -> Optional[datetime]:
        """Get timestamp from metadata"""
        if "timestamp" in self.metadata:
            return datetime.fromisoformat(self.metadata["timestamp"])
        return None

    @property
    def source(self) -> Optional[SourceType]:
        """Get source type from metadata"""
        if "source" in self.metadata:
            return SourceType(self.metadata["source"])
        return None

    @property
    def char_length(self) -> int:
        """Get character length"""
        return self.metadata.get("char_length", len(self.text))


class SearchResult(BaseModel):
    """Search result schema"""
    text: str
    metadata: Dict[str, Any]
    similarity: float
    document_id: str


class SearchFilters(BaseModel):
    """Filters for search queries"""
    time_range: Optional[Dict[str, Optional[str]]] = None
    source_filter: Optional[List[SourceType]] = None
    min_similarity: Optional[float] = None
