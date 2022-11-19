from enum import Enum
from typing import Union, Optional
from pydantic import BaseModel


class SortCriterion(str, Enum):
    RELEVANCE = "relevance"
    NEWEST_TO_OLDEST = "recent"
    OLDEST_TO_NEWEST = "old"


class AbstractPaperSearch(BaseModel):
    search_query: str
    limit: int
    sort_by: Optional[SortCriterion] = SortCriterion.RELEVANCE
    from_year: Optional[int] = None
    to_year: Optional[int] = None


class AcademicPaperSearch(BaseModel):
    core_search: Optional[AbstractPaperSearch] = None
    arxiv_search: Optional[AbstractPaperSearch] = None
    emerald_search: Optional[AbstractPaperSearch] = None
    scienceopen_search: Optional[AbstractPaperSearch] = None
    garuda_search: Optional[AbstractPaperSearch] = None

    def is_search_query_empty(self) -> bool:
        return self.core_search is None and self.arxiv_search is None and self.emerald_search is None and self.scienceopen_search is None and self.garuda_search is None

    def get_search_query(self):
        if self.core_search:
            return self.core_search.search_query
        if self.arxiv_search:
            return self.arxiv_search.search_query
        if self.emerald_search:
            return self.emerald_search.search_query
        if self.scienceopen_search:
            return self.scienceopen_search.search_query
        if self.garuda_search:
            return self.garuda_search.search_query
        return ""
