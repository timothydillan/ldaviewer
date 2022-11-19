from . import models
from . import abstractscraper
from . import const
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_academic_corpus_by_search(search: models.AcademicPaperSearch):
    thread_map = {}
    extracted_corpus = {
        const.ABSTRACT_DICTIONARY_KEY: [],
        const.PUBLISHED_DATE_DICTIONARY_KEY: [],
        const.SOURCE_DICTIONARY_KEY: []
    }

    if search.scienceopen_search:
        scienceopen_db = abstractscraper.ScienceOpen()
        parsed_from_year, parsed_to_year = None, None
        if search.scienceopen_search.from_year:
            parsed_from_year = f"{search.scienceopen_search.from_year}-01-01"
        if search.scienceopen_search.to_year:
            parsed_to_year = f"{search.scienceopen_search.to_year}-12-31"
        thread_map[scienceopen_db.get_data_from_query] = {
            'search_query': search.scienceopen_search.search_query,
            'start_date': parsed_from_year,
            'end_date': parsed_to_year,
            'sort_by': search.scienceopen_search.sort_by,
            'limit': search.scienceopen_search.limit
        }

    if search.emerald_search:
        emerald_db = abstractscraper.Emerald()
        thread_map[emerald_db.get_data_from_query] = {
            'search_query': search.emerald_search.search_query,
            'start_year': search.emerald_search.from_year,
            'end_year': search.emerald_search.to_year,
            'sort_by': search.emerald_search.sort_by,
            'limit': search.emerald_search.limit
        }

    if search.core_search:
        core_db = abstractscraper.CORE()
        thread_map[core_db.get_data_from_query] = {
            'search_query': search.core_search.search_query,
            'from_year': search.core_search.from_year,
            'to_year': search.core_search.to_year,
            'sort_by': search.core_search.sort_by,
            'limit': search.core_search.limit
        }

    if search.arxiv_search:
        arxiv_db = abstractscraper.Arxiv()
        parsed_from_year, parsed_to_year = None, None
        if search.arxiv_search.from_year:
            parsed_from_year = f"{search.arxiv_search.from_year}-01-01"
        if search.arxiv_search.to_year:
            parsed_to_year = f"{search.arxiv_search.to_year}-12-31"
        thread_map[arxiv_db.get_data_from_query] = {
            'search_query': search.arxiv_search.search_query,
            'filter_by_date': parsed_from_year or parsed_to_year,
            'start_date': parsed_from_year,
            'end_date': parsed_to_year,
            'sort_by': search.arxiv_search.sort_by,
            'limit': search.arxiv_search.limit
        }

    if search.garuda_search:
        garuda_db = abstractscraper.Garuda()
        thread_map[garuda_db.get_data_from_query] = {
            'search_query': search.emerald_search.search_query,
            'start_year': search.emerald_search.from_year,
            'end_year': search.emerald_search.to_year,
            'limit': search.emerald_search.limit
        }

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(db, **kwargs)
                   for db, kwargs in thread_map.items()]

    for future in as_completed(futures):
        corpus_dict = future.result()
        extracted_corpus[const.ABSTRACT_DICTIONARY_KEY].extend(
            corpus_dict[const.ABSTRACT_DICTIONARY_KEY])
        extracted_corpus[const.PUBLISHED_DATE_DICTIONARY_KEY].extend(
            corpus_dict[const.PUBLISHED_DATE_DICTIONARY_KEY])
        extracted_corpus[const.SOURCE_DICTIONARY_KEY].extend(
            corpus_dict[const.SOURCE_DICTIONARY_KEY])

    return extracted_corpus


def get_extracted_corpus_breakdown(extracted_corpus):
    extracted_corpus = pd.DataFrame.from_dict(extracted_corpus)
    total_documents_processed = len(extracted_corpus.index)
    document_details = {
        "total_documents_processed": total_documents_processed, "document_details": []}
    for database in extracted_corpus[const.SOURCE_DICTIONARY_KEY].value_counts().index:
        document_details["document_details"].append({
            "database_name": database,
            "documents_retrieved": extracted_corpus[const.SOURCE_DICTIONARY_KEY].value_counts()[database].item()
        })
    return document_details
