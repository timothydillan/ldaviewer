import json
import pandas as pd
from time import time
from typing import Union, Optional
from fastapi import FastAPI, Path, Query, Response, status
from pydantic import BaseModel
from enum import Enum
from fastapi.middleware.cors import CORSMiddleware
import papersearch.models as papersearch_models
import papersearch.service as papersearch_service
import lda.service as lda_service
import database.service as database_service
from fastapi.responses import StreamingResponse
import io


def search(search: papersearch_models.AcademicPaperSearch):
    start_time = time()

    search_results = papersearch_service.get_academic_corpus_by_search(search)

    metadata = papersearch_service.get_extracted_corpus_breakdown(
        search_results)

    extracted_keyphrase_per_topic, lda_dataframe_output, coherence_score = lda_service.get_lda_keyphrases_and_dataframe(
        search_results)

    topics_over_time = lda_service.parse_lda_ouput_trend_response(
        extracted_keyphrase_per_topic, lda_dataframe_output)

    keyphrases_and_weights = lda_service.parse_extracted_kephrases_response(
        extracted_keyphrase_per_topic, lda_dataframe_output)

    metadata["coherence_score"] = f"{coherence_score:.2f}"
    end_time = time()
    metadata["process_time"] = f"{(end_time - start_time) / 60:.2f}"
    metadata["num_of_topics"] = f"{len(extracted_keyphrase_per_topic)}"

    # Store results to db before returning response
    inserted_id = database_service.store_search_results(search.get_search_query(), lda_dataframe_output.to_json(
    ), json.dumps(keyphrases_and_weights), json.dumps(topics_over_time), json.dumps(metadata))

    return {"status": "success", "id": inserted_id, "extracted_topics_and_weights": keyphrases_and_weights, "extracted_topics_over_time": topics_over_time, "metadata": metadata}


if __name__ == "__main__":
    academicPaper = papersearch_models.AcademicPaperSearch()
    abstractSearch = papersearch_models.AbstractPaperSearch(**{"search_query":"image recognition", "limit":50})
    academicPaper.garuda_search = abstractSearch
    search(academicPaper)
