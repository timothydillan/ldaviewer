import json
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


class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"


# Run with uvicron filename:instancename --reload (--reload for hotreload)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World no no no"}


@app.post("/search")
async def search(search: papersearch_models.AcademicPaperSearch, response: Response):
    start_time = time()

    if search.is_search_query_empty():
        response.status_code = status.HTTP_400_BAD_REQUEST
        return {"status": "error", "message": "empty search query for all databases"}

    try:
        search_results = papersearch_service.get_academic_corpus_by_search(
            search)
    except Exception as e:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"status": "error", "message": repr(e)}

    metadata = papersearch_service.get_extracted_corpus_breakdown(
        search_results)

    try:
        extracted_keyphrase_per_topic, lda_dataframe_output, coherence_score = lda_service.get_lda_keyphrases_and_dataframe(
            search_results)
    except Exception as e:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"status": "error", "message": repr(e)}

    try:
        topics_over_time = lda_service.parse_lda_ouput_trend_response(
            extracted_keyphrase_per_topic, lda_dataframe_output)
    except Exception as e:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"status": "error", "message": repr(e)}

    try:
        keyphrases_and_weights = lda_service.parse_extracted_kephrases_response(
            extracted_keyphrase_per_topic, lda_dataframe_output)
    except Exception as e:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"status": "error", "message": repr(e)}

    metadata["coherence_score"] = f"{coherence_score:.2f}"
    end_time = time()
    metadata["process_time"] = f"{(end_time - start_time) / 60:.2f}"
    metadata["num_of_topics"] = f"{len(extracted_keyphrase_per_topic)}"

    try:
        # Store results to db before returning response
        inserted_id = database_service.store_search_results(search.get_search_query(), lda_dataframe_output.to_json(
        ), json.dumps(keyphrases_and_weights), json.dumps(topics_over_time), json.dumps(metadata))
    except Exception as e:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"status": "error", "message": repr(e)}

    return {"status": "success", "id": inserted_id, "extracted_topics_and_weights": keyphrases_and_weights, "extracted_topics_over_time": topics_over_time, "metadata": metadata}


@app.get("/search_results")
async def get_search_results(response: Response):
    try:
        # Store results to db before returning response
        results = database_service.get_general_search_results()
    except Exception as e:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"status": "error", "message": repr(e)}

    return {"status": "success", "results": results}


@app.get("/search_result/{id}")
async def get_specific_search_result(id: int, divide_n: Optional[int] = None, response: Response = Response):
    try:
        # Store results to db before returning response
        result = database_service.get_specific_search_result(id)
    except Exception as e:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"status": "error", "message": repr(e)}

    if divide_n and isinstance(result, dict):
        result["extracted_topics_over_time"] = lda_service.parse_lda_json_trend_response(
            json.loads(result["extracted_topics_and_weights"]), result["document_df"], divide_n)
        result["extracted_topics_over_time"] = json.dumps(
            result["extracted_topics_over_time"])

    return {"status": "success", "result": result}


@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    if model_name is ModelName.alexnet:
        return {"model_name": model_name, "message": "Deep Learning FTW!"}

    if model_name.value == "lenet":
        return {"model_name": model_name, "message": "LeCNN all the images"}

    return {"model_name": model_name, "message": "Have some residuals"}
