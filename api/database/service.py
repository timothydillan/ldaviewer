from . import database


def store_search_results(search_query, document_df, topics_and_weights, topics_over_time, metadata):
    success = database.insert_data_to_search_results(
        search_query, document_df, topics_and_weights, topics_over_time, metadata)
    database.close_db_and_cursor()
    return success


def get_general_search_results():
    results = database.get_search_results()
    database.close_db_and_cursor()
    if results:
        return [construct_search_results_response(result) for result in results]
    return None


def get_specific_search_result(id):
    result = database.get_specific_search_result(id)
    database.close_db_and_cursor()
    if result:
        return construct_specific_search_result_response(result)
    return None


def construct_specific_search_result_response(response):
    id = response[0]
    search_query = response[1]
    document_df = response[2]
    topics_and_weights = response[3]
    topics_over_time = response[4]
    metadata = response[5]
    create_time = response[6]
    json_response = {
        "id": id,
        "search_query": search_query,
        "document_df": document_df,
        "extracted_topics_and_weights": topics_and_weights,
        "extracted_topics_over_time": topics_over_time,
        "metadata": metadata,
        "create_time": create_time
    }
    return json_response


def construct_search_results_response(response):
    id = response[0]
    search_query = response[1]
    create_time = response[2]
    json_response = {
        "id": id,
        "search_query": search_query,
        "create_time": create_time
    }
    return json_response
