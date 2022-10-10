import mysql.connector

db = None
db_cursor = None

db_closed = False
cursor_closed = False

INSERT_SEARCH_RESULTS_QUERY = """
INSERT INTO search_results(search_query, document_dataframe, topics_and_weights, topics_over_time, metadata) VALUES (%s, %s, %s, %s, %s)
"""

GET_SEARCH_RESULTS_QUERY = """
SELECT id, search_query, created_at FROM search_results
"""

GET_SPECIFIC_SEARCH_RESULT_QUERY = """
SELECT id, search_query, document_dataframe, topics_and_weights, topics_over_time, metadata, created_at FROM search_results WHERE id = %s
"""


def __get_db():
    global db_closed
    global db
    if not db_closed and db:
        return db
    db = mysql.connector.connect(
        host="localhost", user="root", password="Ldathesis123*", database="lda_thesis_db")
    if db.is_connected():
        db_closed = False
        return db
    raise Exception("failed to connect to database!")


def __get_db_cursor():
    global cursor_closed
    global db_cursor
    if not cursor_closed and db_cursor:
        return db_cursor
    db_cursor = __get_db().cursor()
    cursor_closed = False
    return db_cursor


def insert_data_to_search_results(search_query, document_df, topics_and_weights, topics_over_time, metadata):
    cursor = __get_db_cursor()
    cursor.execute(INSERT_SEARCH_RESULTS_QUERY, params=(
        search_query, document_df, topics_and_weights, topics_over_time, metadata))
    __get_db().commit()
    return cursor.lastrowid


def get_search_results():
    cursor = __get_db_cursor()
    cursor.execute(GET_SEARCH_RESULTS_QUERY)
    return cursor.fetchall()


def get_specific_search_result(id):
    cursor = __get_db_cursor()
    cursor.execute(GET_SPECIFIC_SEARCH_RESULT_QUERY, params=[id])
    return cursor.fetchone()


def close_db_and_cursor():
    global db_closed
    global cursor_closed
    __get_db_cursor().close()
    __get_db().close()
    cursor_closed = True
    db_closed = True


def main():
    print(insert_data_to_search_results("test", "{}", "{}", "{}", "{}"))
    __get_db_cursor().close()
    __get_db().close()


if __name__ == "__main__":
    main()
