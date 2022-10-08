from datetime import datetime, tzinfo
from . import ldapipeline
import dateutil.parser as dateparser
import pandas as pd


def get_lda_keyphrases_and_dataframe(corpus_dict):
    # Preprocess received text
    preprocessor = ldapipeline.Preprocessor(corpus=corpus_dict["abstract"])
    preprocessor.tokenize_corpus()
    preprocessor.cleanse_tokens()
    preprocessor.remove_stopwords_tokens()
    preprocessor.lemmatize_tokens()
    preprocessed_corpus = preprocessor.get_preprocessed_corpus()

    # Vectorize text, convert text to its numeric representation
    text_vectorizer = ldapipeline.TextVectorization()
    text_vectorizer.fit(preprocessed_corpus)
    training_data = text_vectorizer.transform(preprocessed_corpus)

    # Build LDA model
    lda = ldapipeline.LDAModel(training_data, verbose=True)
    lda.fit_corpus(corpus_dict)
    lda.fit_vectorizer(text_vectorizer.get_vectorizer())
    # lda.fit_best_model_by_coherence(manual=False)
    # lda.fit_best_model_by_optuna()
    lda.fit_best_model_by_hyperopt()
    lda_topic_keyphrases_and_weights = lda.get_top_n_words(top_n=25)
    lda_dataframe_output = lda.get_model_output_dataframe(save_to_csv=True)

    # Extract keyphrases from the top words from each topic and filter them
    lda_labeler = ldapipeline.LDALabeler(verbose=True)
    lda_labeler.fit_topic_to_words(lda_topic_keyphrases_and_weights)
    lda_labeler.extract_keywords_from_topics()
    # lda_labeler.filter_topic_keyphrases_by_wikipedia()
    lda_labeler.filter_topic_keyphrases_by_wikipedia_concurrent_map()
    extracted_keyphrase_per_topic = lda_labeler.get_extracted_keyphrases()

    # Parse extracted keyphrases into a proper json response
    # keyphrases_and_weights_per_topic = __parse_extracted_kephrases_response(extracted_keyphrase_per_topic, lda_dataframe_output)
    return extracted_keyphrase_per_topic, lda_dataframe_output, lda.get_coherence_score()


def parse_lda_ouput_trend_response(extracted_keyphrase_per_topic, lda_output, interval=5):
    lda_output["timestamp"] = lda_output["timestamp"].apply(lambda x: dateparser.parse(
        x, ignoretz=True, fuzzy=True) if not isinstance(x, datetime) else x)
    lda_output["timestamp"] = lda_output["timestamp"].apply(
        lambda x: x.replace(tzinfo=None))

    start_date = min(lda_output["timestamp"])
    end_date = max(lda_output["timestamp"])
    date_range = get_n_contiguous_interval_date(start_date, end_date, interval)

    topic_frequencies_over_time = {}
    for topic_index, keyphrases_and_weights in extracted_keyphrase_per_topic.items():
        # If the current topic has no frequencies at all for all timestamps, then skip it.
        if len(lda_output[lda_output["assigned_topic"] == topic_index].index) <= 0:
            print(f"Topic {topic_index} does not have any document frequency!")
            continue

        topic_frequencies_over_time[f"topic_{topic_index}"] = {
            "main_keyphrase": keyphrases_and_weights[0][0],
            "frequencies": []
        }

        # For n contiguous date ranges, get the frequencies of each topic.
        # For example, date ranges are = [2022-01-01, 2022-02-01, 2022-03-01], then this would
        # Retrive frequencies of topic X on timestamp <= 2022-01-01, 2022-01-02 - 2022-02-01, 2022-02-02 - 2022-03-01
        for i, date in enumerate(date_range):
            date = date.replace(hour=23, minute=59, second=59)
            if i == 0:
                doc_frequency = len(lda_output[(lda_output["assigned_topic"] == topic_index) & (
                    lda_output["timestamp"] <= date)].index)
            else:
                previous_date = date_range[i -
                                           1].replace(hour=23, minute=59, second=59)
                doc_frequency = len(lda_output[(lda_output["assigned_topic"] == topic_index) & (
                    lda_output["timestamp"] > previous_date) & (lda_output["timestamp"] <= date)].index)

            topic_frequency = {
                "frequency": doc_frequency,
                "timestamp": date.strftime("%Y-%m-%d")
            }
            topic_frequencies_over_time[f"topic_{topic_index}"]["frequencies"].append(
                topic_frequency)

    return topic_frequencies_over_time


def parse_extracted_kephrases_response(extracted_keyphrase_per_topic, lda_output):
    keyphrases_and_weights_per_topic = {}
    for topic_index, keyphrases_and_weights in extracted_keyphrase_per_topic.items():
        doc_frequency = len(
            lda_output[lda_output["assigned_topic"] == topic_index].index)
        if doc_frequency <= 0:
            print(
                f"Topic {topic_index} with keyphrase {keyphrases_and_weights[0][0],} does not have any document frequency!")
            continue

        # First keyphrase is the most representative keyphrase, as it has the heaviest weight
        keyphrases_and_weights_per_topic[f"topic_{topic_index}"] = {
            "main_keyphrase": keyphrases_and_weights[0][0],
            "keyphrases": [],
            "doc_frequency": doc_frequency
        }

        # Append the reset of keyphrases to the keyphrases object.
        for keyphrase, weight in keyphrases_and_weights[1:]:
            keyphrases_and_weights_per_topic[f"topic_{topic_index}"]["keyphrases"].append(
                {"name": keyphrase, "weight": weight})

    # # Impute values for doc frequency <= 0
    # min_freq = min(d['doc_frequency']
    #                for d in keyphrases_and_weights_per_topic.values() if d['doc_frequency'] > 0)
    # new_min_freq = min_freq
    # if min_freq > 1:
    #     new_min_freq -= 1
    # for topic_index in keyphrases_and_weights_per_topic:
    #     if keyphrases_and_weights_per_topic[topic_index]["doc_frequency"] <= 0:
    #         keyphrases_and_weights_per_topic[topic_index]["doc_frequency"] = new_min_freq

    return keyphrases_and_weights_per_topic


def parse_lda_json_trend_response(json_keyphrase_per_topic, lda_output, interval=5):
    lda_output = pd.read_json(lda_output)
    lda_output["timestamp"] = lda_output["timestamp"].apply(lambda x: dateparser.parse(
        x, ignoretz=True, fuzzy=True) if not isinstance(x, datetime) else x)
    lda_output["timestamp"] = lda_output["timestamp"].apply(
        lambda x: x.replace(tzinfo=None))

    start_date = min(lda_output["timestamp"])
    end_date = max(lda_output["timestamp"])
    date_range = get_n_contiguous_interval_date(start_date, end_date, interval)

    topic_frequencies_over_time = {}
    for topic_index, keyphrases_and_weights in json_keyphrase_per_topic.items():
        actual_topic_index = int(''.join(filter(str.isdigit, topic_index)))
        # If the current topic has no frequencies at all for all timestamps, then skip it.
        if len(lda_output[lda_output["assigned_topic"] == actual_topic_index].index) <= 0:
            print(f"Topic {topic_index} does not have any document frequency!")
            continue

        topic_frequencies_over_time[topic_index] = {
            "main_keyphrase": keyphrases_and_weights["main_keyphrase"],
            "frequencies": []
        }

        # For n contiguous date ranges, get the frequencies of each topic.
        # For example, date ranges are = [2022-01-01, 2022-02-01, 2022-03-01], then this would
        # Retrive frequencies of topic X on timestamp <= 2022-01-01, 2022-01-02 - 2022-02-01, 2022-02-02 - 2022-03-01
        for i, date in enumerate(date_range):
            date = date.replace(hour=23, minute=59, second=59)
            if i == 0:
                doc_frequency = len(lda_output[(lda_output["assigned_topic"] == actual_topic_index) & (
                    lda_output["timestamp"] <= date)].index)
            else:
                previous_date = date_range[i -
                                           1].replace(hour=23, minute=59, second=59)
                doc_frequency = len(lda_output[(lda_output["assigned_topic"] == actual_topic_index) & (
                    lda_output["timestamp"] > previous_date) & (lda_output["timestamp"] <= date)].index)

            topic_frequency = {
                "frequency": doc_frequency,
                "timestamp": date.strftime("%Y-%m-%d")
            }
            topic_frequencies_over_time[topic_index]["frequencies"].append(
                topic_frequency)

    return topic_frequencies_over_time


def get_n_contiguous_interval_date(start, end, interval=5):
    dates = []
    diff = (end - start) / interval
    for i in range(interval):
        dates.append(start + diff * i)
    dates.append(end)
    return dates
