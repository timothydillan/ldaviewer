# Numpy
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE
import abstractscraper
from cgitb import text
from random import random
import numpy as np
import pandas as pd
import re
import nltk
import spacy
import math
from scipy.stats import uniform

# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Keyphrase Vectorizer
from keyphrase_vectorizers import KeyphraseCountVectorizer, KeyphraseTfidfVectorizer

# Plotting tools
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt

# Gensim
import gensim

# NLTK
import nltk

# Hyperparameter Tuning
# Hyperopt
from hyperopt import atpe, tpe, hp, fmin, STATUS_OK, Trials, rand
from hyperopt.pyll.base import scope
from joblib import Parallel, delayed

# Scikit Optimize
from skopt.space import Integer
from skopt.space import Real
from skopt.space import Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize

# Optuna
import optuna

import time

# KeyBERT for Keyphrase Extraction
from keybert import KeyBERT

# ConceptNet
import conceptnet_lite
from conceptnet_lite import Label, edges_between

# Wikipedia
import wikipedia

# Utils
import itertools
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


class Preprocessor:
    # Preprocessing options that will be done on the corpus.
    PREPROCESSING_OPTIONS = [
        lambda x: x.lower(),
        gensim.parsing.preprocessing.strip_tags,
        gensim.parsing.preprocessing.strip_punctuation,
        gensim.parsing.preprocessing.strip_multiple_whitespaces,
        gensim.parsing.preprocessing.strip_non_alphanum,
        gensim.parsing.preprocessing.strip_numeric,
        gensim.parsing.preprocessing.strip_multiple_whitespaces,
        gensim.parsing.preprocessing.remove_stopwords
    ]

    EN_STOPWORDS = {"".join(list(gensim.utils.tokenize(doc, lower=True)))
                    for doc in nltk.corpus.stopwords.words('english')}

    SPACY_MODEL = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    def __init__(self, corpus):
        self.corpus = corpus
        self.tokenized_corpus = None

    # Validate whether corpus is already tokenized.
    def __validate_corpus_tokenized(self):
        if self.tokenized_corpus is None:
            raise Exception(
                "Tokens have not been created yet! Please use the `tokenize_corpus` method first!")

    # Tokenizes the corpus. It will also conduct case folding (lowercase) and accent removals.
    def tokenize_corpus(self):
        self.tokenized_corpus = [list(gensim.utils.tokenize(
            doc, lower=True, deacc=True)) for doc in self.corpus]

    # Cleanses the tokens with the prespecificed preprocessing options constant.
    def cleanse_tokens(self):
        self.__validate_corpus_tokenized()
        self.tokenized_corpus = [gensim.parsing.preprocessing.preprocess_string(
            " ".join(doc), self.PREPROCESSING_OPTIONS) for doc in self.tokenized_corpus]

    # Lemmatize tokens to reduce each token (word) to its base form.
    def lemmatize_tokens(self, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        self.__validate_corpus_tokenized()
        lemmatized_tokens = []
        for tokens in self.tokenized_corpus:
            doc = self.SPACY_MODEL(" ".join(tokens))
            if len(allowed_postags) > 0:
                lemmatized_tokens.append(
                    [token.lemma_ for token in doc if token.pos_ in allowed_postags])
            else:
                lemmatized_tokens.append([token.lemma_ for token in doc])
        self.tokenized_corpus = lemmatized_tokens

    # Remove stopwords from tokens.
    def remove_stopwords_tokens(self):
        self.__validate_corpus_tokenized()
        no_stopwords_tokens = []
        for tokens in self.tokenized_corpus:
            tokens = [token for token in tokens if token not in self.EN_STOPWORDS]
            no_stopwords_tokens.append(tokens)
        self.tokenized_corpus = no_stopwords_tokens

    # Get the final preprocessed corpus.
    def get_preprocessed_corpus(self):
        self.__validate_corpus_tokenized()
        preprocessed_corpus = []
        for tokens in self.tokenized_corpus:
            preprocessed_corpus.append(" ".join(token for token in tokens))
        return preprocessed_corpus


class TextVectorization:
    KEYPHRASE_POS_PATTERN = r"""
CHUNK: 
    {<J.*><N.*><N.*>|<J.*><N.*>|<J.*><N.*>+}
    {<N.*><N.*>}
    {<N.*><IN.*><N.*>}
"""

    MAX_DOCUMENT_FREQUENCY_RATIO = 0.9

    def __init__(self):
        self.vectorizer = None

    def fit(self, corpus):
        self.vectorizer = KeyphraseCountVectorizer(max_df=int(len(
            corpus)*self.MAX_DOCUMENT_FREQUENCY_RATIO), pos_pattern=self.KEYPHRASE_POS_PATTERN).fit(corpus)
        return self

    def transform(self, corpus):
        return self.vectorizer.transform(corpus)

    def get_vectorizer(self):
        return self.vectorizer


class LDAModel:
    MAX_ITERATIONS = 50
    N_JOBS = -1
    TOP_N_WORDS = 10

    RANDOM_STATE = 420

    def __init__(self, training_data, verbose=False):
        self.training_data = training_data
        self.verbose = verbose
        self.lda_model = None
        self.vectorizer = None
        self.corpus_dict = None

    def __validate_training_data(self):
        if self.training_data is None:
            raise Exception(
                "Training data has not been passed yet! Please initialize the instance with training data!")

    def __validate_model_ready(self):
        if self.lda_model is None:
            raise Exception(
                "LDA Model has not been trained! Please fit the model!")

    def __validate_vectorizer_loaded(self):
        if self.vectorizer is None:
            raise Exception(
                "Vectorizer has not been loaded! Please load the vectorizer!")

    def __validate_corupus_loaded(self):
        if self.corpus_dict is None:
            raise Exception(
                "Corpus has not been loaded! Please load the corpus!")

    def fit_vectorizer(self, vectorizer):
        self.vectorizer = vectorizer

    def fit_corpus(self, corpus_dict):
        self.corpus_dict = corpus_dict

    def __get_umass_score(self, dt_matrix, i, j):
        zo_matrix = (dt_matrix > 0).astype(int)
        col_i, col_j = zo_matrix[:, i], zo_matrix[:, j]
        col_ij = col_i + col_j
        col_ij = (col_ij == 2).astype(int)
        # Dwi = number of documents containing the word wi
        # Dwij = number of documents containing the word wi and wj
        # Umass measure = conditional log-probability log p(wj|wi) = log p(wi,wj)/p(wj) smoothed by adding one to D(wi, wj).
        Di, Dij = col_i.sum(), col_ij.sum()
        return math.log((Dij + 1) / Di)

    def __get_topic_coherence(self, dt_matrix, topic, n_top_words):
        indexed_topic = zip(topic, range(0, len(topic)))
        topic_top = sorted(
            indexed_topic, key=lambda x: 1 - x[0])[0:n_top_words]
        coherence = 0
        for j_index in range(0, len(topic_top)):
            for i_index in range(0, j_index - 1):
                i = topic_top[i_index][1]
                j = topic_top[j_index][1]
                coherence += self.__get_umass_score(dt_matrix, i, j)
        return coherence

    def get_average_topic_coherence(self, dt_matrix, topics, n_top_words):
        total_coherence = 0
        for i in range(0, len(topics)):
            total_coherence += self.__get_topic_coherence(
                dt_matrix, topics[i], n_top_words)
        return total_coherence / len(topics)

    def __objective(self, model_parameters):
        lda = LatentDirichletAllocation(**model_parameters, max_iter=self.MAX_ITERATIONS,
                                        n_jobs=self.N_JOBS, random_state=self.RANDOM_STATE).fit(self.training_data)
        coherence = self.get_average_topic_coherence(
            self.training_data, lda.components_, self.TOP_N_WORDS)
        # Since its a minimization function, the negative version of coherence should be used.
        return -coherence

    def __optuna_objective(self, trial):
        search_space = {
            "n_components": trial.suggest_int('n_components', 2, 30),
            "doc_topic_prior": trial.suggest_uniform("doc_topic_prior", 0.001, 5),
            "topic_word_prior": trial.suggest_uniform("topic_word_prior", 0.001, 5),
            "learning_decay": trial.suggest_discrete_uniform("learning_decay", 0.5, 1.0, 0.1),
        }
        lda = LatentDirichletAllocation(**search_space, max_iter=self.MAX_ITERATIONS,
                                        n_jobs=self.N_JOBS, random_state=self.RANDOM_STATE).fit(self.training_data)
        coherence = self.get_average_topic_coherence(
            self.training_data, lda.components_, self.TOP_N_WORDS)
        return coherence

    def fit(
            self,
            k=2,
            max_iter=MAX_ITERATIONS,
            n_jobs=N_JOBS):
        self.__validate_training_data()
        self.lda_model = LatentDirichletAllocation(
            n_components=k, max_iter=max_iter, n_jobs=n_jobs, random_state=self.RANDOM_STATE).fit(self.training_data)
        return self

    def fit_best_model_by_coherence(self, start=2, end=20, manual=True):
        self.__validate_training_data()
        coherence_scores = []

        for k in range(start, end+1):
            lda = LatentDirichletAllocation(n_components=k, max_iter=self.MAX_ITERATIONS,
                                            n_jobs=self.N_JOBS, random_state=self.RANDOM_STATE).fit(self.training_data)
            coherence_score = self.get_average_topic_coherence(
                self.training_data, lda.components_, self.TOP_N_WORDS)
            coherence_scores.append(coherence_score)
            if self.verbose:
                print("K =", k)
                print(
                    f"Coherence = {coherence_score} | Log Likelihood = {lda.score(self.training_data)} | Perplexity = {lda.perplexity(self.training_data)}")

        best_k = coherence_scores.index(max(coherence_scores)) + start
        if self.verbose:
            plt.title("Coherence Scores")
            plt.plot([k for k in range(start, end+1)], coherence_scores)
            plt.show()
            print("Best K is", best_k)

        self.lda_model = LatentDirichletAllocation(
            n_components=best_k, max_iter=self.MAX_ITERATIONS, n_jobs=self.N_JOBS, random_state=self.RANDOM_STATE).fit(self.training_data)

    def fit_best_model_by_hyperopt(self, start=2, end=30):
        self.__validate_training_data()
        self.__validate_vectorizer_loaded()
        trials = Trials()

        learning_decay_choices = np.arange(float(0.5), float(1.0), float(0.1))
        search_space = {
            "n_components": hp.randint("n_components", start, end),
            "doc_topic_prior": hp.uniform("doc_topic_prior", 0.001, 5),
            "topic_word_prior": hp.uniform("topic_word_prior", 0.001, 5),
            "learning_decay": hp.choice("learning_decay", learning_decay_choices),
        }

        best = fmin(
            fn=self.__objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=50,
            trials=trials
        )

        if self.verbose:
            print("Best:", best)
            # print(trials.results)
        best_params = best
        best_params['learning_decay'] = learning_decay_choices[best['learning_decay']]
        self.lda_model = LatentDirichletAllocation(
            **best_params, max_iter=self.MAX_ITERATIONS, n_jobs=self.N_JOBS, random_state=self.RANDOM_STATE).fit(self.training_data)
        if self.verbose:
            print("Params:", self.lda_model.get_params())

    def fit_best_model_by_skopt(self, start=2, end=30):
        self.__validate_training_data()
        self.__validate_vectorizer_loaded()
        start_time = time.time()
        search_space = [
            Real(0.001, 5, 'uniform', name='doc_topic_prior'),
            Real(0.001, 5, 'uniform', name='topic_word_prior'),
            Integer(start, end, name='n_components'),
            Categorical(np.arange(float(0.5), float(1.0), float(0.1)),
                        name='learning_decay')
        ]

        @use_named_args(dimensions=search_space)
        def objective_wrapper(*args, **kwargs):
            return self.__objective(kwargs)

        result = gp_minimize(objective_wrapper, search_space)

        if self.verbose:
            print('Best Accuracy: %.3f' % (1.0 - result.fun))
            print('Best Parameters: %s' % (result.x))
            end_time = time.time()
            print("Time elapsed:", end_time - start_time)

    def fit_best_model_by_optuna(self, start=2, end=30):
        self.__validate_training_data()
        self.__validate_vectorizer_loaded()
        study = optuna.create_study(direction='maximize')
        study.optimize(self.__optuna_objective, n_trials=50)
        best_trial = study.best_trial
        if self.verbose:
            print('Best coherence: {}'.format(best_trial.value))
            print("Best hyperparameters: {}".format(best_trial.params))
            # optuna.visualization.plot_optimization_history(study)
            # optuna.visualization.plot_slice(study)
        self.lda_model = LatentDirichletAllocation(
            **best_trial.params, max_iter=self.MAX_ITERATIONS, n_jobs=self.N_JOBS, random_state=self.RANDOM_STATE).fit(self.training_data)
        if self.verbose:
            print("Params:", self.lda_model.get_params())

    def get_model(self):
        self.__validate_model_ready()
        return self.lda_model

    def get_top_n_words(self, top_n=10):
        self.__validate_model_ready()
        self.__validate_vectorizer_loaded()
        topic_word_map = {}
        for topic_index, component in enumerate(self.lda_model.components_):
            vocab_comp = zip(
                self.vectorizer.get_feature_names_out(), component)
            sorted_words = sorted(
                vocab_comp, key=lambda x: x[1], reverse=True)[:top_n]
            topic_word_map[topic_index] = sorted_words
            if self.verbose:
                print(f"Topic {topic_index}:")
                for word in sorted_words:
                    print(word[0], end=" ")
                print("\n")
        return topic_word_map

    def get_model_output_dataframe(self, save_to_csv=False):
        self.__validate_corupus_loaded()
        self.__validate_training_data()
        self.__validate_model_ready()
        # standardize array values
        self.corpus_dict["published_date"] = self.corpus_dict["published_date"][:len(
            self.corpus_dict["abstract"])]

        # lda_output = self.get_model().transform(self.training_data)
        # r, c = np.where(lda_output >= 0.5)
        # assigned_topics = np.split(c,np.searchsorted(r,range(1, lda_output.shape[0])))

        assigned_topics = np.argmax(
            self.get_model().transform(self.training_data), axis=1)
        df = pd.DataFrame.from_dict(
            {"documents": self.corpus_dict["abstract"], "assigned_topic": assigned_topics, "timestamp": self.corpus_dict["published_date"]})
        if save_to_csv:
            df.to_csv("model_output.csv")
        return df


class LDALabeler:
    def __init__(self, verbose=False):
        self.keybert_model = KeyBERT()
        self.verbose = verbose
        # Initialize ConceptNet database
        conceptnet_lite.connect(
            "/Users/timothy.dillan/Downloads/conceptnet.db")
        self.topic_keyphrases = {}

    def __validate_topic_keyphrases(self):
        if not self.topic_keyphrases:
            raise Exception(
                "Topic keyphrases has not been loaded! Please load them using the `extract_keywords_from_topics` method!")

    def fit_topic_to_words(self, topic_to_words):
        self.topic_keyphrases = topic_to_words

    # Outputs: {0: [('keyphrase', weight), ..., ('keyphrase', weight)]}
    def extract_keywords_from_topics(self):
        keybert_topic_keypharses = {}
        for topic_index, keyphrases_and_weights in self.topic_keyphrases.items():
            keybert_topic_keypharses[topic_index] = self.keybert_model.extract_keywords(" ".join(
                [i[0] for i in keyphrases_and_weights]), top_n=10, vectorizer=KeyphraseCountVectorizer(pos_pattern=TextVectorization.KEYPHRASE_POS_PATTERN))
        if self.verbose:
            print("Extracted keyphrases:", self.topic_keyphrases)
        self.topic_keyphrases = keybert_topic_keypharses

    def filter_topic_keyphrases_by_conceptnet(self):
        self.__validate_topic_keyphrases()
        filtered_keyphrases = {}
        # For each keyphrase that we'd like to verify (whether its a concept or not)
        for topic_index, keyphrases_and_weights in self.topic_keyphrases.items():
            filtered_keyphrases[topic_index] = []
            for keyphrase_and_weight in keyphrases_and_weights:
                keyphrase = keyphrase_and_weight[0]
                weight = keyphrase_and_weight[1]
                # Reformat the keyphrase (if it has more than one word) into ConceptNet acceptable format.
                if len(keyphrase.split()) > 1:
                    new_keyphrase = "_".join(keyphrase.split())
                # Try to check the concept from the keyphrase.
                try:
                    # If it does exist, add the keyphrase.
                    concepts = Label.get(text=new_keyphrase).concepts
                    filtered_keyphrases[topic_index].append(
                        (keyphrase.capitalize(), weight))
                except:
                    # Else, if it doesn't and the keyphrase only contains one word, no concept was found.
                    if len(keyphrase.split()) < 2:
                        print("No concept found for", keyphrase)
                        continue

                    # Else, for every possible n-gram combination from 2 - len(keyphrase), check every possible n-gram combination to ConceptNet
                    # and repeat the previous process.
                    start = 2
                    end = len(keyphrase.split())
                    if len(keyphrase.split()) > 2:
                        end -= 1

                    for split in range(start, end+1):
                        for ngram in itertools.permutations(keyphrase.split(), split):
                            new_keyphrase = "_".join(ngram)
                            try:
                                concepts = Label.get(
                                    text=new_keyphrase).concepts
                                new_formatted_keyphrase = " ".join(
                                    ngram).capitalize()
                                filtered_keyphrases[topic_index].append(
                                    (new_formatted_keyphrase, weight))
                            except:
                                print("No concept found for", new_keyphrase)

            # If the filtration resulted in 0 results, then fallback to default keyphrases and weights.
            if len(filtered_keyphrases[topic_index]) <= 0:
                if self.verbose:
                    print(
                        f"No concepts were found for all the keyphrases provided in topic {topic_index}. Returning the provided keyphrases instead.")
                filtered_keyphrases[topic_index] = keyphrases_and_weights

        self.topic_keyphrases = filtered_keyphrases

    def filter_topic_keyphrases_by_wikipedia(self):
        self.__validate_topic_keyphrases()
        filtered_keyphrases = {}
        # For each keyphrase that we'd like to verify (whether its a concept or not)
        for topic_index, keyphrases_and_weights in self.topic_keyphrases.items():
            filtered_keyphrases[topic_index] = []
            for keyphrase_and_weight in keyphrases_and_weights:
                keyphrase = keyphrase_and_weight[0]
                weight = keyphrase_and_weight[1]

                wikipedia_results = wikipedia.search(
                    keyphrase, results=1, suggestion=False)
                if len(wikipedia_results) > 0:
                    filtered_keyphrases[topic_index].append(
                        (keyphrase.capitalize(), weight))
                    continue

                # Else, if it doesn't and the keyphrase only contains one word, no concept was found.
                if len(keyphrase.split()) < 2:
                    print("No concept found for", keyphrase)
                    continue

                # Else, for every possible n-gram combination from 2 - len(keyphrase), check every possible n-gram combination to ConceptNet
                # and repeat the previous process.
                start = 2
                end = len(keyphrase.split())
                if len(keyphrase.split()) > 2:
                    end -= 1

                for split in range(start, end+1):
                    for ngram in itertools.permutations(keyphrase.split(), split):
                        new_keyphrase = " ".join(ngram)
                        wikipedia_results = wikipedia.search(
                            keyphrase, results=1, suggestion=False)
                        if len(wikipedia_results) > 0:
                            filtered_keyphrases[topic_index].append(
                                (keyphrase.capitalize(), weight))
                            continue
                        print("No concept found for", new_keyphrase)

            # If the filtration resulted in 0 results, then fallback to default keyphrases and weights.
            if len(filtered_keyphrases[topic_index]) <= 0:
                if self.verbose:
                    print(
                        f"No concepts were found for all the keyphrases provided in topic {topic_index}. Returning the provided keyphrases instead.")
                filtered_keyphrases[topic_index] = keyphrases_and_weights

        self.topic_keyphrases = filtered_keyphrases

    def get_extracted_keyphrases(self):
        self.__validate_topic_keyphrases()
        return self.topic_keyphrases


def parse_extracted_kephrases_response(extracted_keyphrase_per_topic):
    keyphrases_and_weights_per_topic = {}
    for topic_index, keyphrases_and_weights in extracted_keyphrase_per_topic.items():
        keyphrases_and_weights_per_topic[f"topic_{topic_index}"] = {
            "main_keyphrase": ""
        }
        # First keyphrase is the most representative keyphrase
        keyphrases_and_weights_per_topic[f"topic_{topic_index}"]["main_keyphrase"] = keyphrases_and_weights[0][0]
        # Append the reset of keyphrases to the keyphrases object.
        keyphrases_and_weights_per_topic[f"topic_{topic_index}"]["keyphrases"] = [
        ]
        for keyphrase, weight in keyphrases_and_weights[1:]:
            keyphrases_and_weights_per_topic[f"topic_{topic_index}"]["keyphrases"].append(
                {"name": keyphrase, "weight": weight})
    return keyphrases_and_weights_per_topic


def combine_extracted_keyphrases_and_topic_frequency(keyphrases_and_weights_per_topic, lda_dataframe):
    empty_topic_indexes = []
    for topic_index in keyphrases_and_weights_per_topic:
        actual_topic_index = int(''.join(filter(str.isdigit, topic_index)))
        doc_frequency = len(
            lda_dataframe[lda_dataframe["assigned_topic"] == actual_topic_index].index)
        if doc_frequency > 0:
            keyphrases_and_weights_per_topic[topic_index]["doc_frequency"] = doc_frequency
        else:
            print(
                f"Topic {topic_index} with keyphrase {keyphrases_and_weights_per_topic[topic_index]['main_keyphrase']} does not have any document frequency!")
            empty_topic_indexes.append(topic_index)

    for topic_index in empty_topic_indexes:
        keyphrases_and_weights_per_topic.pop(topic_index)

    # # Impute values
    # min_freq = min(d['doc_frequency']
    #                for d in keyphrases_and_weights_per_topic.values() if d['doc_frequency'] > 0)
    # new_min_freq = min_freq
    # if min_freq > 1:
    #     new_min_freq -= 1
    # for topic_index in keyphrases_and_weights_per_topic:
    #     if keyphrases_and_weights_per_topic[topic_index]["doc_frequency"] <= 0:
    #         keyphrases_and_weights_per_topic[topic_index]["doc_frequency"] = new_min_freq

    return keyphrases_and_weights_per_topic


def pipeline(corpus_dict):
    # Preprocess received text
    preprocessor = Preprocessor(corpus=corpus_dict["abstract"])
    preprocessor.tokenize_corpus()
    preprocessor.cleanse_tokens()
    preprocessor.remove_stopwords_tokens()
    preprocessor.lemmatize_tokens()
    preprocessed_corpus = preprocessor.get_preprocessed_corpus()

    # Vectorize text, convert text to its numeric representation
    text_vectorizer = TextVectorization()
    text_vectorizer.fit(preprocessed_corpus)
    training_data = text_vectorizer.transform(preprocessed_corpus)

    # Build LDA model
    lda = LDAModel(training_data, verbose=True)
    lda.fit_corpus(corpus_dict)
    lda.fit_vectorizer(text_vectorizer.get_vectorizer())
    # lda.fit_best_model_by_coherence(manual=False)
    # lda.fit_best_model_by_optuna()
    # lda.fit_best_model_by_skopt()
    lda.fit_best_model_by_hyperopt()
    lda_topic_keyphrases_and_weights = lda.get_top_n_words(top_n=25)

    # Extract keyphrases from the top words from each topic and filter them
    lda_labeler = LDALabeler(verbose=True)
    lda_labeler.fit_topic_to_words(lda_topic_keyphrases_and_weights)
    lda_labeler.extract_keywords_from_topics()
    lda_labeler.filter_topic_keyphrases_by_wikipedia()
    extracted_keyphrase_per_topic = lda_labeler.get_extracted_keyphrases()

    # Parse extracted keyphrases into a proper json response
    keyphrases_and_weights_per_topic = parse_extracted_kephrases_response(
        extracted_keyphrase_per_topic)
    keyphrase_weight_and_frequency_per_topic = combine_extracted_keyphrases_and_topic_frequency(
        keyphrases_and_weights_per_topic, lda.get_model_output_dataframe(save_to_csv=True))
    return keyphrase_weight_and_frequency_per_topic


# def main():
#     # Replace with query from user
#     # corpus = pd.read_csv('abstracts.csv')
#     # corpus.abstract = corpus.abstract.apply(lambda x: x.split(".Comment")[0])
#     core_db = abstractscraper.CORE()
#     core_db.get_works_by_search_query(
#         """"image recognition" AND fieldsOfStudy:"computer science" AND documentType:"research" AND (yearPublished>=2010 AND yearPublished<=2022)""", limit=10)
#     corpus_dict = core_db.get_data_from_works()
#     print(corpus_dict)

#     # Preprocess received text
#     preprocessor = Preprocessor(corpus=corpus_dict["abstract"])
#     preprocessor.tokenize_corpus()
#     preprocessor.cleanse_tokens()
#     preprocessor.remove_stopwords_tokens()
#     preprocessor.lemmatize_tokens()
#     preprocessed_corpus = preprocessor.get_preprocessed_corpus()

#     # Vectorize text, convert text to its numeric representation
#     text_vectorizer = TextVectorization()
#     text_vectorizer.fit(preprocessed_corpus)
#     training_data = text_vectorizer.transform(preprocessed_corpus)

#     # Build LDA model
#     lda = LDAModel(training_data, verbose=True)
#     lda.fit_corpus(corpus_dict)
#     lda.fit_vectorizer(text_vectorizer.get_vectorizer())
#     # lda.fit_best_model_by_coherence(manual=False)
#     # lda.fit_best_model_by_optuna()
#     # lda.fit_best_model_by_skopt()
#     lda.fit_best_model_by_hyperopt()
#     lda_topic_keyphrases_and_weights = lda.get_top_n_words(top_n=25)

#     # Extract keyphrases from the top words from each topic and filter them
#     lda_labeler = LDALabeler(verbose=True)
#     lda_labeler.fit_topic_to_words(lda_topic_keyphrases_and_weights)
#     lda_labeler.extract_keywords_from_topics()
#     lda_labeler.filter_topic_keyphrases_by_wikipedia()
#     extracted_keyphrase_per_topic = lda_labeler.get_extracted_keyphrases()

#     # Parse extracted keyphrases into a proper json response
#     keyphrases_and_weights_per_topic = parse_extracted_kephrases_response(
#         extracted_keyphrase_per_topic)
#     print(lda.get_model_output_dataframe())
#     keyphrase_weight_and_frequency_per_topic = combine_extracted_keyphrases_and_topic_frequency(
#         keyphrases_and_weights_per_topic, lda.get_model_output_dataframe())
#     print(keyphrase_weight_and_frequency_per_topic)


# if __name__ == "__main__":
#     main()
