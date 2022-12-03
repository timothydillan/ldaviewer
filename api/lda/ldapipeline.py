# Utils
import dateutil.parser as dateparser
from datetime import datetime, tzinfo
import wikipedia
# from conceptnet_lite import Label, edges_between
# import conceptnet_lite
from keybert import KeyBERT
import time
import optuna
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.space import Categorical
from skopt.space import Real
from skopt.space import Integer
from joblib import Parallel, delayed
from hyperopt.pyll.base import scope
from hyperopt import atpe, tpe, hp, fmin, STATUS_OK, Trials, rand
import gensim
import matplotlib.pyplot as plt
import pyLDAvis.sklearn
import pyLDAvis
from keyphrase_vectorizers import KeyphraseCountVectorizer, KeyphraseTfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from scipy.stats import uniform
import math
import spacy
import nltk
import re
import pandas as pd
import numpy as np
import itertools
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from langdetect import detect
import stanfordnlp
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

# Numpy

# Sklearn

# Keyphrase Vectorizer

# Plotting tools

# Gensim

# NLTK

# Hyperparameter Tuning
# Hyperopt

# Scikit Optimize

# Optuna


# KeyBERT for Keyphrase Extraction

# ConceptNet

# Wikipedia


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
    INDO_STOPWORDS = {"".join(list(gensim.utils.tokenize(doc, lower=True)))
                      for doc in nltk.corpus.stopwords.words('indonesian')}

    STOPWORDS = EN_STOPWORDS.union(INDO_STOPWORDS)

    SPACY_MODEL = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    INDONESIAN_MODEL = stanfordnlp.Pipeline(
        lang='id', processors='tokenize,pos')

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
            docs = " ".join(tokens)
            lang = detect(docs)
            if lang == "id":
                indo_doc = self.INDONESIAN_MODEL(docs)
                for docs in indo_doc.sentences:
                    # if len(allowed_postags) > 0:
                    #     lemmatized_tokens.append(
                    #         [word.lemma for word in docs.words if word.upos in allowed_postags])
                    # else:
                    lemmatized_tokens.append(
                        [word.text for word in docs.words])
            else:
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
            tokens = [token for token in tokens if token not in self.STOPWORDS]
            no_stopwords_tokens.append(tokens)
        self.tokenized_corpus = no_stopwords_tokens

    # Get the final preprocessed corpus.
    def get_preprocessed_corpus(self):
        self.__validate_corpus_tokenized()
        preprocessed_corpus = []
        for tokens in self.tokenized_corpus:
            try:
                preprocessed_corpus.append(" ".join(token for token in tokens))
            except:
                print(tokens)
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
        self.indonesian_pipeline = stanfordnlp.Pipeline(
            lang='id', processors='tokenize,pos')
        self.en_pipeline = spacy.load(
            'en_core_web_sm', disable=['parser', 'ner'])

    # define custom POS-tagger function using flair
    def custom_pos_tagger(self, raw_documents):
        """
        Important: 

        The mandatory 'raw_documents' parameter can NOT be named differently and has to expect a list of strings. 
        Any other parameter of the custom POS-tagger function can be arbitrarily defined, depending on the respective use case. 
        Furthermore the function has to return a list of (word token, POS-tag) tuples.
        """

        # split texts into sentences
        sentences = []
        for doc in raw_documents:
            lang = detect(doc)
            sentences.append((doc, lang))

        pos_tags = []
        words = []
        for sentence in sentences:
            if sentence[1] == "id":
                indo_doc = self.indonesian_pipeline(sentence[0].lower())
                for docs in indo_doc.sentences:
                    for word in docs.words:
                        words.append(word.text)
                        pos_tags.append(word.xpos)
            else:
                doc = self.en_pipeline(sentence[0])
                for token in doc:
                    words.append(token.text)
                    pos_tags.append(token.tag_)

        return list(zip(words, pos_tags))

    def fit(self, corpus):
        self.vectorizer = KeyphraseCountVectorizer(max_df=int(len(
            corpus)*self.MAX_DOCUMENT_FREQUENCY_RATIO), custom_pos_tagger=self.custom_pos_tagger, stop_words=list(Preprocessor.STOPWORDS), pos_pattern=TextVectorization.KEYPHRASE_POS_PATTERN).fit(corpus)
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

    LEARNING_METHOD = "batch"

    def __init__(self, training_data, verbose=False):
        self.training_data = training_data
        self.verbose = verbose
        self.lda_model = None
        self.vectorizer = None
        self.corpus_dict = None
        self.coherence_score = None

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
                                        n_jobs=self.N_JOBS, random_state=self.RANDOM_STATE, learning_method=self.LEARNING_METHOD).fit(self.training_data)
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
                                        n_jobs=self.N_JOBS, random_state=self.RANDOM_STATE, learning_method=self.LEARNING_METHOD).fit(self.training_data)
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
            n_components=k, max_iter=max_iter, n_jobs=n_jobs, random_state=self.RANDOM_STATE, learning_method=self.LEARNING_METHOD).fit(self.training_data)
        return self

    def fit_best_model_by_coherence(self, start=2, end=20, manual=True):
        self.__validate_training_data()
        coherence_scores = []

        for k in range(start, end+1):
            lda = LatentDirichletAllocation(n_components=k, max_iter=self.MAX_ITERATIONS,
                                            n_jobs=self.N_JOBS, random_state=self.RANDOM_STATE, learning_method=self.LEARNING_METHOD).fit(self.training_data)
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
            n_components=best_k, max_iter=self.MAX_ITERATIONS, n_jobs=self.N_JOBS, random_state=self.RANDOM_STATE, learning_method=self.LEARNING_METHOD).fit(self.training_data)
        self.coherence_score = max(coherence_scores)

    def fit_best_model_by_hyperopt(self, start=2, end=30):
        self.__validate_training_data()
        self.__validate_vectorizer_loaded()
        trials = Trials()

        learning_decay_choices = [round(i, 2)
                                  for i in np.arange(0.5, 1.01, 0.1)]
        search_space = {
            "n_components": hp.randint("n_components", start, end),
            "doc_topic_prior": hp.uniform("doc_topic_prior", 0.001, 5),
            "topic_word_prior": hp.uniform("topic_word_prior", 0.001, 5),
        }

        if self.LEARNING_METHOD == "online":
            search_space["learning_decay"] = hp.choice(
                "learning_decay", learning_decay_choices)

        # use tpe.suggest if atpe sucks
        best = fmin(
            fn=self.__objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=50,
            trials=trials
        )

        if self.verbose:
            print("Best:", best)

        best_params = best
        if self.LEARNING_METHOD == "online":
            best_params['learning_decay'] = learning_decay_choices[best['learning_decay']]
        self.lda_model = LatentDirichletAllocation(**best_params, max_iter=self.MAX_ITERATIONS, n_jobs=self.N_JOBS,
                                                   random_state=self.RANDOM_STATE, learning_method=self.LEARNING_METHOD).fit(self.training_data)
        self.coherence_score = -min(trials.losses())
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
            **best_trial.params, max_iter=self.MAX_ITERATIONS, n_jobs=self.N_JOBS, random_state=self.RANDOM_STATE, learning_method=self.LEARNING_METHOD).fit(self.training_data)
        self.coherence_score = best_trial.value
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
            if sorted_words in topic_word_map.values():
                continue
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
        self.corpus_dict["title"] = self.corpus_dict["title"][:len(
            self.corpus_dict["abstract"])]
        # lda_output = self.get_model().transform(self.training_data)
        # r, c = np.where(lda_output >= 0.5)
        # assigned_topics = np.split(c,np.searchsorted(r,range(1, lda_output.shape[0])))
        assigned_topics = np.argmax(
            self.get_model().transform(self.training_data), axis=1)
        df = pd.DataFrame.from_dict({"title": self.corpus_dict["title"], "documents": self.corpus_dict["abstract"], "source": self.corpus_dict["source"],
                                    "assigned_topic": assigned_topics, "timestamp": self.corpus_dict["published_date"]})
        df["timestamp"] = df["timestamp"].apply(lambda x: dateparser.parse(
            x, ignoretz=True, fuzzy=True) if not isinstance(x, datetime) else x)
        df["timestamp"] = df["timestamp"].apply(
            lambda x: x.replace(tzinfo=None))
        df["published_date"] = df["timestamp"].apply(
            lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
        if save_to_csv:
            df.to_csv("model_output.csv")
        return df

    def get_coherence_score(self):
        return self.coherence_score


class LDALabeler:
    def __init__(self, verbose=False):
        self.keybert_model = KeyBERT(
            model="paraphrase-multilingual-MiniLM-L12-v2")
        self.verbose = verbose
        # Initialize ConceptNet database
        # conceptnet_lite.connect(
        #     "/Users/timothy.dillan/Downloads/conceptnet.db")
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
            # TODO: detect language, use indo model if joined text is indo
            # also use custom pos tagger for keyphrasevectorizer if text is indo using stanfordnlp. read keyphrasevectorizer doc
            candidate_words = " ".join([i[0] for i in keyphrases_and_weights])
            textVectorization = TextVectorization()
            keybert_topic_keypharses[topic_index] = self.keybert_model.extract_keywords(
                candidate_words, top_n=10, vectorizer=KeyphraseCountVectorizer(custom_pos_tagger=textVectorization.custom_pos_tagger, stop_words=list(Preprocessor.STOPWORDS), pos_pattern=TextVectorization.KEYPHRASE_POS_PATTERN))
        if self.verbose:
            print("Extracted keyphrases:", self.topic_keyphrases)
        self.topic_keyphrases = keybert_topic_keypharses

    # def filter_topic_keyphrases_by_conceptnet(self):
    #     self.__validate_topic_keyphrases()
    #     filtered_keyphrases = {}
    #     # For each keyphrase that we'd like to verify (whether its a concept or not)
    #     for topic_index, keyphrases_and_weights in self.topic_keyphrases.items():
    #         filtered_keyphrases[topic_index] = []
    #         for keyphrase_and_weight in keyphrases_and_weights:
    #             keyphrase = keyphrase_and_weight[0]
    #             weight = keyphrase_and_weight[1]
    #             # Reformat the keyphrase (if it has more than one word) into ConceptNet acceptable format.
    #             if len(keyphrase.split()) > 1:
    #                 new_keyphrase = "_".join(keyphrase.split())
    #             # Try to check the concept from the keyphrase.
    #             try:
    #                 # If it does exist, add the keyphrase.
    #                 concepts = Label.get(text=new_keyphrase).concepts
    #                 filtered_keyphrases[topic_index].append(
    #                     (keyphrase.capitalize(), weight))
    #             except:
    #                 # Else, if it doesn't and the keyphrase only contains one word, no concept was found.
    #                 if len(keyphrase.split()) < 2:
    #                     print("No concept found for", keyphrase)
    #                     continue

    #                 # Else, for every possible n-gram combination from 2 - len(keyphrase), check every possible n-gram combination to ConceptNet
    #                 # and repeat the previous process.
    #                 start = 2
    #                 end = len(keyphrase.split())
    #                 if len(keyphrase.split()) > 2:
    #                     end -= 1

    #                 for split in range(start, end+1):
    #                     for ngram in itertools.permutations(keyphrase.split(), split):
    #                         new_keyphrase = "_".join(ngram)
    #                         try:
    #                             concepts = Label.get(
    #                                 text=new_keyphrase).concepts
    #                             new_formatted_keyphrase = " ".join(
    #                                 ngram).capitalize()
    #                             filtered_keyphrases[topic_index].append(
    #                                 (new_formatted_keyphrase, weight))
    #                         except:
    #                             print("No concept found for", new_keyphrase)

    #         # If the filtration resulted in 0 results, then fallback to default keyphrases and weights.
    #         if len(filtered_keyphrases[topic_index]) <= 0:
    #             if self.verbose:
    #                 print(
    #                     f"No concepts were found for all the keyphrases provided in topic {topic_index}. Returning the provided keyphrases instead.")
    #             filtered_keyphrases[topic_index] = keyphrases_and_weights

    #     self.topic_keyphrases = filtered_keyphrases

    def filter_topic_keyphrases_by_wikipedia(self):
        self.__validate_topic_keyphrases()
        filtered_keyphrases = {}
        # For each keyphrase that we'd like to verify (whether its a concept or not)
        for topic_index, keyphrases_and_weights in self.topic_keyphrases.items():
            filtered_keyphrases[topic_index] = []

            for keyphrase_and_weight in keyphrases_and_weights:
                keyphrase = keyphrase_and_weight[0]
                weight = keyphrase_and_weight[1]

                lang = detect(keyphrase)
                if lang != "en":
                    filtered_keyphrases[topic_index].append(
                        (keyphrase.capitalize(), weight))
                    continue

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

    def get_filtered_keyphrase_topic_index(self, topic_index, keyphrase, weight):
        lang = detect(keyphrase)
        if lang != "en":
            return topic_index, (keyphrase.capitalize(), weight)

        wikipedia_results = wikipedia.search(
            keyphrase, results=1, suggestion=False)
        if len(wikipedia_results) > 0:
            return topic_index, (keyphrase.capitalize(), weight)

        # Else, if it doesn't and the keyphrase only contains one word, no concept was found.
        if len(keyphrase.split()) < 2:
            print("No concept found for", keyphrase)
            return

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
                    return topic_index, (keyphrase.capitalize(), weight)
                print("No concept found for", new_keyphrase)

    def filter_topic_keyphrases_by_wikipedia_concurrent_map(self):
        self.__validate_topic_keyphrases()
        filtered_keyphrases = {}

        # For each keyphrase that we'd like to verify (whether its a concept or not)
        futures = []
        with ThreadPoolExecutor() as executor:
            for topic_index, keyphrases_and_weights in self.topic_keyphrases.items():
                filtered_keyphrases[topic_index] = []
                for keyphrase_and_weight in keyphrases_and_weights:
                    keyphrase = keyphrase_and_weight[0]
                    weight = keyphrase_and_weight[1]
                    futures.append(executor.submit(
                        self.get_filtered_keyphrase_topic_index, topic_index, keyphrase, weight))

        for future in as_completed(futures):
            filtered_topic_keyphrase_and_weight = future.result()
            if filtered_topic_keyphrase_and_weight is not None:
                topic_index = filtered_topic_keyphrase_and_weight[0]
                keyphrase = filtered_topic_keyphrase_and_weight[1][0]
                weight = filtered_topic_keyphrase_and_weight[1][1]
                filtered_keyphrases[topic_index].append(
                    (keyphrase.capitalize(), weight))

        for topic_index, keyphrases_and_weights in self.topic_keyphrases.items():
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
