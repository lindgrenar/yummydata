import json
import logging
import re
import sqlite3
from datetime import datetime
from typing import List, Optional

import nltk.classify
import numpy as np
import requests
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer


class JobScraper:
    """
    A class for scraping job data from the Platsbanken API and storing it in a SQLite database.

    Attributes:
        api_root (str): The root URL for the Platsbanken API.
        text (str): The text content of the job posting.
        data (dict): The JSON data for a job posting.
        query (str): The search query for job postings.
        harvested_ids (list): A list of job IDs that have been harvested from the API.

    Author: https://github.com/lindgrenar/
    Repository: https://github.com/lindgrenar/yummydata
    """

    def __init__(self, query):
        """
        Initializes a new instance of the JobScraper class.

        Args:
            query (str): The search query for job postings.
        """
        self.api_root = "https://platsbanken-api.arbetsformedlingen.se/jobs/v1/"
        self.text = None
        self.data = None
        self.query = query
        self.harvested_ids = []
        self.create_database()

    def jobid_harvester(self):
        """
        Harvests job IDs from the Platsbanken API based on a search query.
        """
        url = self.api_root + "search/"
        headers = {'Content-Type': 'application/json'}

        for source in ["pb"]:
            data = dict(filters=[{"type": "freetext", "value": self.query}], fromDate=None, order="relevance",
                        maxRecords=25, startIndex=0, toDate=datetime.utcnow().isoformat() + 'Z',
                        source=source)

            while True:
                response = requests.post(url, headers=headers, data=json.dumps(data))
                ads = response.json()['ads']
                if not ads:
                    break
                for ad in ads:
                    self.harvested_ids.append(ad['id'])
                    print(f"Found:  {ad['id']}")
                data['startIndex'] += 25

    @staticmethod
    def create_database():
        conn = sqlite3.connect('dataset.db')
        cursor = conn.cursor()
        cursor.execute(
            '''CREATE TABLE IF NOT EXISTS jobids (ID INTEGER PRIMARY KEY, rawtext TEXT, company TEXT, title TEXT, preprocessed TEXT)''')
        cursor.close()

    def database_commit(self):
        """
        Commits the harvested job IDs to a SQLite database.
        """
        try:
            conn = sqlite3.connect('dataset.db')
            c = conn.cursor()
            # Create a table if it doesn't exist
            c.execute(
                '''CREATE TABLE IF NOT EXISTS jobids (ID INTEGER PRIMARY KEY, rawtext TEXT, company TEXT, title TEXT, preprocessed TEXT)''')

            for jobid in self.harvested_ids:
                # Insert ID into the table if it doesn't already exist
                try:
                    c.execute("INSERT OR IGNORE INTO jobids (ID) VALUES (?)", (jobid,))
                    conn.commit()
                except sqlite3.Error as e:
                    print(f"Error: Failed to execute SQL statement: {e}")

        except sqlite3.Error as e:
            print(f"Error: Failed to connect to database or execute SQL statement: {e}")

        finally:
            if conn:
                conn.close()

    def fetch_data_platsbanken(self):
        """
        Fetches job data from the Platsbanken API and stores it in a SQLite database.
        """
        try:
            conn = sqlite3.connect('dataset.db')
            c = conn.cursor()
            # Create a table if it doesn't exist
            c.execute(
                '''CREATE TABLE IF NOT EXISTS jobids (ID INTEGER PRIMARY KEY, rawtext TEXT, company TEXT, title TEXT, preprocessed TEXT)''')
            # Select IDs from the table where rawtext is NULL
            c.execute("SELECT ID FROM jobids WHERE rawtext IS NULL")
            ids_to_process = c.fetchall()

            total_ids = len(ids_to_process)

            for index, id_to_process in enumerate(ids_to_process):
                jobid = id_to_process[0]
                url = self.api_root + "job/" + str(jobid)
                headers = {'Accept': 'application/json'}
                response = requests.get(url, headers=headers)

                if not response.ok:
                    print(f"Error: Request failed with status code {response.status_code}")
                    continue

                print(f"Processing ID {index + 1} of {total_ids}. Response status code: {response.status_code}")

                try:
                    data = response.json()
                except json.JSONDecodeError:
                    print("Error: Failed to decode JSON response")
                    continue

                title = data.get('title')
                company = data['company']['name']
                description = data.get('description')
                soup = BeautifulSoup(description, "html.parser")
                text = soup.get_text()
                # Update the table with the fetched data
                try:
                    c.execute("UPDATE jobids SET rawtext=?, company=?, title=? WHERE ID=?",
                              (text, company, title, jobid))
                    conn.commit()
                except sqlite3.Error as e:
                    print(f"Error: Failed to execute SQL statement: {e}")

        except sqlite3.Error as e:
            print(f"Error: Failed to connect to database or execute SQL statement: {e}")
            conn = None
        finally:
            if conn:
                conn.close()

    def run(self):
        """
        Run
        """
        self.jobid_harvester()
        self.database_commit()
        self.fetch_data_platsbanken()


class Preprocessor:
    """
    A class for preprocessing text data using various techniques such as removing symbols, tokenizing, removing stop words,
    and lemmatizing.
    """

    def __init__(self):
        """
        Initialize a new instance of the Preprocessor class.
        """
        self.stop_words_lang = None
        self.stop_words = None
        self.lemmatizer = None
        self.stemmer = None

    def load_stop_words(self, lang: str):
        """
        Load stop words l NLTK's corpus for the given language.

        :param lang: A string specifying the language for which stop words should be loaded.
        """
        if lang == 'en':
            self.stop_words = stopwords.words('english')
        elif lang == 'sv':
            self.stop_words = stopwords.words('swedish')
        else:
            raise ValueError(f"Unsupported language: {lang}")

    @staticmethod
    def detect_language(text: str) -> Optional[str]:
        """
        Detect the language of the input text using NLTK's TextCat.

        :param text: A string containing the text to detect the language of.
        :return: A string representing the detected language or None if the language could not be detected.
        """
        tc = nltk.classify.textcat.TextCat()
        lang_code = tc.guess_language(text)
        if lang_code == 'unknown':
            return None
        try:
            if lang_code == "eng":
                lang_name = "en"
            elif lang_code == "swe":
                lang_name = "sv"
            else:
                lang_name = None
        except ValueError:
            lang_name = None
        return lang_name

    def preprocess(self, text: str, lang: str) -> str:
        """
        Preprocess text by removing non-alphanumeric characters, HTML tags, stop words, and lemmatizing the remaining words.

        :param text: A string containing the text to preprocess.
        :param lang: A string specifying the language of the text.
        :return: A string representing the preprocessed text.
        """
        text = self.remove_symbols(text)
        tokens = self.tokenize(text, lang)
        tokens = self.remove_stop_words(tokens, lang)
        lemmatized_tokens = self.lemmatize(tokens, lang)
        return ' '.join(lemmatized_tokens).encode('utf-8').decode('utf-8')

    @staticmethod
    def remove_symbols(text: str) -> str:
        """
        Remove non-alphanumeric characters from text and replace with a space.

        Args:
            text (str): A string containing the text to remove symbols from.

        Returns:
            str: A string representing the text with symbols removed.
        """
        print('Input text:', text, type(text))
        if text is None:
            return ''
        # Remove non-alphanumeric characters, except non-standard Scandinavian characters
        text = re.sub(r"[^A-Za-zÅåÄäÖöÆæØøÞþÐðßĐđŁłŊŋŠšŽžČčĆćŚśÝýŤťŘřÁáÉéÍíÓóÚúŃńŇňĹĺŔŕ]", " ", text)
        # Remove line breaks, emojis, and similar symbols
        text = re.sub(r"[\n\r\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]",
                      " ", text)
        # Convert text to lowercase
        text = text.lower()
        # Replace multiple spaces with a single space
        text = re.sub(r"\s+", " ", text)
        # Strip leading and trailing spaces
        text = text.strip()
        print('Output text:', text, type(text))

        return text

    @staticmethod
    def tokenize(text: str, lang: str) -> List[str]:
        """
        Tokenize text using NLTK's word_tokenize for the given language.

        :param text: A string containing the text to tokenize.
        :param lang: A string specifying the language of the text.
        :return: A list of strings representing the tokens.
        """
        if lang == 'en':
            tokens = word_tokenize(text, language='english')
        elif lang == 'sv':
            tokens = word_tokenize(text, language='swedish')
        else:
            raise ValueError(f"Unsupported language: {lang}")
        return tokens

    def remove_stop_words(self, tokens: List[str], lang: str) -> List[str]:
        """
        Remove stop words from tokens for the given language.

        :param tokens: A list of strings representing the tokens to remove stop words from.
        :param lang: A string specifying the language of the tokens.
        :return: A list of strings representing the tokens with stop words removed.
        """
        if self.stop_words is None or self.stop_words_lang != lang:
            self.load_stop_words(lang)
        self.stop_words_lang = lang
        filtered_tokens = [token for token in tokens if token.lower() not in self.stop_words]
        return filtered_tokens

    def lemmatize(self, tokens: List[str], lang: str) -> List[str]:
        """
        Lemmatize tokens using NLTK's WordNetLemmatizer or SnowballStemmer for the given language.

        :param tokens: A list of strings representing the tokens to lemmatize.
        :param lang: A string specifying the language of the tokens.
        :return: A list of strings representing the lemmas.
        """
        if lang == 'en':
            if self.lemmatizer is None:
                self.lemmatizer = WordNetLemmatizer()
            lemmas = [self.lemmatizer.lemmatize(token, pos='v') for token in tokens]
        elif lang == 'sv':
            if self.stemmer is None:
                self.stemmer = SnowballStemmer('swedish')
            lemmas = [self.stemmer.stem(token) for token in tokens]
        else:
            raise ValueError(f"Unsupported language: {lang}")
        return lemmas

    def preprocess_and_save(self):
        """
        Load raw text data from SQLite database, preprocess it, and save the preprocessed data back to the database.
        """
        conn = sqlite3.connect('dataset.db')
        c = conn.cursor()
        c.execute('SELECT rowid, rawtext FROM jobids WHERE preprocessed IS NULL ')
        rows = c.fetchall()
        number_rows = len(rows)
        counter = 0
        for row in rows:
            counter += 1
            job_id = row[0]
            raw_text = row[1]
            lang = self.detect_language(raw_text)
            if lang is None:
                c.execute('UPDATE jobids SET preprocessed=? WHERE rowid=?', ("Fel språk", job_id))
                break
            preprocessed_text = self.preprocess(raw_text, lang)
            c.execute('UPDATE jobids SET preprocessed=? WHERE rowid=?', (preprocessed_text, job_id))
            print(f"Preprocessed job ad: {counter}. out of {number_rows}. Lang: {lang}")
        conn.commit()
        conn.close()
        print("Preprocessing finished")


class TFIDFVectorizer:
    """
    A class for vectorizing preprocessed text data using the TF-IDF algorithm and storing the vectorized data in a SQLite database.

    Attributes:
        db_name (str): The name of the SQLite database.
        table_name (str): The name of the table containing the preprocessed text data in the SQLite database.
        text_column (str): The name of the column containing the preprocessed text data in the SQLite database.
        vectorized_column (str): The name of the column to store the vectorized data in the SQLite database.
        vectorizer (TfidfVectorizer): The TF-IDF vectorizer to use for vectorizing the preprocessed text data.
        logger (Logger): The logger to use for logging error messages.
    """

    def __init__(self):
        self.db_name = 'dataset.db'
        self.table_name = 'jobids'
        self.text_column = 'preprocessed'
        self.vectorized_column = 'vectorized'
        self.vectorizer = TfidfVectorizer()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.handler = logging.StreamHandler()
        self.handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(self.handler)

    def load_data(self):
        """
        Load preprocessed text data from the SQLite database.

        Returns:
            ids (list): A list of the ids of the preprocessed text data.
            preprocessed_corpus (list): A list of the preprocessed text data.
        """
        try:
            conn = sqlite3.connect(self.db_name)
            c = conn.cursor()
            c.execute(f'SELECT id, {self.text_column} FROM {self.table_name}')
            data = c.fetchall()
            conn.close()
        except Exception as e:
            self.logger.error(f'Error loading data from database: {e}')
            return None, None

        ids, preprocessed_corpus = zip(*data)

        return ids, preprocessed_corpus

    def save_vectorized_data(self, ids, vectorized_data_s2):
        """
        Save vectorized data to the SQLite database.

        Args:
            ids (list): A list of the ids of the preprocessed text data.
            vectorized_data_s2 (numpy.ndarray): An array of the vectorized data.
        """
        try:
            conn = sqlite3.connect(self.db_name)
            c = conn.cursor()

            # Check if vectorized column already exists
            c.execute(f"PRAGMA table_info({self.table_name})")
            columns = c.fetchall()
            if self.vectorized_column not in [column[1] for column in columns]:
                c.execute(f"ALTER TABLE {self.table_name} ADD COLUMN {self.vectorized_column} BLOB")

            # Update vectorized data in database
            for i in range(len(ids)):
                c.execute(f'UPDATE {self.table_name} SET {self.vectorized_column} = ? WHERE id = ?',
                          (vectorized_data_s2[i], ids[i]))

            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f'Error saving vectorized data to database: {e}')

    def run(self):
        """
        Fit and transform the preprocessed text data using the TF-IDF algorithm and save the vectorized data to the SQLite database.

        Returns:
            feature_names_s1 (list): A list of the feature names for the vectorized data.
            vectorized_data_s1 (numpy.ndarray): An array of the vectorized data.
        """
        self.logger.info('Loading preprocessed text data from database...')
        ids, preprocessed_corpus = self.load_data()
        if ids is None or preprocessed_corpus is None:
            self.logger.error('Failed to load preprocessed text data from database')
            return None, None

        self.logger.info('Fitting and transforming preprocessed text data using TF-IDF...')
        vectorized_data_s1 = self.vectorizer.fit_transform(preprocessed_corpus).toarray()
        feature_names_s1 = self.vectorizer.get_feature_names_out()

        self.logger.info('Saving vectorized data to database...')
        self.save_vectorized_data(ids, vectorized_data_s1)

        return feature_names_s1, vectorized_data_s1


class LDATopicModel:
    """
    A class for performing topic modeling on preprocessed and vectorized text data using the LDA algorithm.

    Attributes:
        db_name (str): The name of the SQLite database.
        table_name (str): The name of the table containing the vectorized text data in the SQLite database.
        id_column (str): The name of the column containing the document IDs in the SQLite database.
        vectorized_column (str): The name of the column containing the vectorized text data in the SQLite database.
        lda_model (LatentDirichletAllocation): The LDA model to use for topic modeling.
        num_topics (int): The number of topics to extract from the text data.
        logger (Logger): The logger to use for logging error messages.
    """

    def __init__(self, num_topics=10):
        self.db_name = 'dataset.db'
        self.table_name = 'jobids'
        self.id_column = 'id'
        self.vectorized_column = 'vectorized'
        self.lda_topics_column = 'lda_topics'
        self.lda_model = None
        self.num_topics = num_topics
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.handler = logging.StreamHandler()
        self.handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(self.handler)

    def load_data(self):
        """
        Load vectorized text data from the SQLite database.

        Returns:
            ids (list): A list of the document IDs.
            vectorized_data_s1 (numpy.ndarray): An array of the vectorized text data.
        """
        try:
            conn = sqlite3.connect(self.db_name)
            c = conn.cursor()
            c.execute(f'SELECT {self.id_column}, {self.vectorized_column} FROM {self.table_name}')
            data = c.fetchall()
            ids = [row[0] for row in data]
            vectorized_data_s1 = np.frombuffer(data[0][1])
            conn.close()
            vectorized_data_s1 = vectorized_data.reshape(-1, 1)  # Reshape to 2D
        except Exception as e:
            self.logger.error(f'Error loading vectorized data from database: {e}')
            return None, None

        return ids, vectorized_data_s1

    def fit(self, random_state=42):
        """
        Fit an LDA model to the vectorized text data.

        Args:
            random_state (int): The random state to use for reproducibility.

        Returns:
            feature_names_s4 (list): A list of the feature names for the vectorized data.
        """
        self.logger.info('Loading vectorized data from database...')
        ids, vectorized_data_s1 = self.load_data()
        if vectorized_data_s1 is None:
            self.logger.error('Failed to load vectorized data from database')
            return None

        self.logger.info('Fitting LDA model to vectorized data...')
        self.lda_model = LatentDirichletAllocation(n_components=self.num_topics, random_state=random_state)
        self.lda_model.fit(vectorized_data_s1)

        feature_names_s4 = self.lda_model.get_feature_names_out()

        self.logger.info(f'LDA model fitted with {self.num_topics} topics')

        return feature_names_s4

    def save_topics(self, ids, vectorized_data_s2):
        """
        Save the topic assignments for each document to a new column in the SQLite database.

        Args:
            ids (list): A list of the document IDs.
            vectorized_data_s2 (numpy.ndarray): An array of the vectorized text data.
        """
        if self.lda_model is None:
            self.logger.error('LDA model not fitted')
            return

        self.logger.info('Saving topic assignments to database...')

        # Get the topic assignments for each document
        lda_topics = self.lda_model.transform(vectorized_data_s2)

        try:
            conn = sqlite3.connect(self.db_name)
            c = conn.cursor()
            # Create a new column for the LDA topics if it doesn't exist
            c.execute(f"PRAGMA table_info({self.table_name})")
            columns = [col[1] for col in c.fetchall()]
            if self.lda_topics_column not in columns:
                c.execute(f"ALTER TABLE {self.table_name} ADD COLUMN {self.lda_topics_column} BLOB")
            # Save the topic assignments for each document to the new column
            for i, doc_id in enumerate(ids):
                c.execute(f"UPDATE {self.table_name} SET {self.lda_topics_column} = ? WHERE {self.id_column} = ?",
                          (lda_topics[i].tobytes(), doc_id))
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f'Error saving topic assignments to database: {e}')

    def print_topics(self, num_words=10):
        """
        Print the top words for each topic in the LDA model.

        Args:
            num_words (int): The number of top words to print for each topic.
        """
        if self.lda_model is None:
            self.logger.error('LDA model not fitted')
            return

        feature_names_s2 = self.lda_model.get_feature_names_out()
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_n_words = [feature_names_s2[i] for i in topic.argsort()[:-num_words - 1:-1]]
            print(f'Topic {topic_idx}: {" ".join(top_n_words)}')

    def run(self, random_state=42):
        """
        Load the vectorized text data from the SQLite database, fit an LDA model to the data, and save the topic
        assignments to a new column in the database.

        Args:
            random_state (int): The random state to use for reproducibility.
        """
        feature_names_s3 = self.fit(random_state=random_state)
        if feature_names_s3 is not None:
            self.print_topics()


if __name__ == '__main__':
    scraper = JobScraper("DevOps")
    preprocessor = Preprocessor()
    scraper.run()
    preprocessor.preprocess_and_save()
    tfidf_vectorizer = TFIDFVectorizer()
    vectorizer = TFIDFVectorizer()
    feature_names, vectorized_data = vectorizer.run()
    if feature_names is not None and vectorized_data is not None:
        print(feature_names)
        print(vectorized_data)
    # Create an instance of the LDATopicModel class
    lda_model = LDATopicModel()
    # Run the topic modeling pipeline
    lda_model.run(random_state=42)
