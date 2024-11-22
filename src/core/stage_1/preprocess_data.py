from string import punctuation
from re import sub
from csv import DictReader

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize

import numpy as np
import pandas as pd
from pandas import DataFrame

from loguru import logger

from src.core.consts import DATA_SOURCE, KEY_PHRASES

np.random.seed(42)


class Preprocess:
    def __init__(self):
        self.df: DataFrame = pd.read_csv(DATA_SOURCE)

    @staticmethod
    def lemmas(text_data: str) -> list[str]:
        try:
            word_lemmas = WordNetLemmatizer()
            words = word_tokenize(text_data)
            return [word_lemmas.lemmatize(word) for word in words]
        except ValueError as e:
            logger.error(f'Error in lemmatizing: {e!r}')

    @staticmethod
    def get_rid_of_punctuation_and_nan() -> list[str]:
        try:
            try_list = []
            with open(DATA_SOURCE, 'r') as file:
                csv_reader = DictReader(file)
                for row in csv_reader:
                    text_data = row[KEY_PHRASES]
                    text_data = str(text_data)
                    text_data = text_data.replace('nan', '').strip()
                    cleaned_text = sub(r'[^\w\s]', ' ', text_data)
                    sentences = cleaned_text.splitlines()
                    sentences = [sentence.strip() for sentence in sentences if sentence.strip() != '']
                    try_list.extend(sentences)
            return try_list
        except ValueError as e:
            logger.error(f'Error in removing punctuation: {e!r}')

    @staticmethod
    async def stop_words_removal(word_list: list[str]) -> list[str]:
        stop_words: set[str] = set(stopwords.words('english'))
        return [word for word in word_list if word not in stop_words]

    @classmethod
    async def fully_preprocessed_data(cls) -> list[str]:
        try:
            df: DataFrame = pd.read_csv(DATA_SOURCE)['Key Phrases'].tolist()
            # cleaned_texts = [cls.get_rid_of_punctuation_and_nan() for text in df]
            cleaned_texts = cls.get_rid_of_punctuation_and_nan()
            lemmatized_texts = [cls.lemmas(text) for text in cleaned_texts]
            stopwords_removed = [await cls.stop_words_removal(text) for text in lemmatized_texts]
            return [' '.join(text) for text in stopwords_removed]
        except (ValueError, IndexError) as e:
            logger.error(f'Error in preprocessing data: {e!r}')
