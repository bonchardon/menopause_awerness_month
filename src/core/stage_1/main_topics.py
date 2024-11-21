# todo:
#  1) main topics discussed in the online community (+ apply web scrapping techniques);
import pandas as pd
from pandas import DataFrame

import gensim
from gensim import corpora

import numpy as np

from nltk.tokenize import word_tokenize
import nltk

from loguru import logger

from torch import no_grad, device, cuda
from transformers import BertTokenizer, BertModel

from src.core.consts import DATA_SOURCE
from src.core.stage_1.preprocess_data import Preprocess

nltk.download('punkt')


class GeneralDiscussionAnalysis:

    def __init__(self):
        ...

    async def lda_topics(self):

        try:
            # df: list[str] = pd.read_csv(DATA_SOURCE)['Key Phrases'].tolist()
            df: DataFrame = await Preprocess.fully_preprocessed_data()
            df: list[str] = df['Key Phrases'].tolist()
            # todo: preprocess here
            tokenized_docs = [word_tokenize(str(doc)) for doc in df]
            dictionary = corpora.Dictionary(tokenized_docs)
            corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
            lda_model = gensim.models.LdaMulticore(corpus, num_topics=3, id2word=dictionary, passes=10)
            for idx, topic in lda_model.print_topics():
                return f"Topic {idx}: {topic}"

        except Exception as e:
            logger.error(f'Here is an error/ More details provided: {e!r}')

    async def main_topics_data_collection(self):
        # todo 1: get main topics by keywords provided in 'Key Phrases' column
        try:
            df: DataFrame = pd.read_csv(DATA_SOURCE)[['Key Phrases']]
            corpus: str = ' '.join(df['Key Phrases'].astype(str))

            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertModel.from_pretrained('bert-base-uncased')

            inputs = tokenizer(corpus, return_tensors='pt', truncation=True, padding=True, max_length=512)
            model.to(device('cuda' if cuda.is_available() else 'cpu'))
            if not model:
                logger.warning('Seems like there is an issue with model itself. Check it out.')
            with no_grad():
                outputs = model(**inputs)

            token_embeddings = outputs.last_hidden_state
            # token_embeddings = token_embeddings.squeeze(0).cpu().numpy()
            tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze(0))
            average_token_embeddings = np.mean(token_embeddings, axis=1)
            token_scores = np.linalg.norm(average_token_embeddings, axis=1)
            sorted_indices = np.argsort(token_scores)[::-1]
            sorted_tokens = [tokens[i] for i in sorted_indices]

            top_n = 10
            print("Top 10 important tokens:")
            for i in range(top_n):
                print(f"{sorted_tokens[i]} (Score: {token_scores[sorted_indices[i]]})")

        except (ValueError, FileNotFoundError) as e:
            logger.error(f'There are issues here. More info: {e!r}')
