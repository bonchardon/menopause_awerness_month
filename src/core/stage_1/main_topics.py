# todo:
#  1) main topics discussed in the online community (apply web scrapping techniques);
#  2) key themes emerging from the conversation (NER).

import gensim
from gensim import corpora

from nltk.tokenize import word_tokenize
import nltk

from loguru import logger

from src.core.preprocess_data import Preprocess

nltk.download('punkt')


class GeneralDiscussionAnalysis:

    async def lda_topics(self):
        try:
            df: list[str] = await Preprocess.fully_preprocessed_data()
            tokenized_docs = [word_tokenize(str(doc)) for doc in df]
            dictionary = corpora.Dictionary(tokenized_docs)
            corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
            lda_model = gensim.models.LdaMulticore(corpus, num_topics=3, id2word=dictionary, passes=10)
            for idx, topic in lda_model.print_topics():
                print(f"Topic {idx}: {topic}")

        except Exception as e:
            logger.error(f'Error in LDA topics: {e!r}')
