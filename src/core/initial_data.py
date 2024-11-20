import pandas as pd
from pandas import DataFrame

from loguru import logger


class MenopauseData:

    def __init__(self):
        self.df: DataFrame = pd.read_csv('core/data/meltwater_data_menopause_awareness_month.csv')

    @staticmethod
    def open_source():
        try:
            df: DataFrame = pd.read_csv('src/core/data/meltwater_data_menopause_awareness_month.csv')
            return df
        except (ValueError, SyntaxWarning) as e:
            logger.error(f'Seems like something went wrong. Here is the issue: {e!r}')
