# todo:
#  1) main topics discussed in the online community (apply web scrapping techniques);
import pandas as pd
from pandas import DataFrame
from asyncio import run

from loguru import logger

from src.core.initial_data import MenopauseData
from src.core.consts import DATA_SOURCE


class GeneralDiscussionAnalysis:

    def __init__(self):
        ...

    async def topics_data_collection(self):
        try:
            df: DataFrame = pd.read_csv(DATA_SOURCE[['']])
            return df
        except (ValueError, FileNotFoundError) as e:
            logger.error(f'There are issues here. More info: {e!r}')


async def main():
    get_data = GeneralDiscussionAnalysis()
    data = await get_data.topics_data_collection()
    print(data)


if __name__ == '__main__':
    print(DATA_SOURCE)
    run(main())
