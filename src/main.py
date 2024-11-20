from pandas import DataFrame
from asyncio import run

from core.initial_data import MenopauseData
from core.stage_1.main_topics import GeneralDiscussionAnalysis


async def main():
    get_data = GeneralDiscussionAnalysis()
    data = await get_data.topics_data_collection()
    print(data)


if __name__ == '__main__':
    run(main())
