from asyncio import run

from loguru import logger

from core.stage_1.main_topics import GeneralDiscussionAnalysis
from src.core.preprocess_data import Preprocess


async def main():
    get_data = Preprocess()
    data = await get_data.fully_preprocessed_data()
    if not data:
        logger.error('Something went wrong with your data. Check it out.')
    logger.info(f'All the data is ready to be processed. The dataset is {len(data)} long. ')

    compute_lda = GeneralDiscussionAnalysis()
    lsa_output = await compute_lda.lda_topics()
    print(lsa_output)

if __name__ == '__main__':
    run(main())
