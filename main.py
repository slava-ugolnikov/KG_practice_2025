import logging
from models.classic_model import run_classic_ner
from models.llm_model import run_llm_ner

def setup_logger():
    logger = logging.getLogger("NER_Evaluation")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


if __name__ == "__main__":
    logger = setup_logger()

    data_dir = "dataset/"
    save_dir = "ner_results/"
    api_key = "<your-openai-api-key>"

    logger.info("Запуск классической модели")
    run_classic_ner(data_dir=data_dir, logger=logger)

    logger.info("Запуск LLM модели")
    run_llm_ner(data_dir=data_dir, save_dir=save_dir, api_key=api_key, logger=logger)

