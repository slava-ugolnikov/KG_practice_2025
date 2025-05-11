from models.llm_model import run_llm_ner
from models.ml_model import run_ml_ner
from utils import load_all_conll_sets
import argparse
import logging


logging.basicConfig(filename='results.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Run NER experiments")
    parser.add_argument("--model", choices=["llm", "ml"], required=True, help="Model to run")
    args = parser.parse_args()

    logging.info(f"Starting experiment with model: {args.model}")
    
    conll_train, conll_test, conll_valid, df_conll = load_all_conll_sets()

    if args.model == "llm":
        run_llm_ner(conll_train, df_conll)
    elif args.model == "ml":
        run_ml_ner(conll_train, df_conll)

if __name__ == "__main__":
    main()

