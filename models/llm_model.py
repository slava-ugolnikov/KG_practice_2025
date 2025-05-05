import logging
import os
import json
import openai
import re
import pandas as pd
from utils.helpers import parse_llm_output, evaluate

openai.api_key = os.getenv("OPENAI_API_KEY")

def ner_llama(sentences, batch_id=0):
    prompt = f"""You've got a task.

    Find all named entities in the following sentence using the BIO tagging format:

    - Allowed labels: "B-LOC", "B-MISC", "B-ORG", "B-PER", "I-LOC", "I-MISC", "I-ORG", "I-PER", "O".
    - Where:
      - B- = beginning of an entity
      - I- = inside an entity
      - O = outside any entity
    - Entity types:
      - PERS — person
      - LOC — location
      - ORG — organization
      - MISC — miscellaneous
    For example George Bush would be tagged like that: George (B-PER) Bush (I-PER) - single word per entity strictly.

    Notice that you should label not only entites, you should use 'O' label for words that are not named entities (they can be any part of speech: verbs, prepositions) and include them please in your output.
    Also it is very important that sentences that starts with different numbers are separated. To remember it, include the number of sentence in the output.

    Example of an output:
    Sentence 1: EU rejects German call to boycott British lamb;
    Entities: ('EU', 'B-ORG') ('rejects', 'O') ('German', 'B-MISC') ('call', 'O') ('to', 'O') ('boycott', 'O') ('British', 'B-MISC') ('lamb', 'O') ('.', 'O').

    Sentence: "{sentences}"
    """

    for i, sentence in enumerate(sentences, 1):
        prompt += f"{i}. {sentence}\n"
    prompt += "\nAnswer:\n"

    response = openai.ChatCompletion.create(
        model="meta-llama/Llama-3-8b-chat-hf",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=2048
    )
    return response["choices"][0]["message"]["content"]

def run_llm_ner(conll_data, df_conll):
    sentences = [" ".join([w for w, _ in s]) for s in conll_data[:20]]
    raw_output = ner_llama(sentences)
    df_llm = parse_llm_output(raw_output)
    print(df_conll.columns)
    print(df_llm.columns)
    precision, recall, f1 = evaluate(df_conll, df_llm)
    print(precision, recall, f1)

    logging.info(f"LLM Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

