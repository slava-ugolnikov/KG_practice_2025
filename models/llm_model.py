import logging
import os
import json
import openai
import re
import pandas as pd
from utils.helpers import parse_llm_output, evaluate
from tqdm import tqdm

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = "https://api.together.xyz/v1"

def ner_llama(sentences, sentence_ids):
    prompt = f"""You've got a task.

    Find all named entities in the following sentences using the BIO tagging format:

    - Allowed labels: "B-LOC", "B-MISC", "B-ORG", "B-PER", "I-LOC", "I-MISC", "I-ORG", "I-PER", "O".
    - Where:
      - B — beginning of an entity
      - I — inside an entity
      - O — outside any entity
    - Entity types:
      - PERS — person
      - LOC — location
      - ORG — organization
      - MISC — miscellaneous
    For example George Bush would be tagged like that: George (B-PER) Bush (I-PER) - single word per entity strictly.

    Notice that you should label not only entites, you should use 'O' label for words that are not named entities (they can be any part of speech: verbs, prepositions) and include them please in your output. 
    It is very important for you to annotate every word in a sentence and include it in an output. 
    Also it is very important that sentences that starts with different numbers are separated. To remember it, include the number of sentence in the output. 
    Distinguish between entities like Britain and British, Germany and German etc. Britain and Germany are LOC (because they are countries) while British and German are MISC (because they mean an affiliation to the country).
    
    Example of an input:
    (21, 'Rare Hendrix song draft sells for almost $ 17,000 .')
    
    Example of an output:
    Sentence 21: 'Rare Hendrix song draft sells for almost $ 17,000 .'
    Entities: ('Rare', 'O') ('Hendrix', 'B-PER') ('song', 'O') ('draft', 'O') ('sells', 'O') ('for', 'O') ('almost', 'O') ('$ 17,000', 'O') ('.', 'O')

    Sentences: "{list(zip(sentence_ids, sentences))}"

    Answer:
    """

    response = openai.ChatCompletion.create(
        model="meta-llama/Llama-3-8b-chat-hf",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=4096
    )
    return response["choices"][0]["message"]["content"]


def run_llm_ner(conll_data, df_conll, batch_size=20):
    pre_sentences = [" ".join([w for w, _ in s]) for s in conll_data[:500]]
    rows_id = [i for i in range(1, len(pre_sentences))]
    sentences = list(zip(rows_id, pre_sentences))
    all_dfs = []

    for i in tqdm(range(0, len(sentences), batch_size), desc="Processing batches"):
        batch = sentences[i:i + batch_size]
        texts = [text for id, text in batch]
        ids = [id for id, text in batch]
        try:
            raw_output = ner_llama(texts, ids)
            print(raw_output)
            df_llm_batch = parse_llm_output(raw_output, ids[0])
            print(df_llm_batch)
            all_dfs.append(df_llm_batch)
        except Exception as e:
            print(f"[!] Ошибка в батче {i // batch_size}: {e}")
            continue

    df_llm = pd.concat(all_dfs, ignore_index=True)
    precision, recall, f1 = evaluate(df_conll, df_llm, pred_col='Label_LLM')
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

