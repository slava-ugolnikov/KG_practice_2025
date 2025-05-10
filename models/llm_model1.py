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

    output = response["choices"][0]["message"]["content"]

    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        parsed = output

    filename = os.path.join(save_dir, f"batch_{batch_id}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)

    return parsed


def run_llm_ner(conll_data, df_conll, batch_size=20):

      for i in range(0, len(sentences[:20]), batch_size):
          batch = sentences[i:i+batch_size]
          ner_llama(batch, save_dir="ner_results", batch_id=i//batch_size)

      folder_path = '/content/ner_results'

      json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

      
      texts = []      
      for file_name in json_files:
          file_path = os.path.join(folder_path, file_name)
          with open(file_path, 'r', encoding='utf-8') as f:
              data = json.load(f)
              text = json.dumps(data, ensure_ascii=False)
              texts.append(text)
      pattern = r"Entities: ((?:\('.*?', '.*?'\) ?)+)"

      entity_dict = {}
      counter = 1
      
      for text in texts:
          matches = re.findall(pattern, text)
          for entities_raw in matches:
              entities = re.findall(r"\('(.*?)', '(.*?)'\)", entities_raw)
              named_entities = [(word, label) for word, label in entities]
              entity_dict[counter] = named_entities
              counter += 1
              
      allowed_labels = ['B-LOC', 'B-MISC', 'B-ORG', 'B-PER', 'I-LOC', 'I-MISC', 'I-ORG', 'I-PER', 'O']
      invalid_rows = df[~df["Label"].isin(allowed_labels)]
      df = df.drop(invalid_rows.index)

      df_LLM = df
      df_LLM = df_LLM.rename(columns={'Label': 'Label_LLM', 'Sentence_id': 'Sentence_ID'})
      print(df_llm.Label_LLM.unique())
      merged_df = pd.merge(df_conll, df_LLM, on=['Sentence_ID'], how='inner')

      precision = precision_score(merged_df['Label'], merged_df['Label_LLM'], average='weighted')
      recall = recall_score(merged_df['Label'], merged_df['Label_LLM'], average='weighted')
      f1 = f1_score(merged_df['Label'], merged_df['Label_LLM'], average='weighted')
      
      print(f"Precision: {precision:.4f}")
      print(f"Recall: {recall:.4f}")
      print(f"F1: {f1:.4f}")
