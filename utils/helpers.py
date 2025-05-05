import re
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

def parse_llm_output(text):
    pattern = r"Entities: ((?:\('.*?', '.*?'\) ?)+)"
    entity_dict = {}
    counter = 1
    matches = re.findall(pattern, text)
    for entities_raw in matches:
        entities = re.findall(r"\('(.*?)', '(.*?)'\)", entities_raw)
        entity_dict[counter] = entities
        counter += 1
    rows = []
    for sid, ents in entity_dict.items():
        for entity, label in ents:
            rows.append({"Sentence_ID": sid, "Entity": entity, "Label": label})
    return pd.DataFrame(rows)

def evaluate(df_true, df_pred, pred_col='Label'):
    df_merged = pd.merge(df_true, df_pred, on=['Sentence_ID', 'Entity'])
    precision = precision_score(df_merged['Label'], df_merged[pred_col], average='weighted')
    recall = recall_score(df_merged['Label'], df_merged[pred_col], average='weighted')
    f1 = f1_score(df_merged['Label'], df_merged[pred_col], average='weighted')
    return precision, recall, f1

  
