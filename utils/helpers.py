import re
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

def parse_llm_output(text, counter):
    pattern = r"Entities: ((?:\('.*?', '.*?'\) ?)+)"
    entity_dict = {}
    matches = re.findall(pattern, text)
    for entities_raw in matches:
        entities = re.findall(r"\('(.*?)', '(.*?)'\)", entities_raw)
        entity_dict[counter] = entities
        counter += 1
    rows = []
    for sid, ents in entity_dict.items():
        for entity, label in ents:
            rows.append({"Sentence_ID": sid, "Entity": entity, "Label_LLM": label})
    df = pd.DataFrame(rows)
    allowed_labels = ['B-LOC', 'B-MISC', 'B-ORG', 'B-PER', 'I-LOC', 'I-MISC', 'I-ORG', 'I-PER', 'O']
    invalid_rows = df[~df["Label_LLM"].isin(allowed_labels)]
    df_final = df.drop(invalid_rows.index)
    return df_final

def map_spacy_labels_to_conll(labels):
    label_map = {'B-ORG': 'B-ORG', 
             'O': 'O', 
             'I-ORG': 'I-ORG', 
             'B-NORP': 'B-MISC', 
             'B-PERSON': 'B-PER', 
             'I-PERSON': 'I-PER', 
             'B-LOC': 'B-LOC', 
             'B-GPE': 'B-LOC', 
             'B-DATE': 'B-MISC',
             'I-DATE': 'I-MISC', 
             'B-CARDINAL': 'B-MISC', 
             'I-CARDINAL': 'I-MISC',
             'B-QUANTITY': 'B-MISC', 
             'I-QUANTITY': 'I-MISC', 
             'B-PERCENT': 'B-MISC', 
             'I-PERCENT': 'I-MISC', 
             'B-MONEY': 'B-MISC',
             'I-MONEY': 'I-MISC', 
             'B-LANGUAGE': 'B-MISC', 
             'B-TIME': 'B-MISC', 
             'I-TIME': 'I-MISC', 
             'B-ORDINAL': 'B-MISC',
             'B-EVENT': 'B-MISC', 
             'I-EVENT': 'I-MISC', 
             'B-PRODUCT': 'B-MISC', 
             'I-GPE': 'I-LOC', 
             'I-LOC': 'I-LOC', 
             'I-NORP': 'I-MISC', 
             'B-FAC': 'B-MISC',
             'I-FAC': 'I-MISC', 
             'B-LAW': 'B-MISC', 
             'I-LAW': 'I-MISC', 
             'B-WORK_OF_ART': 'B-MISC', 
             'I-WORK_OF_ART': 'I-MISC',
             'I-PRODUCT': 'I-MISC'}
    return labels.map(label_map).fillna('O')


def evaluate(df_true, df_pred, pred_col='Label'):
    df_merged = pd.merge(df_true, df_pred, on=['Sentence_ID', 'Entity'])
    print(df_merged)
    precision = precision_score(df_merged['Label'], df_merged[pred_col], average='weighted')
    recall = recall_score(df_merged['Label'], df_merged[pred_col], average='weighted')
    f1 = f1_score(df_merged['Label'], df_merged[pred_col], average='weighted')
    return precision, recall, f1

  
