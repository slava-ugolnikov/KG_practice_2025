import logging
import spacy
import pandas as pd
from spacy.training import offsets_to_biluo_tags, biluo_to_iob
from utils.helpers import evaluate, map_spacy_labels_to_conll

def run_ml_ner(conll_data, df_conll):
    nlp = spacy.load("en_core_web_sm")
    texts = [" ".join([w for w, _ in s]) for s in conll_data[:100]]

    all_tokens = []
    all_bio_tags = []
    all_sentence_ids = []

    for i, text in enumerate(texts):
        doc = nlp(text)
        entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        biluo_tags = offsets_to_biluo_tags(doc, entities)
        bio_tags = biluo_to_iob(biluo_tags)
        all_tokens.extend([token.text for token in doc])
        all_bio_tags.extend(bio_tags)
        all_sentence_ids.extend([i] * len(doc))

    df_ml = pd.DataFrame({"Sentence_ID": all_sentence_ids, "Entity": all_tokens, "Label_ML": all_bio_tags})
    df_ml['Label_ML'] = map_spacy_labels_to_conll(df_ml['Label_ML'])
    precision, recall, f1 = evaluate(df_conll, df_ml, pred_col='Label_ML')
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    logging.info(f"ML Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
