import pandas as pd


def load_sentences(filepath):
    final, sentence = [], []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip() == '' or line.startswith("-DOCSTART"):
                if sentence:
                    final.append(sentence)
                    sentence = []
            else:
                parts = line.split()
                sentence.append((parts[0], parts[-1]))
    return final

def load_all_conll_sets():
    train = load_sentences("dataset/train.txt")
    test = load_sentences("dataset/test.txt")
    valid = load_sentences("dataset/valid.txt")

    rows = []
    for i, sentence in enumerate(train):
        for word, label in sentence:
            rows.append({"Sentence_ID": i+1, "Entity": word, "Label": label})
    df = pd.DataFrame(rows)
    return train, test, valid, df
