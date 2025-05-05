def load_sentences(filepath):
    final, sentence = [], []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip() == '' or line.startswith('-DOCSTART-'):
                if sentence:
                    final.append(sentence)
                    sentence = []
            else:
                word, label = line.split()[0], line.split()[-1]
                sentence.append((word, label.strip()))
    if sentence:
        final.append(sentence)
    return final
