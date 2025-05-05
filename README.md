# Named Entity Recognition Experiments

This project includes two approaches for Named Entity Recognition (NER) with a subequent graph modeling:

* **ML method** based on SpaCy pre-trained model.
* **LLM method** bsaed on prompting.

All experiments are performed on the **CoNLL-2003** dataset. Results are logged to the `results.log` file.

---

## Dataset

We use the [CoNLL-2003 NER dataset](https://www.kaggle.com/datasets/alaakhaled/conll003-englishversion/data).

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the main script

```bash
python main.py
```

Results will be saved in the `results.log` file.
