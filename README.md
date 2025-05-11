# Named Entity Recognition Experiments

This project includes two approaches for Named Entity Recognition (NER) with a subequent graph modeling:

* **ML method** based on SpaCy pre-trained model annotation.
* **LLM method** bsaed on prompting.

All experiments are performed on the **CoNLL-2003** dataset. Results are logged to the `results.log` file.

---

## Dataset

We use the [CoNLL-2003 NER dataset](https://www.kaggle.com/datasets/alaakhaled/conll003-englishversion/data).

---

## How to Run

### 1. Clone repository

```bash
git clone https://github.com/slava-ugolnikov/KG_practice_2025.git
cd KG_practice_2025
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get an API for LLM

Get an API from https://api.together.xyz/v1 to get access to LLM. It is free (almost).

```bash
import os
os.environ["OPENAI_API_KEY"] = "<your-API>"
```

### 4. Download the dataset from kaggle.com
Notice! Before running the code, download personal kaggle.json file from kaggle.com and put it into secrets/ folder

```bash
mkdir -p ~/.kaggle
cp secrets/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
kaggle datasets download alaakhaled/conll003-englishversion
mkdir -p dataset
unzip /content/KG_practice_2025/conll003-englishversion.zip -d /content/KG_practice_2025/dataset
```

### 5. Run the main script

```bash
python main.py --model llm
```
```bash
python main.py --model ml
```

### 6. Seeing the results

```bash
!cat results.log
```
