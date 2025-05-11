# Named Entity Recognition Experiments with SpaCy and Prompting

This project includes two approaches for Named Entity Recognition (NER) with a subequent graph modeling:

* **ML method** based on SpaCy pre-trained model en_core_web_sm for text annotation.
* **LLM method** bsaed on prompting with meta-llama/Llama-3-8b-chat-hf model from llama.

All experiments are performed on the **CoNLL-2003** dataset. Results (Precision, Recall and F1-score) are logged to the `results.log` file.

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

### 6. Watch the results

```bash
!cat results.log
```

--- 

## Results

After running the experiment, we can get values of classification metrics as it is depicted on the table:

| Model                        | Precision |   Recall  |    F1     |
|------------------------------|----------:|-----------|-----------|
|en_core_web_sm                |     0.9116|     0.8348|     0.8678|
|meta-llama/Llama-3-8b-chat-hf |     0.8769|     0.8926|     0.8799|

