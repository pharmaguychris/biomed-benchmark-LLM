# run_benchmark.py

from datasets_loader import load_all
from models import HFQA, HFNER, HFNLI
from evaluation import accuracy, ner_f1, nli_accuracy
from config import HF_QA_MODELS, HF_NER_MODELS, HF_NLI_MODELS
from tqdm import tqdm
import pandas as pd

def batched(xs, size=8):
    batch = []
    for x in xs:
        batch.append(x)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch

def main():
    data = load_all()
    results = []

    # PubMedQA
    for m in HF_QA_MODELS:
        model = HFQA(m)
        preds, refs = [], []
        for batch in tqdm(batched(data["pubmedqa"]), desc=f"PubMedQA {m}"):
            p = model.predict(batch)
            preds.extend(p)
            refs.extend([ex["answer"] for ex in batch])
        results.append({
            "dataset": "PubMedQA",
            "model": m,
            "metric": "accuracy",
            "value": accuracy(preds, refs)
        })

    # MedNLI
    for m in HF_NLI_MODELS:
        model = HFNLI(m)
        preds, refs = [], []
        for batch in tqdm(batched(data["mednli"]), desc=f"MedNLI {m}"):
            p = model.predict(batch)
            preds.extend(p)
            refs.extend([ex["label"] for ex in batch])
        results.append({
            "dataset": "MedNLI",
            "model": m,
            "metric": "accuracy",
            "value": nli_accuracy(preds, refs)
        })

    # NCBI-Disease NER
    for m in HF_NER_MODELS:
        model = HFNER(m)
        pred_labels, true_labels = [], []
        for ex in tqdm(data["ncbi"], desc=f"NCBI {m}"):
            p = model.predict([ex["tokens"]])[0]
            pred_labels.append(p)
            true_labels.append(ex["labels"])
        results.append({
            "dataset": "NCBI-Disease",
            "model": m,
            "metric": "f1",
            "value": ner_f1(pred_labels, true_labels)
        })

    df = pd.DataFrame(results)
    df.to_csv("benchmark_results.csv", index=False)
    print(df)
    print("\nSaved: benchmark_results.csv")

if __name__ == "__main__":
    main()
