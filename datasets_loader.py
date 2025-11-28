# datasets_loader.py

from datasets import load_dataset
from config import MAX_EXAMPLES

def load_mednli(split="train"):
    ds = load_dataset("mednli", "matched", split=split)

    if MAX_EXAMPLES:
        ds = ds.select(range(min(MAX_EXAMPLES, len(ds))))

    label_map = ["entailment", "neutral", "contradiction"]

    examples = []
    for i, ex in enumerate(ds):
        examples.append({
            "id": f"mednli-{split}-{i}",
            "task": "mednli",
            "premise": ex["sentence1"],
            "hypothesis": ex["sentence2"],
            "label": label_map[ex["label"]],
        })

    return examples



def load_pubmedqa(split="train"):
    ds = load_dataset("pubmed_qa", "pqa_artificial", split=split)
    if MAX_EXAMPLES:
        ds = ds.select(range(min(MAX_EXAMPLES, len(ds))))

    examples = []
    for i, ex in enumerate(ds):
        examples.append({
            "id": f"pubmedqa-{split}-{i}",
            "task": "pubmedqa",
            "question": ex["question"],
            "context": ex["context"],
            "answer": ex["final_decision"]
        })
    return examples


def load_ncbi(split="train"):
    ds = load_dataset("ncbi_disease", split=split)
    if MAX_EXAMPLES:
        ds = ds.select(range(min(MAX_EXAMPLES, len(ds))))

    examples = []
    for i, ex in enumerate(ds):
        examples.append({
            "id": f"ncbi-{split}-{i}",
            "task": "ncbi_ner",
            "tokens": ex["tokens"],
            "labels": ex["ner_tags"]
        })
    return examples


def load_all():
    return {
        "mednli": load_mednli("train"),
        "pubmedqa": load_pubmedqa("train"),
        "ncbi": load_ncbi("train")
    }
