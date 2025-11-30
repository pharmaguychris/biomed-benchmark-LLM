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



def load_pubmedqa():
    """
    Loads pubmed_qa:pqa_labeled from HF hub.
    Your dataset only has a 'train' split with 1000 rows.
    Returns standardized list of dicts:
        {question, context, answer}
    """
    from datasets import load_dataset

    print("Loading PubMedQA from HF Hub...")
    raw = load_dataset("pubmed_qa", "pqa_labeled")["train"]  # only split available

    data = []
    for item in raw:
        question = item["question"]
        context_list = item["context"]
        context = context_list[0] if isinstance(context_list, list) and len(context_list) > 0 else ""
        answer = item["final_decision"].lower()

        if answer not in ["yes", "no", "maybe"]:
            answer = "maybe"

        data.append({
            "question": question,
            "context": context,
            "answer": answer
        })

    return data



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
