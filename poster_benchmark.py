import os
import re
import csv
from collections import Counter

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
)
from seqeval.metrics import f1_score

# ============================================================
# DEVICE & SETTINGS
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = DEVICE == "cuda"

PUBMED_LIMIT = 500
MEDQA_LIMIT = 300
NCBI_LIMIT = 500

NCBI_TSV_PATH = r"C:\Users\chris\Downloads\Biomedical-NLP-Benchmarks-v1.0.0\BIDS-Xu-Lab-Biomedical-NLP-Benchmarks-510cad1\benchmarks\[NER]NCBI_Disease\datasets\full_set\test.tsv"


# ============================================================
# HELPERS
# ============================================================
def truncate_words(text, max_tokens):
    if not text:
        return ""
    words = text.split()
    return " ".join(words[:max_tokens])


def accuracy(preds, refs):
    return sum(p == r for p, r in zip(preds, refs)) / len(refs)


def parse_yesno_maybe(output):
    s = output.lower()
    idx = {k: s.rfind(k) for k in ("yes", "no", "maybe")}
    best = max(idx, key=idx.get)
    return best if idx[best] != -1 else "maybe"


def parse_abcd(output):
    s = output.upper()

    m = re.search(r"ANSWER\s*[:\-]?\s*([ABCD])", s)
    if m:
        return m.group(1)

    for ch in reversed(s):
        if ch in "ABCD":
            return ch

    for ch in s:
        if ch in "ABCD":
            return ch

    return "A"


# ============================================================
# DATA LOADERS
# ============================================================
def load_pubmedqa():
    print("Loading PubMedQA from HF Hub...")
    raw = load_dataset("pubmed_qa", "pqa_labeled")["train"]

    out = []
    for ex in raw:
        q = ex["question"]
        ctx_dict = ex["context"]
        if isinstance(ctx_dict, dict) and "contexts" in ctx_dict:
            ctx = " ".join(ctx_dict["contexts"])
        else:
            ctx = str(ctx_dict)
        label = ex["final_decision"].lower().strip()
        if label not in {"yes", "no", "maybe"}:
            label = "maybe"
        out.append({"question": q, "context": ctx, "label": label})

    print(f"PubMedQA size: {len(out)}")
    print("Label counts:", Counter([e["label"] for e in out]))
    return out


def load_medqa():
    print("Loading MedQA-USMLE from HF Hub...")
    ds = load_dataset(
        "bigbio/med_qa",
        "med_qa_en_4options_bigbio_qa",
        trust_remote_code=True,
    )["test"]

    out = []
    for ex in ds:
        choices = ex["choices"]
        ans_text = ex["answer"][0]
        gold_idx = choices.index(ans_text) if ans_text in choices else 0
        gold_letter = "ABCD"[gold_idx]
        out.append({
            "question": ex["question"],
            "choices": choices,
            "label": gold_letter
        })

    print(f"MedQA size: {len(out)}")
    return out


def load_ncbi_local():
    print("Loading NCBI Disease from local TSV...")
    if not os.path.exists(NCBI_TSV_PATH):
        raise FileNotFoundError(NCBI_TSV_PATH)

    items = []
    tokens, labels = [], []

    with open(NCBI_TSV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "":
                if tokens:
                    items.append({"tokens": tokens, "labels": labels})
                    tokens, labels = [], []
                continue

            parts = line.split("\t")
            if len(parts) != 2:
                continue
            tok, lab = parts
            tokens.append(tok)
            labels.append(lab)

    if tokens:
        items.append({"tokens": tokens, "labels": labels})

    print(f"NCBI disease sentences: {len(items)}")
    return items


# ============================================================
# MODEL LOADING (WITH FIXED PADDING)
# ============================================================
def load_generative_model(name):
    print(f"Loading model: {name}")
    tok = AutoTokenizer.from_pretrained(name)

    try:
        # seq2seq
        model = AutoModelForSeq2SeqLM.from_pretrained(name)
        is_seq2seq = True

    except Exception:
        # causal LM (BioGPT)
        model = AutoModelForCausalLM.from_pretrained(name)
        is_seq2seq = False

        # ★★★★★ FIX PADDING FOR BioGPT ★★★★★
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

    if DEVICE == "cuda":
        model.to(DEVICE)
        if USE_FP16:
            model.half()

    model.eval()
    return tok, model, is_seq2seq


def load_ner_model(name="d4data/biomedical-ner-all"):
    print(f"Loading NER model: {name}")
    tok = AutoTokenizer.from_pretrained(name, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(name)

    if DEVICE == "cuda":
        model.to(DEVICE)
        if USE_FP16:
            model.half()

    model.eval()
    return tok, model, model.config.id2label


# ============================================================
# PUBMEDQA
# ============================================================
def eval_pubmedqa(model_name, data):
    tok, model, _ = load_generative_model(model_name)

    preds, refs = [], []
    bs = 8
    n = min(len(data), PUBMED_LIMIT)
    print(f"Evaluating {model_name} on PubMedQA (n={n}) ...")

    for start in range(0, n, bs):
        batch = data[start:start + bs]

        prompts = []
        for ex in batch:
            q = truncate_words(ex["question"], 40)
            ctx = truncate_words(ex["context"], 200)
            if ctx:
                p = (
                    "Answer with yes, no, or maybe.\n"
                    f"Question: {q}\n"
                    f"Context: {ctx}\n"
                    "Answer:"
                )
            else:
                p = (
                    "Answer with yes, no, or maybe.\n"
                    f"Question: {q}\n"
                    "Answer:"
                )
            prompts.append(p)

        enc = tok(
            prompts, return_tensors="pt",
            padding=True, truncation=True, max_length=512
        ).to(DEVICE)

        with torch.no_grad():
            if DEVICE == "cuda" and USE_FP16:
                with torch.cuda.amp.autocast():
                    out = model.generate(**enc, max_new_tokens=5)
            else:
                out = model.generate(**enc, max_new_tokens=5)

        decoded = tok.batch_decode(out, skip_special_tokens=True)

        for ex, dec in zip(batch, decoded):
            preds.append(parse_yesno_maybe(dec))
            refs.append(ex["label"])

    acc = accuracy(preds, refs)
    print(f"Accuracy: {acc:.3f}")
    return acc


# ============================================================
# MEDQA
# ============================================================
def eval_medqa(model_name, data):
    tok, model, _ = load_generative_model(model_name)

    preds, refs = [], []
    bs = 4
    n = min(len(data), MEDQA_LIMIT)
    print(f"Evaluating {model_name} on MedQA (n={n}) ...")

    for start in range(0, n, bs):
        batch = data[start:start + bs]

        prompts = []
        for ex in batch:
            q = truncate_words(ex["question"], 120)
            choices = ex["choices"]
            lines = "\n".join(f"{l}. {t}" for l, t in zip("ABCD", choices))
            prompts.append(
                "Choose the single best answer and reply with only A, B, C, or D.\n\n"
                f"Question: {q}\n\nChoices:\n{lines}\n\nAnswer:"
            )

        enc = tok(
            prompts, return_tensors="pt",
            padding=True, truncation=True, max_length=512
        ).to(DEVICE)

        with torch.no_grad():
            if DEVICE == "cuda" and USE_FP16:
                with torch.cuda.amp.autocast():
                    out = model.generate(**enc, max_new_tokens=3)
            else:
                out = model.generate(**enc, max_new_tokens=3)

        decoded = tok.batch_decode(out, skip_special_tokens=True)

        for ex, dec in zip(batch, decoded):
            preds.append(parse_abcd(dec))
            refs.append(ex["label"])

    acc = accuracy(preds, refs)
    print(f"Accuracy: {acc:.3f}")
    return acc


# ============================================================
# NCBI NER
# ============================================================
def map_ncbi_gold(l):
    return "B-Disease" if l in ("B", "I") else "O"


def map_ncbi_pred(l):
    ll = l.lower()
    return "B-Disease" if ("disease" in ll or "disorder" in ll) else "O"


def eval_ncbi(ncbi_data):
    tok, model, id2label = load_ner_model()

    preds_all, refs_all = [], []
    n = min(len(ncbi_data), NCBI_LIMIT)
    print(f"Evaluating NER on NCBI (n={n}) ...")

    for ex in ncbi_data[:n]:
        tokens = ex["tokens"]
        gold = [map_ncbi_gold(l) for l in ex["labels"]]

        enc = tok(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).to(DEVICE)

        with torch.no_grad():
            if DEVICE == "cuda" and USE_FP16:
                with torch.cuda.amp.autocast():
                    logits = model(**enc).logits
            else:
                logits = model(**enc).logits

        pred_ids = logits.argmax(-1)[0].cpu().tolist()
        word_ids = enc.word_ids(batch_index=0)

        pred_raw = []
        for i in range(len(tokens)):
            sub = [pred_ids[j] for j, wid in enumerate(word_ids) if wid == i]
            pred_raw.append(id2label[sub[0]] if sub else "O")

        pred = [map_ncbi_pred(l) for l in pred_raw]

        refs_all.append(gold)
        preds_all.append(pred)

    f1 = f1_score(refs_all, preds_all)
    print(f"NER F1 (Disease vs O): {f1:.3f}")
    return f1


# ============================================================
# MAIN
# ============================================================
def main():
    print("DEVICE:", DEVICE, "| fp16:", USE_FP16)

    pubmed = load_pubmedqa()
    medqa = load_medqa()
    ncbi = load_ncbi_local()

    print("\n===== PUBMEDQA (google/flan-t5-base) =====")
    eval_pubmedqa("google/flan-t5-base", pubmed)

    print("\n===== PUBMEDQA (microsoft/BioGPT-Large) =====")
    eval_pubmedqa("microsoft/BioGPT-Large", pubmed)

    print("\n===== MEDQA (google/flan-t5-base) =====")
    eval_medqa("google/flan-t5-base", medqa)

    print("\n===== MEDQA (microsoft/BioGPT-Large) =====")
    eval_medqa("microsoft/BioGPT-Large", medqa)

    print("\n===== NCBI DISEASE NER (d4data/biomedical-ner-all) =====")
    eval_ncbi(ncbi)


if __name__ == "__main__":
    main()
