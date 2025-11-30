import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from datasets import load_dataset
from collections import Counter

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------
# MODELS
# ----------------------------------------------------
def load_model(name):
    print(f"\nLoading model: {name}")
    try:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSeq2SeqLM.from_pretrained(name).to(DEVICE)
        is_seq2seq = True
    except:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name).to(DEVICE)
        is_seq2seq = False
    return tok, model, is_seq2seq


def run_gen(tok, model, prompt, max_new_tokens=10):
    enc = tok(prompt, return_tensors="pt", truncation=True).to(DEVICE)
    out = model.generate(**enc, max_new_tokens=max_new_tokens)
    return tok.decode(out[0], skip_special_tokens=True)


# ----------------------------------------------------
# PUBMEDQA DEBUG
# ----------------------------------------------------
def debug_pubmedqa():
    print("\n===== DEBUG PUBMEDQA =====")
    ds = load_dataset("pubmed_qa", "pqa_labeled")["train"]  # 1000 examples
    print("Label counts:", Counter(ds["final_decision"]))

    # Take 5 samples
    samples = ds.select(range(5))

    models = ["google/flan-t5-base", "microsoft/BioGPT-Large"]

    for m in models:
        tok, model, _ = load_model(m)
        print(f"\n--- MODEL: {m} ---")

        for ex in samples:
            prompt = (
                "Answer with yes, no, or maybe.\n"
                f"Question: {ex['question']}\n"
                f"Context: {ex['context']}\nAnswer:"
            )
            raw = run_gen(tok, model, prompt, max_new_tokens=5).lower()

            # normalization
            if raw.startswith("yes"): pred = "yes"
            elif raw.startswith("no"): pred = "no"
            elif raw.startswith("maybe"): pred = "maybe"
            else: pred = "maybe"

            gold = ex["final_decision"]

            print("\nQUESTION:", ex["question"])
            print("GOLD:", gold)
            print("PROMPT:", prompt)
            print("RAW MODEL OUTPUT:", raw)
            print("PRED:", pred)
            print("CORRECT?" , pred == gold)


# ----------------------------------------------------
# MEDQA DEBUG
# ----------------------------------------------------
def debug_medqa():
    print("\n===== DEBUG MEDQA =====")
    ds = load_dataset(
        "bigbio/med_qa",
        "med_qa_en_4options_bigbio_qa",
        trust_remote_code=True
    )["test"]

    samples = ds.select(range(5))

    models = ["google/flan-t5-base", "microsoft/BioGPT-Large"]

    for m in models:
        tok, model, _ = load_model(m)
        print(f"\n--- MODEL: {m} ---")

        for ex in samples:
            choices = ex["choices"]
            gold_answer_text = ex["answer"][0]
            gold_idx = choices.index(gold_answer_text)
            gold_letter = "ABCD"[gold_idx]

            prompt = (
                "Choose the correct answer. Reply ONLY with A, B, C, or D.\n\n"
                f"Question: {ex['question']}\n\n"
                f"A. {choices[0]}\n"
                f"B. {choices[1]}\n"
                f"C. {choices[2]}\n"
                f"D. {choices[3]}\n\n"
                "Answer:"
            )

            raw = run_gen(tok, model, prompt, max_new_tokens=3).strip().upper()

            # normalize to ABCD
            pred = raw[0] if raw and raw[0] in "ABCD" else "A"

            print("\nQUESTION:", ex["question"])
            print("GOLD:", gold_letter)
            print("RAW OUTPUT:", raw)
            print("PRED:", pred)
            print("CORRECT?", pred == gold_letter)


# ----------------------------------------------------
# NCBI DISEASE DEBUG
# ----------------------------------------------------
def debug_ncbi():
    print("\n===== DEBUG NCBI DISEASE (LOCAL TSV) =====")

    path = r"C:\Users\chris\Downloads\Biomedical-NLP-Benchmarks-v1.0.0\BIDS-Xu-Lab-Biomedical-NLP-Benchmarks-510cad1\benchmarks\[NER]NCBI_Disease\datasets\full_set\test.tsv"

    sentences = []
    tokens, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append((tokens, labels))
                tokens, labels = [], []
            else:
                t, l = line.split("\t")
                tokens.append(t)
                labels.append(l)

    # take first 5 sentences
    samples = sentences[:5]

    tok, model, _ = load_model("d4data/biomedical-ner-all")

    for tokens, gold in samples:
        enc = tok(tokens, is_split_into_words=True, return_tensors="pt").to(DEVICE)
        logits = model(**enc).logits[0]
        pred_ids = logits.argmax(-1).cpu().tolist()
        wid = enc.word_ids()

        pred = []
        for i in range(len(tokens)):
            sub = [pred_ids[j] for j in range(len(pred_ids)) if wid[j] == i]
            if sub:
                pred.append(model.config.id2label[sub[0]])
            else:
                pred.append("O")

        print("\nTOKENS:", tokens)
        print("GOLD:", gold)
        print("PRED:", pred)


# ----------------------------------------------------
# RUN ALL DEBUG
# ----------------------------------------------------
if __name__ == "__main__":
    debug_pubmedqa()
    debug_medqa()
    debug_ncbi()


