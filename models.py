# models.py

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification
)
from config import DEVICE
from prompts import prompt_pubmedqa, prompt_mednli


class HFQA:
    def __init__(self, name):
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(name).to(DEVICE)

    @torch.no_grad()
    def predict(self, batch):
        prompts = [prompt_pubmedqa(ex) for ex in batch]
        toks = self.tok(prompts, return_tensors="pt", padding=True).to(DEVICE)
        outs = self.model.generate(**toks, max_new_tokens=16)
        decoded = self.tok.batch_decode(outs, skip_special_tokens=True)
        cleaned = [d.strip().lower() for d in decoded]
        final = []
        for x in cleaned:
            if x.startswith("y"): final.append("yes")
            elif x.startswith("n"): final.append("no")
            else: final.append("maybe")
        return final


class HFNLI:
    def __init__(self, name):
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(DEVICE)
        self.id2label = self.model.config.id2label

    @torch.no_grad()
    def predict(self, batch):
        texts = [prompt_mednli(ex) for ex in batch]
        tok = self.tok(texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        logits = self.model(**tok).logits
        preds = logits.argmax(-1).cpu().tolist()
        return [self.id2label[p] for p in preds]


class HFNER:
    def __init__(self, name):
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForTokenClassification.from_pretrained(name).to(DEVICE)
        self.id2label = self.model.config.id2label

    @torch.no_grad()
    def predict(self, tokens_batch):
        preds_all = []
        for tokens in tokens_batch:
            tok = self.tok(tokens, is_split_into_words=True, return_tensors="pt", padding=True).to(DEVICE)
            logits = self.model(**tok).logits[0]
            pred_ids = logits.argmax(-1).cpu().tolist()
            labels = [self.id2label[i] for i in pred_ids][:len(tokens)]
            preds_all.append(labels)
        return preds_all
