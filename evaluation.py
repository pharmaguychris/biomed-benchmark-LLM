# evaluation.py

from seqeval.metrics import f1_score
import numpy as np

def normalize(s):
    return " ".join(str(s).lower().strip().split())

def accuracy(preds, refs):
    correct = sum(normalize(p) == normalize(r) for p, r in zip(preds, refs))
    return correct / len(refs)

def ner_f1(pred_labels, true_labels):
    return f1_score(true_labels, pred_labels)

def nli_accuracy(preds, refs):
    return accuracy(preds, refs)
