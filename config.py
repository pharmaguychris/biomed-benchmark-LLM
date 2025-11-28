# config.py

DATASET_MEDNLI = "mednli"
DATASET_PUBMEDQA = "pubmed_qa"
DATASET_NCBI = "ncbi_disease"

MAX_EXAMPLES = 300   # reduce for speed. Increase later.

HF_QA_MODELS = [
    "google/flan-t5-base",
    "microsoft/BioGPT-Large-PubMedQA"
]

HF_NER_MODELS = [
    "d4data/biomedical-ner-all",
    "alvaroalon2/biobert_diseases_ner"
]

HF_NLI_MODELS = [
    "emilyalsentzer/Bio_ClinicalBERT"  # good for MedNLI
]

DEVICE = "cuda"

