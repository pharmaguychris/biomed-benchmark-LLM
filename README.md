# Biomedical Foundation Model Benchmark
A Reproducible Framework for Evaluating LLM Reliability in Biomedical Text Understanding

Author: Xuncheng Zhang (Westcliff University)
Contact: x.zhang4123@westcliff.edu
Repository: https://github.com/pharmaguychris/biomed-benchmark-LLM

--------------------------------------------------------------------------------
Overview
--------------------------------------------------------------------------------

This repository provides a lightweight, fully reproducible benchmarking framework
for evaluating the performance and reliability of foundation models across three
representative biomedical NLP tasks:

1. Factual Yes/No Biomedical Question Answering (BioASQ style)
2. Evidence-based Research Reasoning (PubMedQA style)
3. Disease Named-Entity Recognition (NCBI Disease Corpus)

The goal is to help researchers test whether general-purpose or domain-specific
LLMs behave consistently and accurately in biomedical contexts, where
hallucinations and factual drift pose major risks.

This project was developed as part of an ISCB poster and is suitable for
scientific reproducibility studies, LLM safety evaluations, and biomedical NLP
research.

--------------------------------------------------------------------------------
Key Features
--------------------------------------------------------------------------------

- Runs entirely locally; no external dataset downloads required
- Poster-ready datasets are embedded directly in the script
- Supports both generative and classifier LLMs
- Metrics: Accuracy (QA) and F1 (NER)
- Tested models:
  - google/flan-t5-base
  - microsoft/BioGPT-Large
  - allenai/biomed_roberta_base
  - d4data/biomedical-ner-all
- Outputs clean, interpretable summary tables

--------------------------------------------------------------------------------
Project Structure
--------------------------------------------------------------------------------

biomed-benchmark-LLM/
│
├── poster_benchmark.py        # Main evaluation pipeline
├── README.md                  
├── requirements.txt           
│
├── data/                      # Optional folder for extended datasets
└── figures/                   # Auto-generated plots

--------------------------------------------------------------------------------
Installation
--------------------------------------------------------------------------------

1. Clone the repository:

    git clone https://github.com/pharmaguychris/biomed-benchmark-LLM.git
    cd biomed-benchmark-LLM

2. Install dependencies:

    pip install -r requirements.txt

--------------------------------------------------------------------------------
Running the Benchmark
--------------------------------------------------------------------------------

Run the main script:

    python poster_benchmark.py

Example output:

    ===== BIOASQ YES/NO QA =====
    flan-t5-base accuracy = 0.80
    biomed_roberta accuracy = 0.20
    biogpt accuracy = 0.40

    ===== PUBMEDQA REASONING =====
    flan-t5-base accuracy = 0.50
    ...

    ===== NCBI DISEASE NER =====
    biomedical-ner-all F1 = 0.753

Figures are saved automatically to the figures/ directory.

--------------------------------------------------------------------------------
Example Results
--------------------------------------------------------------------------------

BioASQ Factual Yes/No QA
- Flan-T5-Base: 0.80
- BioMed-RoBERTa: 0.20
- BioGPT-Large: 0.40

PubMedQA Reasoning
- Flan-T5-Base: 0.50
- BioMed-RoBERTa: 0.30
- BioGPT-Large: 0.40

NCBI Disease NER
- biomedical-ner-all: 0.753 F1

--------------------------------------------------------------------------------
Datasets
--------------------------------------------------------------------------------

This repository uses distilled, reproducible subsets of:
- BioASQ Yes/No
- PubMedQA
- NCBI Disease Corpus

The subsets are chosen for reliability, reproducibility, and speed.

--------------------------------------------------------------------------------
Citation
--------------------------------------------------------------------------------

If you use this framework, please cite:

Zhang X. (2025). A Reproducible Benchmarking Framework for Evaluating Foundation
Models in Biomedical Text Understanding. ISCB Asia 2025 Poster.

Please also cite original datasets:
- Tsatsaronis et al., BioASQ
- Jin et al., PubMedQA
- Dogan et al., NCBI Disease Corpus

--------------------------------------------------------------------------------
Future Work
--------------------------------------------------------------------------------

- Add PubMedBERT, SciFive, Galactica, BioMedLM
- Add hallucination detection metrics
- Add HuggingFace Evaluate integration
- Add full dataset download options
- Add Gradio-based Web UI

--------------------------------------------------------------------------------
Acknowledgements
--------------------------------------------------------------------------------

This project was developed for the ISCB Asia 2025 Conference and biomedical NLP
research at Westcliff University.
