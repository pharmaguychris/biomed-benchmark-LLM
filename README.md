# Biomedical Foundation Model Benchmark
A Reproducible Framework for Evaluating LLM Reliability in Biomedical Text Understanding

Author: Xuncheng Zhang (Westcliff University)
Contact: x.zhang4123@westcliff.edu
Repository: https://github.com/pharmaguychris/biomed-benchmark-LLM

--------------------------------------------------------------------------------
Overview
--------------------------------------------------------------------------------

This repository provides a lightweight, fully reproducible evaluation framework
for testing the reliability of foundation models on three representative 
biomedical NLP tasks:

1. PubMedQA (yes/no/maybe biomedical reasoning)
2. MedQA-USMLE (multiple-choice clinical knowledge)
3. NCBI Disease Named-Entity Recognition (NER)

The benchmark enables fast, local evaluation on a single GPU and produces 
results that match trends reported in recent biomedical LLM research.
This work supports reproducibility studies, LLM safety analysis, and biomedical 
NLP education.

--------------------------------------------------------------------------------
Key Features
--------------------------------------------------------------------------------

- Runs fully locally with HuggingFace datasets + local Xu-Lab NCBI TSV
- GPU-accelerated FP16 inference supported
- Evaluates both generative and token-classification LLMs
- Metrics: Accuracy (PubMedQA, MedQA) and F1 (NCBI NER)
- Clean and interpretable console output
- Designed for academic poster work and reproducible benchmarking

Models evaluated in this version:
- google/flan-t5-base
- microsoft/BioGPT-Large
- d4data/biomedical-ner-all

--------------------------------------------------------------------------------
Project Structure
--------------------------------------------------------------------------------

biomed-benchmark-LLM/
│
├── poster_benchmark.py        # Main evaluation pipeline
├── datasets_loader.py         # Dataset loaders
├── models.py                  # Model loading utilities
├── debug.py                   # Sample inspection script
├── requirements.txt
└── README.md

--------------------------------------------------------------------------------
Installation
--------------------------------------------------------------------------------

1. Clone the repository:

    git clone https://github.com/pharmaguychris/biomed-benchmark-LLM.git
    cd biomed-benchmark-LLM

2. Install dependencies:

    pip install -r requirements.txt

3. (Optional) Verify GPU:

    nvidia-smi

--------------------------------------------------------------------------------
Running the Benchmark
--------------------------------------------------------------------------------

Run:

    python poster_benchmark.py

--------------------------------------------------------------------------------
Final Benchmark Results
--------------------------------------------------------------------------------

PubMedQA (500 samples):
- Flan-T5-Base Accuracy: 0.554
- BioGPT-Large Accuracy: 0.308

MedQA-USMLE (300 samples):
- Flan-T5-Base Accuracy: 0.247
- BioGPT-Large Accuracy: 0.283

NCBI Disease NER (500 sentences):
- biomedical-ner-all F1: 0.449

--------------------------------------------------------------------------------
Interpretation of Results
--------------------------------------------------------------------------------

1. PubMedQA:
   Flan-T5 performs significantly better than BioGPT, consistent with studies
   showing instruction-tuned models excel at reasoning over biomedical text.

2. MedQA-USMLE:
   Both models struggle, which aligns with published work showing clinical MCQs
   require domain-tuned LLMs; BioGPT performs slightly better.

3. NCBI Disease NER:
   A moderate F1 score (0.449) is expected because the evaluation collapses
   detailed disease labels into a coarse Disease/O mapping.

The overall pattern matches previously published performance trends, supporting
that this framework produces valid, meaningful scientific results.

--------------------------------------------------------------------------------
Datasets
--------------------------------------------------------------------------------

This benchmark uses the following datasets:

- PubMedQA (HuggingFace: pubmed_qa, pqa_labeled)
- MedQA-USMLE (HuggingFace: bigbio/med_qa)
- NCBI Disease Corpus (Xu-Lab Biomedical NLP Benchmarks, test.tsv)

Subsets are used for speed, reproducibility, and poster-friendly evaluation.

--------------------------------------------------------------------------------
Citation
--------------------------------------------------------------------------------

If you use this repository, please cite:

Zhang, X. (2025). A Reproducible Benchmarking Framework for Evaluating 
Foundation Models in Biomedical Text Understanding. ISCB Asia 2025 Poster.

Also cite the underlying datasets:
- PubMedQA: A Dataset for Biomedical Research Question Answering (Jin et al., EMNLP-IJCNLP 2019)
- MedQA-USMLE : A Dataset for Biomedical Research Question Answering (Jin et al., EMNLP-IJCNLP 2019)
- Doğan, R. I., Leaman, R., & Lu, Z. (2014). NCBI disease corpus: a resource for disease name recognition and concept normalization. Journal of biomedical informatics, 47, 1–10. https://doi.org/10.1016/j.jbi.2013.12.006
- Xuguang Ai, Qingyu Chen, Yan(Sawyer) Hu, & Jimin Huang. (2025). BIDS-Xu-Lab/Biomedical-NLP-Benchmarks: v1.0.1 (v1.0.1). Zenodo. https://doi.org/10.5281/zenodo.15002764

--------------------------------------------------------------------------------
Future Work
--------------------------------------------------------------------------------

- Add biomedical models such as PubMedBERT, BioMedLM, SciFive
- Add hallucination detection and robustness metrics
- Add answer consistency and calibration scoring
- Add larger evaluation subsets or full-dataset runs
- Add Gradio-based benchmarking interface

--------------------------------------------------------------------------------
Acknowledgements
--------------------------------------------------------------------------------

This project was developed for biomedical NLP research and the ISCB Asia 2025 
Conference under the support of Westcliff University.
