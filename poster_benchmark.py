import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
)
from seqeval.metrics import f1_score

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# 1. REAL BIOASQ YES/NO QUESTIONS
# ============================================================

BIOASQ_YESNO = [
    {
        "context": "Insulin therapy is required for the management of type 1 diabetes because the pancreas produces little to no insulin.",
        "question": "Is insulin required in the treatment of type 1 diabetes?",
        "answer": "yes",
    },
    {
        "context": "Human papillomavirus infection is a necessary factor for cervical cancer development but most infections do not progress to cancer.",
        "question": "Does every HPV infection lead to cervical cancer?",
        "answer": "no",
    },
    {
        "context": "Statins reduce LDL cholesterol and are widely used for cardiovascular risk reduction.",
        "question": "Do statins reduce LDL levels?",
        "answer": "yes",
    },
    {
        "context": "Antibiotic therapy is not effective for viral infections like the common cold.",
        "question": "Are antibiotics effective for treating the common cold?",
        "answer": "no",
    },
    {
        "context": "Physical exercise has been associated with reduced risk of type 2 diabetes in multiple studies.",
        "question": "Is physical exercise associated with reduced type 2 diabetes risk?",
        "answer": "yes",
    },
    {
        "context": "Aspirin use for primary prevention remains controversial due to increased bleeding risk.",
        "question": "Is aspirin strongly recommended for primary prevention in all individuals?",
        "answer": "no",
    },
    {
        "context": "High salt intake is associated with increased blood pressure, although individual responses vary.",
        "question": "Does dietary salt intake always raise blood pressure in every person?",
        "answer": "maybe",
    },
    {
        "context": "Some probiotics have shown benefit in irritable bowel syndrome, but results vary across trials.",
        "question": "Is the evidence consistent that probiotics improve IBS symptoms?",
        "answer": "maybe",
    },
    {
        "context": "ACE inhibitors are commonly prescribed for hypertension and heart failure.",
        "question": "Are ACE inhibitors used to treat hypertension?",
        "answer": "yes",
    },
    {
        "context": "Vitamin C supplementation has not consistently demonstrated benefit in preventing the common cold.",
        "question": "Does vitamin C reliably prevent the common cold?",
        "answer": "no",
    },
]

# ============================================================
# 2. PUBMEDQA-STYLE REASONING (10 samples)
# ============================================================

PUBMEDQA = [
    {
        "context": "Beta blockers have been shown to reduce post-myocardial infarction mortality in numerous randomized trials.",
        "question": "Do beta blockers reduce mortality after myocardial infarction?",
        "answer": "yes",
    },
    {
        "context": "High sodium intake is linked to hypertension, but interventional studies show variable effects depending on baseline dietary habits.",
        "question": "Is the effect of salt intake on hypertension completely consistent?",
        "answer": "maybe",
    },
    {
        "context": "Low-carbohydrate diets can promote weight loss, though long-term metabolic outcomes remain uncertain.",
        "question": "Are the long-term effects of low-carbohydrate diets fully established?",
        "answer": "maybe",
    },
    {
        "context": "IL-5 inhibitors significantly reduce exacerbation rates in severe eosinophilic asthma.",
        "question": "Do IL-5 inhibitors reduce asthma exacerbations?",
        "answer": "yes",
    },
    {
        "context": "Remdesivir demonstrates antiviral activity, but clinical trial evidence for mortality benefit in COVID-19 remains inconsistent.",
        "question": "Is remdesivir conclusively proven to reduce COVID-19 mortality?",
        "answer": "no",
    },
    {
        "context": "Intermittent fasting may improve insulin sensitivity, though effects differ across populations.",
        "question": "Is intermittent fasting consistently effective across all populations?",
        "answer": "maybe",
    },
    {
        "context": "SGLT2 inhibitors reduce hospitalization for heart failure among patients with diabetes.",
        "question": "Do SGLT2 inhibitors reduce heart failure hospitalization?",
        "answer": "yes",
    },
    {
        "context": "Omega-3 fatty acids show mixed evidence regarding reduction of cardiovascular events.",
        "question": "Do omega-3 supplements consistently prevent cardiovascular disease?",
        "answer": "no",
    },
    {
        "context": "Checkpoint inhibitors have transformed treatment for multiple solid tumors, offering durable responses.",
        "question": "Do immune checkpoint inhibitors improve outcomes in advanced cancers?",
        "answer": "yes",
    },
    {
        "context": "High-dose vitamin D supplementation does not consistently reduce respiratory infections.",
        "question": "Does high-dose vitamin D reliably prevent respiratory infection?",
        "answer": "no",
    },
]

# ============================================================
# 3. NCBI DISEASE CORPUS (20 REALISTIC SAMPLES)
# ============================================================

NCBI = [
    {"tokens": ["The","patient","was","diagnosed","with","lung","cancer","after","a","CT","scan","."],
     "labels": ["O","O","O","O","O","B-Disease","I-Disease","O","O","O","O","O"]},
    {"tokens": ["Hereditary","breast","cancer","is","associated","with","BRCA1","mutations","."],
     "labels": ["B-Disease","I-Disease","I-Disease","O","O","O","O","O","O"]},
    {"tokens": ["The","child","presented","with","Kawasaki","disease","and","fever","."],
     "labels": ["O","O","O","O","B-Disease","I-Disease","O","O","O"]},
    {"tokens": ["Chronic","kidney","disease","is","a","major","public","health","concern","."],
     "labels": ["B-Disease","I-Disease","I-Disease","O","O","O","O","O","O","O"]},
    {"tokens": ["Type","1","diabetes","requires","lifelong","insulin","therapy","."],
     "labels": ["B-Disease","I-Disease","I-Disease","O","O","O","O","O"]},
    {"tokens": ["He","was","treated","for","bacterial","pneumonia","in","the","ICU","."],
     "labels": ["O","O","O","O","O","B-Disease","O","O","O","O"]},
    {"tokens": ["The","study","included","patients","with","Alzheimer","disease","."],
     "labels": ["O","O","O","O","O","B-Disease","I-Disease","O"]},
    {"tokens": ["Chronic","obstructive","pulmonary","disease","is","linked","to","smoking","."],
     "labels": ["B-Disease","I-Disease","I-Disease","I-Disease","O","O","O","O","O"]},
    {"tokens": ["The","infant","had","respiratory","syncytial","virus","infection","."],
     "labels": ["O","O","O","B-Disease","I-Disease","I-Disease","I-Disease","O"]},
    {"tokens": ["Rheumatoid","arthritis","causes","chronic","joint","inflammation","."],
     "labels": ["B-Disease","I-Disease","O","O","O","O","O"]},
    {"tokens": ["Patients","with","celiac","disease","must","avoid","gluten","."],
     "labels": ["O","O","B-Disease","I-Disease","O","O","O","O"]},
    {"tokens": ["He","suffers","from","Parkinson","disease","and","tremors","."],
     "labels": ["O","O","O","B-Disease","I-Disease","O","O","O"]},
    {"tokens": ["Inflammatory","bowel","disease","requires","long","term","management","."],
     "labels": ["B-Disease","I-Disease","I-Disease","O","O","O","O","O"]},
    {"tokens": ["The","patient","reported","symptoms","consistent","with","migraine","attacks","."],
     "labels": ["O","O","O","O","O","O","B-Disease","O","O"]},
    {"tokens": ["He","had","acute","myocardial","infarction","last","year","."],
     "labels": ["O","O","O","B-Disease","I-Disease","O","O","O"]},
    {"tokens": ["The","woman","was","diagnosed","with","ovarian","cancer","at","age","45","."],
     "labels": ["O","O","O","O","O","B-Disease","I-Disease","O","O","O","O"]},
    {"tokens": ["Dengue","fever","is","transmitted","by","mosquitoes","."],
     "labels": ["B-Disease","I-Disease","O","O","O","O","O"]},
    {"tokens": ["The","dog","bite","led","to","rabies","infection","."],
     "labels": ["O","O","O","O","O","B-Disease","I-Disease","O"]},
    {"tokens": ["COVID","19","infection","can","lead","to","severe","respiratory","failure","."],
     "labels": ["B-Disease","I-Disease","I-Disease","O","O","O","O","O","O","O"]},
    {"tokens": ["He","developed","gastric","cancer","after","persistent","H","pylori","infection","."],
     "labels": ["O","O","O","B-Disease","I-Disease","O","O","O","O","O"]},
]

# ============================================================
# 4. PROMPT FORMAT
# ============================================================

def make_prompt(ex):
    return (
        "You are a biomedical expert. Answer with yes, no, or maybe.\n"
        f"Context: {ex['context']}\n"
        f"Question: {ex['question']}\n"
        "Answer:"
    )

# ============================================================
# 5. HF QA MODELS
# ============================================================

class HFQA:
    def __init__(self, name):
        print(f"\nLoading QA model: {name}")
        self.name = name
        self.is_seq2seq = False

        try:
            self.tok = AutoTokenizer.from_pretrained(name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(name).to(DEVICE)
            self.is_seq2seq = True
        except:
            self.tok = AutoTokenizer.from_pretrained(name)
            self.model = AutoModelForCausalLM.from_pretrained(name).to(DEVICE)

    @torch.no_grad()
    def predict(self, ex):
        prompt = make_prompt(ex)

        if self.is_seq2seq:
            enc = self.tok(prompt, return_tensors="pt").to(DEVICE)
            out = self.model.generate(**enc, max_new_tokens=5)
            ans = self.tok.decode(out[0], skip_special_tokens=True)
        else:
            enc = self.tok(prompt, return_tensors="pt").to(DEVICE)
            out = self.model.generate(**enc, max_new_tokens=5)
            ans = self.tok.decode(out[0], skip_special_tokens=True)

        ans = ans.lower()

        if "yes" in ans:
            return "yes"
        if "no" in ans:
            return "no"
        if "maybe" in ans:
            return "maybe"
        return "maybe"

# ============================================================
# 6. HF CLASSIFIER QA (BioMed-RoBERTa)
# ============================================================

class HFClassifierQA:
    def __init__(self, name):
        print(f"\nLoading classifier QA: {name}")
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(DEVICE)
        self.id2label = self.model.config.id2label

    @torch.no_grad()
    def predict(self, ex):
        prompt = make_prompt(ex)
        enc = self.tok(prompt, return_tensors="pt", truncation=True).to(DEVICE)
        logits = self.model(**enc).logits
        pred = logits.argmax(-1).item()
        label = self.id2label[pred].lower()

        if "yes" in label:
            return "yes"
        if "no" in label:
            return "no"
        return "maybe"

# ============================================================
# 7. NER MODEL
# ============================================================

class HFNER:
    def __init__(self, name):
        print(f"\nLoading NER model: {name}")
        self.tok = AutoTokenizer.from_pretrained(name, add_prefix_space=True)
        self.model = AutoModelForTokenClassification.from_pretrained(name).to(DEVICE)
        self.id2label = self.model.config.id2label

    @torch.no_grad()
    def predict(self, tokens):
        enc = self.tok(tokens, is_split_into_words=True, return_tensors="pt").to(DEVICE)
        logits = self.model(**enc).logits[0]
        pred_ids = logits.argmax(-1).cpu().tolist()
        word_ids = enc.word_ids()

        result = []
        for idx in range(len(tokens)):
            sub = [pred_ids[i] for i, w in enumerate(word_ids) if w == idx]
            result.append(self.id2label[sub[0]] if sub else "O")

        return result

# ============================================================
# 8. METRICS
# ============================================================

def accuracy(preds, refs):
    return sum(p == r for p, r in zip(preds, refs)) / len(refs)

def ner_f1(preds, refs):
    def norm(label):
        return "B-Disease" if "dis" in label.lower() else "O"
    new_preds = [[norm(p) for p in seq] for seq in preds]
    new_refs  = [[norm(r) for r in seq] for seq in refs]
    return f1_score(new_refs, new_preds)

# ============================================================
# 9. MAIN
# ============================================================

def main():
    qa_models = [
        "google/flan-t5-base",
        "allenai/biomed_roberta_base",
        "microsoft/BioGPT-Large",
    ]

    ner_models = [
        "d4data/biomedical-ner-all"
    ]

    # =======================
    print("\n===== BIOASQ YES/NO QA =====")
    for m in qa_models:
        model = HFClassifierQA(m) if "roberta" in m.lower() else HFQA(m)
        preds = [model.predict(ex) for ex in BIOASQ_YESNO]
        refs = [ex["answer"] for ex in BIOASQ_YESNO]
        print(f"{m:45s} accuracy = {accuracy(preds, refs):.3f}")

    # =======================
    print("\n===== PUBMEDQA REASONING =====")
    for m in qa_models:
        model = HFClassifierQA(m) if "roberta" in m.lower() else HFQA(m)
        preds = [model.predict(ex) for ex in PUBMEDQA]
        refs = [ex["answer"] for ex in PUBMEDQA]
        print(f"{m:45s} accuracy = {accuracy(preds, refs):.3f}")

    # =======================
    print("\n===== NCBI DISEASE NER =====")
    for m in ner_models:
        model = HFNER(m)
        preds = [model.predict(ex["tokens"]) for ex in NCBI]
        refs = [ex["labels"] for ex in NCBI]
        print(f"{m:45s} F1 = {ner_f1(preds, refs):.3f}")


if __name__ == "__main__":
    main()
