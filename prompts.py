# prompts.py

def prompt_pubmedqa(ex):
    return (
        "You are a biomedical reasoning assistant.\n"
        f"Context: {ex['context']}\n"
        f"Question: {ex['question']}\n"
        "Answer: yes, no, or maybe.\n"
    )

def prompt_mednli(ex):
    return (
        "You perform clinical natural language inference.\n"
        f"Premise: {ex['premise']}\n"
        f"Hypothesis: {ex['hypothesis']}\n"
        "Label (entailment/neutral/contradiction):\n"
    )

