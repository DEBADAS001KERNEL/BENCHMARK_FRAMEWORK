# load_ THE MMLU dataset from hugging face
from datasets import load_dataset

def load_mmlu(subject="all", split="test"):
    dataset = load_dataset("cais/mmlu", subject if subject != "all" else None, split=split)
    return dataset
