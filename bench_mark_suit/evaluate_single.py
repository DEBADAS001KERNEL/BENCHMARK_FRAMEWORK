# Evaluate on a single data
import torch
from makeprompt import build_prompt

def evaluate_example(example, tokenizer, model, device):
    question = example["question"]
    choices = example["choices"]["text"]
    prompt = build_prompt(question, choices)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    marker = "Answer:"
    idx = output_text.find(marker)
    answer = output_text[idx+len(marker):].strip() if idx != -1 else ""
    return answer[0].upper(), prompt, output_text
